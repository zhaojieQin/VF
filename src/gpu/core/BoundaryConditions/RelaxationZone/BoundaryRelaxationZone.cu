#include "BoundaryConditions/RelaxationZone/BoundaryRelaxationZone.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <basics/constants/NumericConstants.h>
#include <cuda_helper/CudaGrid.h>
#include <cuda_helper/CudaIndexCalculation.h>
#include <lbm/constants/D3Q27.h>
#include <logger/Logger.h>

#include "Parameter/Parameter.h"
#include "Utilities/KernelUtilities.h"

namespace vf::gpu {

namespace {

__global__ void boundaryRelaxation_device(real* distributionsAD, const uint* neighborX, const uint* neighborY,
                                          const uint* neighborZ, const real* velocityX, const real* velocityY,
                                          const real* velocityZ, const real* concentration, real* forceX, real* forceY,
                                          real* forceZ, unsigned long long numberOfLBnodes, bool isEvenTimestep,
                                          const uint* nodeIndices, const real* weights, uint numberOfZoneNodes,
                                          const real* target0, const real* target1, real alpha, real rate,
                                          bool relaxVelocity, bool relaxTemperature)
{
    using namespace vf::basics::constant;
    using namespace vf::lbm::dir;

    const uint index = vf::cuda::get1DIndexFrom2DBlock();
    if (index >= numberOfZoneNodes)
        return;

    const real r = rate * weights[index];
    if (r <= c0o1)
        return;

    const uint k = nodeIndices[index];
    const real beta = c1o1 - alpha;
    const real vx = velocityX[k];
    const real vy = velocityY[k];
    const real vz = velocityZ[k];

    // velocity: body force (acceleration in lattice units) consumed and reset by the next collision
    if (relaxVelocity) {
        const real tx = beta * target0[index] + alpha * target1[index];
        const real ty = beta * target0[numberOfZoneNodes + index] + alpha * target1[numberOfZoneNodes + index];
        const real tz = beta * target0[2 * numberOfZoneNodes + index] + alpha * target1[2 * numberOfZoneNodes + index];
        forceX[k] += r * (tx - vx);
        forceY[k] += r * (ty - vy);
        forceZ[k] += r * (tz - vz);
    }

    // temperature: equilibrium increment of the advection-diffusion distributions (as CellPerturbation)
    if (relaxTemperature && distributionsAD != nullptr) {
        const real tTheta = beta * target0[3 * numberOfZoneNodes + index] + alpha * target1[3 * numberOfZoneNodes + index];
        const real dTheta = r * (tTheta - concentration[k]);
        if (dTheta != c0o1) {
            const ListIndices listIndices(k, neighborX, neighborY, neighborZ);
            Distributions27 g = getDistributionReferences27(distributionsAD, numberOfLBnodes, isEvenTimestep);
            real populations[NUMBER_Of_DIRECTIONS];
            getPreCollisionDistribution(populations, g, listIndices);
            forEachDirection([&](auto direction) {
                populations[direction] +=
                    getWeight<direction>() * dTheta * (c1o1 + c3o1 * getVelocity<direction>(vx, vy, vz));
            });
            setPreCollisionDistribution(g, listIndices, populations);
        }
    }
}

real squaredCosineRamp(real distance, real width)
{
    if (width <= 0 || distance >= width)
        return 0;
    const real pi = real(3.14159265358979323846);
    const real c = std::cos(pi * std::max(distance, real(0)) / (real(2) * width));
    return c * c;
}

} // namespace

BoundaryRelaxationZone::BoundaryRelaxationZone(Parameter* para, int level, real xMin, real xMax, real yMin, real yMax,
                                               real zMax, real width, real topWidth, uint faces)
    : level(level)
{
    if (width <= 0 && !((faces & Face::PZ) && topWidth > 0))
        throw std::runtime_error("BoundaryRelaxationZone: zone width must be positive");

    const auto* parH = para->getParH(level).get();
    std::vector<uint> nodeIndices;
    for (unsigned long long k = 0; k < parH->numberOfNodes; k++) {
        if (!isValidFluidNode(parH->typeOfGridNode[k]))
            continue;
        const real x = parH->coordinateX[k];
        const real y = parH->coordinateY[k];
        const real z = parH->coordinateZ[k];
        real w = 0;
        if (faces & Face::MX)
            w = std::max(w, squaredCosineRamp(x - xMin, width));
        if (faces & Face::PX)
            w = std::max(w, squaredCosineRamp(xMax - x, width));
        if (faces & Face::MY)
            w = std::max(w, squaredCosineRamp(y - yMin, width));
        if (faces & Face::PY)
            w = std::max(w, squaredCosineRamp(yMax - y, width));
        if (faces & Face::PZ)
            w = std::max(w, squaredCosineRamp(zMax - z, topWidth));
        if (w <= 0)
            continue;
        nodeIndices.push_back(static_cast<uint>(k));
        weights.push_back(w);
        coordinatesX.push_back(x);
        coordinatesY.push_back(y);
        coordinatesZ.push_back(z);
    }
    numberOfNodes = static_cast<uint>(nodeIndices.size());

    if (numberOfNodes > 0) {
        checkCudaErrors(cudaMalloc((void**)&nodeIndicesD, numberOfNodes * sizeof(uint)));
        checkCudaErrors(cudaMemcpy(nodeIndicesD, nodeIndices.data(), numberOfNodes * sizeof(uint), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMalloc((void**)&weightsD, numberOfNodes * sizeof(real)));
        checkCudaErrors(cudaMemcpy(weightsD, weights.data(), numberOfNodes * sizeof(real), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMalloc((void**)&target0D, 4 * size_t(numberOfNodes) * sizeof(real)));
        checkCudaErrors(cudaMalloc((void**)&target1D, 4 * size_t(numberOfNodes) * sizeof(real)));
        checkCudaErrors(cudaMemset(target0D, 0, 4 * size_t(numberOfNodes) * sizeof(real)));
        checkCudaErrors(cudaMemset(target1D, 0, 4 * size_t(numberOfNodes) * sizeof(real)));
    }

    VF_LOG_INFO("BoundaryRelaxationZone: level {} with {} fluid nodes (width {}, top width {}, faces mask {})", level,
                numberOfNodes, width, topWidth, faces);
}

BoundaryRelaxationZone::~BoundaryRelaxationZone()
{
    cudaFree(nodeIndicesD);
    cudaFree(weightsD);
    cudaFree(target0D);
    cudaFree(target1D);
}

void BoundaryRelaxationZone::pushTargets(const real* vx, const real* vy, const real* vz, const real* theta)
{
    if (numberOfNodes == 0)
        return;
    // the older frame becomes the new frame's buffer: t0 <- t1, t1 <- new
    std::swap(target0D, target1D);
    const size_t n = numberOfNodes;
    checkCudaErrors(cudaMemcpy(target1D, vx, n * sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(target1D + n, vy, n * sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(target1D + 2 * n, vz, n * sizeof(real), cudaMemcpyHostToDevice));
    if (theta != nullptr) {
        checkCudaErrors(cudaMemcpy(target1D + 3 * n, theta, n * sizeof(real), cudaMemcpyHostToDevice));
        hasTemperatureTarget = true;
    }
    numberOfPushedFrames++;
}

void BoundaryRelaxationZone::apply(Parameter* para, real alpha, real rate, bool relaxVelocity, bool relaxTemperature)
{
    if (numberOfNodes == 0 || rate <= 0 || (!relaxVelocity && !relaxTemperature))
        return;
    if (numberOfPushedFrames < 2)
        throw std::runtime_error("BoundaryRelaxationZone::apply: push two target frames first");
    if (relaxVelocity && !para->getIsBodyForce())
        throw std::runtime_error("BoundaryRelaxationZone: velocity relaxation needs the body force (setIsBodyForce(true))");
    if (relaxTemperature && (!para->getDiffOn() || !hasTemperatureTarget))
        throw std::runtime_error("BoundaryRelaxationZone: temperature relaxation needs the advection-diffusion solver "
                                 "and a theta target");
    alpha = std::min(std::max(alpha, real(0)), real(1));

    // Called between two time steps: isEvenTimestep has already been swapped (see CellPerturbation::apply)
    auto parD = para->getParD(level);
    const vf::cuda::CudaGrid grid(parD->numberofthreads, numberOfNodes);
    boundaryRelaxation_device<<<grid.grid, grid.threads>>>(
        relaxTemperature ? parD->distributionsAD.f[0] : nullptr, parD->neighborX, parD->neighborY, parD->neighborZ,
        parD->velocityX, parD->velocityY, parD->velocityZ, relaxTemperature ? parD->concentration : nullptr,
        parD->forceX_SP, parD->forceY_SP, parD->forceZ_SP, parD->numberOfNodes, parD->isEvenTimestep, nodeIndicesD,
        weightsD, numberOfNodes, target0D, target1D, alpha, rate, relaxVelocity, relaxTemperature);
    getLastCudaError("boundaryRelaxation_device execution failed");
}

} // namespace vf::gpu
