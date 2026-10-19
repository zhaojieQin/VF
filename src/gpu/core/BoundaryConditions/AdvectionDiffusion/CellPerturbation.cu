//=======================================================================================
//  This file is part of VirtualFluids. VirtualFluids is free software: you can
//  redistribute it and/or modify it under the terms of the GNU General Public
//  License as published by the Free Software Foundation, either version 3 of
//  the License, or (at your option) any later version.
//
//  SPDX-License-Identifier: GPL-3.0-or-later
//  SPDX-FileCopyrightText: Copyright © VirtualFluids Project contributors, see AUTHORS.md in root folder
//
//! \addtogroup gpu_BoundaryConditions BoundaryConditions
//! \ingroup gpu_core core
//! \{
//=======================================================================================
#include "BoundaryConditions/AdvectionDiffusion/CellPerturbation.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <basics/constants/NumericConstants.h>
#include <cuda_helper/CudaGrid.h>
#include <cuda_helper/CudaIndexCalculation.h>
#include <lbm/MacroscopicQuantities.h>
#include <lbm/constants/D3Q27.h>
#include <logger/Logger.h>

#include "Parameter/Parameter.h"
#include "Utilities/KernelUtilities.h"

namespace vf::gpu {

namespace {

__global__ void cellPerturbation_device(real* distributionsAD, real* distributions, const uint* neighborX,
                                        const uint* neighborY, const uint* neighborZ, const real* velocityX,
                                        const real* velocityY, const real* velocityZ,
                                        unsigned long long numberOfLBnodes, bool isEvenTimestep,
                                        const uint* nodeIndices, const uint* cellIds, uint numberOfPerturbedNodes,
                                        const real* thetaPrime, const real* uPrime, const real* vPrime)
{
    using namespace vf::basics::constant;
    using namespace vf::lbm::dir;

    const uint index = vf::cuda::get1DIndexFrom2DBlock();
    if (index >= numberOfPerturbedNodes)
        return;

    const uint k = nodeIndices[index];
    const uint cell = cellIds[index];
    const ListIndices listIndices(k, neighborX, neighborY, neighborZ);

    // temperature: g_i += w_i theta' (1 + 3 c_i.u), i.e. an equilibrium increment carrying the local velocity
    if (thetaPrime != nullptr) {
        const real dTheta = thetaPrime[cell];
        if (dTheta != c0o1) {
            Distributions27 g = getDistributionReferences27(distributionsAD, numberOfLBnodes, isEvenTimestep);
            real populations[NUMBER_Of_DIRECTIONS];
            getPreCollisionDistribution(populations, g, listIndices);
            const real vx = velocityX[k];
            const real vy = velocityY[k];
            const real vz = velocityZ[k];
            forEachDirection([&](auto direction) {
                populations[direction] +=
                    getWeight<direction>() * dTheta * (c1o1 + c3o1 * getVelocity<direction>(vx, vy, vz));
            });
            setPreCollisionDistribution(g, listIndices, populations);
        }
    }

    // momentum (optional): f_i += w_i 3 rho c_i.u', density unchanged
    if (uPrime != nullptr && vPrime != nullptr) {
        const real du = uPrime[cell];
        const real dv = vPrime[cell];
        if (du != c0o1 || dv != c0o1) {
            Distributions27 f = getDistributionReferences27(distributions, numberOfLBnodes, isEvenTimestep);
            real populations[NUMBER_Of_DIRECTIONS];
            getPreCollisionDistribution(populations, f, listIndices);
            const real rho = c1o1 + vf::lbm::getDensity(populations); // compressible kernels store the deviation
            forEachNonRestDirection([&](auto direction) {
                populations[direction] += getWeight<direction>() * c3o1 * rho * getVelocity<direction>(du, dv, c0o1);
            });
            setPreCollisionDistribution(f, listIndices, populations);
        }
    }
}

template <typename T>
void allocateAndCopy(T** deviceArray, const T* hostArray, size_t count)
{
    checkCudaErrors(cudaMalloc((void**)deviceArray, count * sizeof(T)));
    if (hostArray != nullptr)
        checkCudaErrors(cudaMemcpy(*deviceArray, hostArray, count * sizeof(T), cudaMemcpyHostToDevice));
}

} // namespace

CellPerturbation::CellPerturbation(Parameter* para, int level, real x0, real y0, real z0, real cellDx, real cellDy,
                                   real dz, const int* cellTable, uint nxc, uint nyc, uint nzc, uint numberOfCells)
    : level(level), numberOfCells(numberOfCells)
{
    if (cellDx <= 0 || cellDy <= 0 || dz <= 0)
        throw std::runtime_error("CellPerturbation: cell sizes must be positive");

    // host-side lookup, done once: node coordinates -> cell id
    const auto* parH = para->getParH(level).get();
    std::vector<uint> nodeIndices;
    std::vector<uint> cellIds;
    for (unsigned long long k = 0; k < parH->numberOfNodes; k++) {
        if (!isValidFluidNode(parH->typeOfGridNode[k]))
            continue;
        const long ix = static_cast<long>(std::floor((parH->coordinateX[k] - x0) / cellDx));
        const long iy = static_cast<long>(std::floor((parH->coordinateY[k] - y0) / cellDy));
        const long iz = std::lround((parH->coordinateZ[k] - z0) / dz);
        if (ix < 0 || iy < 0 || iz < 0 || ix >= (long)nxc || iy >= (long)nyc || iz >= (long)nzc)
            continue;
        const int cell = cellTable[(static_cast<size_t>(iz) * nyc + static_cast<size_t>(iy)) * nxc + static_cast<size_t>(ix)];
        if (cell < 0)
            continue;
        if ((uint)cell >= numberOfCells)
            throw std::runtime_error("CellPerturbation: cell id " + std::to_string(cell) + " >= numberOfCells " +
                                     std::to_string(numberOfCells));
        nodeIndices.push_back(static_cast<uint>(k));
        cellIds.push_back(static_cast<uint>(cell));
    }
    numberOfNodes = static_cast<uint>(nodeIndices.size());

    if (numberOfNodes > 0) {
        allocateAndCopy(&nodeIndicesD, nodeIndices.data(), numberOfNodes);
        allocateAndCopy(&cellIdsD, cellIds.data(), numberOfNodes);
    }
    if (numberOfCells > 0) {
        allocateAndCopy<real>(&thetaPrimeD, nullptr, numberOfCells);
        allocateAndCopy<real>(&uPrimeD, nullptr, numberOfCells);
        allocateAndCopy<real>(&vPrimeD, nullptr, numberOfCells);
    }

    VF_LOG_INFO("CellPerturbation: level {} with {} cells covering {} fluid nodes", level, numberOfCells, numberOfNodes);
}

CellPerturbation::~CellPerturbation()
{
    cudaFree(nodeIndicesD);
    cudaFree(cellIdsD);
    cudaFree(thetaPrimeD);
    cudaFree(uPrimeD);
    cudaFree(vPrimeD);
}

void CellPerturbation::apply(Parameter* para, const real* thetaPrime, const real* uPrime, const real* vPrime)
{
    if (numberOfNodes == 0 || numberOfCells == 0)
        return;
    const bool applyTheta = thetaPrime != nullptr;
    const bool applyMomentum = uPrime != nullptr && vPrime != nullptr;
    if (!applyTheta && !applyMomentum)
        return;
    if (applyTheta && !para->getDiffOn())
        throw std::runtime_error("CellPerturbation: temperature perturbation requires the advection-diffusion solver (diffOn)");

    if (applyTheta)
        checkCudaErrors(cudaMemcpy(thetaPrimeD, thetaPrime, numberOfCells * sizeof(real), cudaMemcpyHostToDevice));
    if (applyMomentum) {
        checkCudaErrors(cudaMemcpy(uPrimeD, uPrime, numberOfCells * sizeof(real), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(vPrimeD, vPrime, numberOfCells * sizeof(real), cudaMemcpyHostToDevice));
    }

    // Called between two timesteps: isEvenTimestep has already been swapped, so the pre-collision
    // access below addresses exactly the populations the next collision kernel will read.
    auto parD = para->getParD(level);
    const vf::cuda::CudaGrid grid(parD->numberofthreads, numberOfNodes);
    cellPerturbation_device<<<grid.grid, grid.threads>>>(
        applyTheta ? parD->distributionsAD.f[0] : nullptr, parD->distributions.f[0], parD->neighborX, parD->neighborY,
        parD->neighborZ, parD->velocityX, parD->velocityY, parD->velocityZ, parD->numberOfNodes, parD->isEvenTimestep,
        nodeIndicesD, cellIdsD, numberOfNodes, applyTheta ? thetaPrimeD : nullptr, applyMomentum ? uPrimeD : nullptr,
        applyMomentum ? vPrimeD : nullptr);
    getLastCudaError("cellPerturbation_device execution failed");
}

} // namespace vf::gpu

//! \}
