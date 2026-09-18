//=======================================================================================
// ____          ____    __    ______     __________   __      __       __        __
// \    \       |    |  |  |  |   _   \  |___    ___| |  |    |  |     /  \      |  |
//  \    \      |    |  |  |  |  |_)   |     |  |     |  |    |  |    /    \     |  |
//   \    \     |    |  |  |  |   _   /      |  |     |  |    |  |   /  /\  \    |  |
//    \    \    |    |  |  |  |  | \  \      |  |     |   \__/   |  /  ____  \   |  |____
//     \    \   |    |  |__|  |__|  \__\     |__|      \________/  /__/    \__\  |_______|
//      \    \  |    |   ________________________________________________________________
//       \    \ |    |  |  ______________________________________________________________|
//        \    \|    |  |  |         __          __     __     __     ______      _______
//         \         |  |  |_____   |  |        |  |   |  |   |  |   |   _  \    /  _____)
//          \        |  |   _____|  |  |        |  |   |  |   |  |   |  | \  \   \_______
//           \       |  |  |        |  |_____   |   \_/   |   |  |   |  |_/  /    _____  |
//            \ _____|  |__|        |________|   \_______/    |__|   |______/    (_______/
//
//  This file is part of VirtualFluids. VirtualFluids is free software: you can
//  redistribute it and/or modify it under the terms of the GNU General Public
//  License as published by the Free Software Foundation, either version 3 of
//  the License, or (at your option) any later version.
//
//  VirtualFluids is distributed in the hope that it will be useful, but WITHOUT
//  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
//  FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
//  for more details.
//
//  SPDX-License-Identifier: GPL-3.0-or-later
//  SPDX-FileCopyrightText: Copyright © VirtualFluids Project contributors, see AUTHORS.md in root folder
//
//! \addtogroup gpu_PreProcessor InitializationFunctions
//! \ingroup gpu_core core
//! \{
//! \author Henry Korb
//=======================================================================================
#ifndef INIT_FUNCTIONS
#define INIT_FUNCTIONS
#include "UbMath.h"
#include <algorithm>
#include <random>

#include <basics/DataTypes.h>
#include <basics/constants/NumericConstants.h>
#include <functional>
#include <stdexcept>

namespace vf::gpu
{

using InitialCondition = std::function<std::tuple<real, real, real, real>(real x, real y, real z)>;
using InitialConditionAD = std::function<real(real x, real y, real z)>;

inline real interpolateProfile(real coord, std::vector<real> coordinates, std::vector<real> data)
{
    if (!std::is_sorted(coordinates.begin(), coordinates.end()))
        throw std::runtime_error("coordinates need to be sorted in ascending order!");
    const uint nCoords = (uint)coordinates.size();
    if (nCoords != data.size())
        throw std::runtime_error("coords and data need to be the same size!");
    if (nCoords == 0)
        throw std::runtime_error("Coords are empty!");

    uint indexUpper = 1;
    while (indexUpper < nCoords - 1 && coordinates[indexUpper] < coord)
        indexUpper++;

    const real coordLower = coordinates[indexUpper - 1];
    return ub_math::lerp(data[indexUpper - 1], data[indexUpper],
                         (coord - coordLower) / (coordinates[indexUpper] - coordLower));
}
inline InitialCondition uniform(real density, real velocityX, real velocityY, real velocityZ, real velocityRatio)
{
    return [=](real, real, real) {
        return std::tuple { density, velocityX / velocityRatio, velocityY / velocityRatio, velocityZ / velocityRatio };
    };
}

inline real logVelocityProfile(real coordZ, real frictionVelocity, real vonKarmanConstant, real roughnessLength)
{
    return frictionVelocity / vonKarmanConstant * std::log(coordZ / roughnessLength);
}

inline InitialCondition logLaw(real frictionVelocity, real vonKarmanConstant, real roughnessLength, real velocityRatio)
{
    using namespace vf::basics::constant;
    return [=](real, real, real coordZ) {
        coordZ = std::max(coordZ, roughnessLength);

        const real vx = logVelocityProfile(coordZ, frictionVelocity, vonKarmanConstant, roughnessLength) / velocityRatio;
        return std::tuple { c0o1, vx, c0o1, c0o1 };
    };
}

inline InitialCondition perturbedLogLaw(real frictionVelocity, real vonKarmanConstant, real roughnessLength, real length,
                                        real width, real height, real boundaryLayerHeight, real velocityRatio)
{
    using namespace vf::basics::constant;

    return [=](real coordX, real coordY, real coordZ) {
        const real uLogLaw = logVelocityProfile(coordZ, frictionVelocity, vonKarmanConstant, roughnessLength);

        const real relativeX = coordX / length;
        const real relativeY = coordY / width;
        const real relativeZ = coordZ / height;

        const real horizontalPerturbation = std::sin(cPi * c16o1 * relativeX);
        const real verticalPerturbation = std::sin(cPi * c8o1 * relativeZ) / (std::pow(relativeZ, c2o1) + c1o1);
        const real perturbation = c2o1 * horizontalPerturbation * verticalPerturbation;
        const real zScale = std::pow(c1o2 * height - coordZ, c2o1) + c1o1;
        const real density = c0o1;
        const real vx = (uLogLaw + perturbation) * (c1o1 - c1o10 * std::abs(relativeY - c1o2)) / velocityRatio;
        const real vy = perturbation / velocityRatio;
        const real vz = c8o1 * frictionVelocity / vonKarmanConstant *
                        (std::sin(cPi * c8o1 * coordY / boundaryLayerHeight) * std::sin(cPi * c8o1 * relativeZ) +
                         std::sin(cPi * c8o1 * relativeX)) /
                        zScale / velocityRatio;
        return std::tuple { density, vx, vy, vz };
    };
}

//! follows <a href="http://link.springer.com/10.1007/BF00172242"> Zilitenkevich (1989) </a>
//! with parameters for a0, b0 from <a href="https://doi.org/10.1063/1.4922339"> Allaerts and Meyers (2015) </a>
inline InitialCondition profileZilitenkevich(real frictionVelocity, real vonKarmanConstant, real roughnessLength,
                                             real inversionHeight, real mergingRegionThickness, real geostrophicWindSpeed,
                                             real geostrophicWindDirection, real velocityRatio)
{
    using namespace vf::basics::constant;

    return [=](real, real, real coordZ) {
        constexpr real A0 { 4.5 };
        constexpr real a0 { 13.2 };
        constexpr real b0 { 1.57 };
        coordZ = std::max(coordZ, roughnessLength);

        const real zeta = coordZ / inversionHeight;
        const real b0Star = -c3o1 * (c1o2 + c1o4 * b0);
        const real a0Star = A0 - a0;
        const real fU = b0 * zeta + b0Star * zeta * zeta;
        const real fV = a0 * zeta + a0Star * zeta * zeta;
        const real uABL = frictionVelocity / vonKarmanConstant * (std::log(coordZ / roughnessLength) + fU);
        const real vABL = -frictionVelocity / vonKarmanConstant * fV;

        // const real veer = -std::asin(A0/vonKarmanConstant*frictionVelocity/geostrophicWindSpeed);
        // real uABLatInversion = frictionVelocity / vonKarmanConstant * (std::log(inversionHeight/roughnessLength) + (b0 +
        // b0Star)); real vABLatInversion = -frictionVelocity / vonKarmanConstant * (a0 + a0Star); vABL/uABL simplifies to
        const real veer = std::atan2(-A0, (std::log(inversionHeight / roughnessLength) + (b0 + b0Star)));

        const real mergingFactor = std::tanh((zeta - c1o2) * c2o1 * inversionHeight / mergingRegionThickness);
        const real u =
            c1o2 * (uABL * (c1o1 - mergingFactor) + geostrophicWindSpeed * std::cos(veer) * (c1o1 + mergingFactor));
        const real v =
            c1o2 * (vABL * (c1o1 - mergingFactor) + geostrophicWindSpeed * std::sin(veer) * (c1o1 + mergingFactor));

        const real density = c0o1;
        const real vx =
            (u * std::cos(geostrophicWindDirection - veer) - v * std::sin(geostrophicWindDirection - veer)) / velocityRatio;
        const real vy =
            (u * std::sin(geostrophicWindDirection - veer) + v * std::cos(geostrophicWindDirection - veer)) / velocityRatio;
        const real vz = c0o1;
        return std::tuple { density, vx, vy, vz };
    };
}

inline InitialCondition TaylorGreenVortex(real velocityLB)
{
    using namespace vf::basics::constant;
    return [=](real coordX, real coordY, real coordZ) {
        // const real rho =
        //     c3o1 * ((velocityLB * velocityLB) / c16o1 * (std::cos(c2o1 * a * coordX) + std::cos(c2o1 * b * coordY)) *
        //             (std::cos(c2o1 * c * coordZ) + c2o1));
        // const real vx = velocityLB * std::sin(a * coordX) * std::cos(b * coordY) * std::cos(c * coordZ);
        // const real vy = -velocityLB * std::cos(a * coordX) * std::sin(b * coordY) * std::cos(c * coordZ);
        // return std::tuple { rho, vx, vy, c0o1 };
        real a = 1.0;
        real b = 1.0;
        real c = 1.0;

        real rho = 3.0 * ((velocityLB * velocityLB) / 16.0 * (cos(2.0 * a * coordX) + cos(2.0 * b * coordY)) *
                     (cos(2.0 * c * coordZ) + 2.0));
        real vx = velocityLB * sin(a * coordX) * cos(b * coordY) * cos(c * coordZ);
        real vy = -velocityLB * cos(a * coordX) * sin(b * coordY) * cos(c * coordZ);
        real vz = 0.0;

        return std::tuple { rho, vx, vy, vz };
    };
}

inline InitialConditionAD uniformAD(real temperature)
{
    return [=](real, real, real) { return temperature; };
}
inline InitialConditionAD constantGradient(real temperatureSurface, real gradient)
{
    return [=](real, real, real z) { return temperatureSurface + z * gradient; };
}

inline InitialConditionAD twoLayers(real surfaceTemperature, real inversionHeight, real lapseRate)
{
    using namespace vf::basics::constant;
    return [=](real, real, real coordZ) { return surfaceTemperature + lapseRate * std::max(coordZ - inversionHeight, c0o1); };
}
//! follows <a href="https://doi.org/10.1175/1520-0450(2004)043<0925:AMTDTC>2.0.CO;2"> Rampanelli and Zardi (2004) </a>
inline InitialConditionAD profileRampanelliAndZardi(real surfaceTemperature, real inversionBaseHeight,
                                                    real inversionThickness, real inversionStrength, real lapseRate)
{
    using namespace vf::basics::constant;
    return [=](real, real, real coordZ) {
        const real c = c1o3;
        const real l = inversionBaseHeight + c1o2 * inversionThickness;
        const real eta = (coordZ - l) / (c * inversionThickness);
        const real f = (std::tanh(eta) + c1o1) * c1o2;
        // argument cosh becomes too big --> limit size of the argument to (-5,5), outside replace with limiting behaviour
        const real g = eta < -c5o1 ? c0o1 : eta >= c5o1 ? eta : (std::log(c2o1 * std::cosh(eta)) + eta) * c1o2;
        const real b = c * lapseRate * inversionThickness;
        // we interpret the inversion strength as dtheta' not dtheta therefore we use eq 19
        const real a = inversionStrength + b;
        return surfaceTemperature + a * f + b * g;
    };
}
inline InitialConditionAD interpolateDataAD(const std::vector<real>& coordinates, const std::vector<real>& temperatures)
{
    return [=](real, real, real z) { return interpolateProfile(z, coordinates, temperatures); };
}
inline InitialConditionAD addRandomPerturbation(InitialConditionAD unPerturbedFunction, real perturbationStrength,
                                                real maximumHeightOfPerturbation)
{
    return [=](real x, real y, real z) {
        const real unPerturbedValue = unPerturbedFunction(x, y, z);
        static std::uniform_real_distribution<real> distribution(-perturbationStrength, perturbationStrength);
        static std::default_random_engine engine(42);
        if (z < maximumHeightOfPerturbation)
            return unPerturbedValue + distribution(engine);
        return unPerturbedValue;
    };
}
} // namespace vf::gpu

#endif
