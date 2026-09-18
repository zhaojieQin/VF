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
//! \addtogroup gpu_Parameter_tests Parameter
//! \ingroup gpu_core_tests core
//! \{
//=======================================================================================
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include "PreProcessor/InitializationFunctions.h"
#include <basics/constants/NumericConstants.h>
#include "basics/tests/testUtilities.h"
#include "tests/testUtilities.h"


TEST(uniform, all)
{
    using namespace vf::basics::constant;
    const real d = c0o1, vx = c1o1, vy = c2o1, vz = c3o1, velRatio = c4o1;
    auto func = vf::gpu::uniform(d, vx*velRatio, vy*velRatio, vz*velRatio, velRatio);
    std::tuple<real, real, real, real> expected {d, vx, vy, vz};
    EXPECT_THAT(func(c0o1, c0o1, c0o1), expected);
    EXPECT_THAT(func(c1o1, c0o1, c0o1), expected);
    EXPECT_THAT(func(c1o1, -c0o1, c0o1), expected);
}


TEST(logLaw, all)
{
    using namespace vf::basics::constant;
    const real frictionVelocity = c1o1, kappa = c1o4, roughnessLength = c1o2, velRatio = c4o1;
    auto func = vf::gpu::logLaw(frictionVelocity*velRatio, kappa, roughnessLength, velRatio);
    std::tuple<real, real, real, real> expected0 {c0o1, c0o1, c0o1, c0o1};
    EXPECT_THAT(func(c0o1, c0o1, c0o1), expected0);
    EXPECT_THAT(func(c1o1, c0o1, c0o1), expected0);
    EXPECT_THAT(func(c1o1, -c0o1, c0o1), expected0);

    std::tuple<real, real, real, real> expected1 {c0o1, c0o1, c0o1, c0o1};
    EXPECT_THAT(func(c0o1, c0o1, c1o2), expected1);
    EXPECT_THAT(func(c1o1, c0o1, c1o2), expected1);
    EXPECT_THAT(func(c1o1, -c0o1, c1o2), expected1);

    std::tuple<real, real, real, real> expected2 {c0o1, std::log(c1o1/roughnessLength)/kappa, c0o1, c0o1};
    EXPECT_THAT(func(c0o1, c0o1, c1o1), expected2);
    EXPECT_THAT(func(c1o1, c0o1, c1o1), expected2);
    EXPECT_THAT(func(c1o1, -c0o1, c1o1), expected2);
}

void TupleNear(std::tuple<real, real, real, real> result, std::tuple<real, real, real, real> expected, real max_abs_error=1e-3)
{
    EXPECT_THAT(std::get<0>(result), RealNear(std::get<0>(expected), max_abs_error));
    EXPECT_THAT(std::get<1>(result), RealNear(std::get<1>(expected), max_abs_error));
    EXPECT_THAT(std::get<2>(result), RealNear(std::get<2>(expected), max_abs_error));
    EXPECT_THAT(std::get<3>(result), RealNear(std::get<3>(expected), max_abs_error));
}

TEST(Zilitenkevich, all)
{
    using namespace vf::basics::constant;

    const real frictionVelocity{0.23}, kappa = c1o4, roughnessLength{1e-4};
    const real inversionHeight{500}, mergingRegionThickness{100}, geostrophicWindSpeed{10}, geostrophicWindDirection{0}, velocityRatio{c1o1};

    auto profile = vf::gpu::profileZilitenkevich(frictionVelocity, kappa, roughnessLength, inversionHeight, mergingRegionThickness, geostrophicWindSpeed, geostrophicWindDirection, velocityRatio);
    std::tuple<real, real, real, real> expected0 {c0o1, c0o1, c0o1, c0o1};
    TupleNear(profile(c0o1, c0o1, c0o1), expected0);
    TupleNear(profile(c1o1, c0o1, c0o1), expected0);
    TupleNear(profile(-c1o1, c0o1, c0o1), expected0);
    TupleNear(profile(c1o1, -c1o1, c0o1), expected0);
    TupleNear(profile(-c1o1, c1o1, c0o1), expected0);

    std::tuple<real, real, real, real> expected1 {c0o1, geostrophicWindSpeed, c0o1, c0o1};

    TupleNear(profile(c0o1, c0o1, inversionHeight), expected1);
    TupleNear(profile(c1o1, c0o1, inversionHeight), expected1);
    TupleNear(profile(-c1o1, c0o1, inversionHeight), expected1);
    TupleNear(profile(c1o1, -c1o1, inversionHeight), expected1);
    TupleNear(profile(-c1o1, c1o1, inversionHeight), expected1);

    TupleNear(profile(c0o1, c0o1, 2*inversionHeight), expected1);
    TupleNear(profile(c1o1, c0o1, 2*inversionHeight), expected1);
    TupleNear(profile(-c1o1, c0o1, 2*inversionHeight), expected1);
    TupleNear(profile(c1o1, -c1o1, 2*inversionHeight), expected1);
    TupleNear(profile(-c1o1, c1o1, 2*inversionHeight), expected1);

}

TEST(UniformAD, all)
{
    using namespace vf::basics::constant;

    auto func = vf::gpu::uniformAD(c1o1);
    EXPECT_THAT(func(c0o1, c0o1, c0o1), c1o1);
    EXPECT_THAT(func(c1o1, c0o1, c0o1), c1o1);
    EXPECT_THAT(func(c0o1, c1o1, c0o1), c1o1);
    EXPECT_THAT(func(c0o1, c0o1, c1o1), c1o1);
}

TEST(ConstantGradient, all)
{
    using namespace vf::basics::constant;

    auto func = vf::gpu::constantGradient(c0o1, c3o1);
    EXPECT_THAT(func(c0o1, c0o1, c0o1), c0o1);
    EXPECT_THAT(func(c1o1, c0o1, c0o1), c0o1);
    EXPECT_THAT(func(c0o1, c1o1, c0o1), c0o1);

    EXPECT_THAT(func(c0o1, c0o1, c1o1), c3o1);
    EXPECT_THAT(func(c1o1, c0o1, c1o1), c3o1);
    EXPECT_THAT(func(c0o1, c1o1, c1o1), c3o1);
}


TEST(profileRampanelliAndZardi, all)
{
    using namespace vf::basics::constant;

    const real surfaceTemperature{0}, inversionBaseHeight {500}, inversionThickness{100}, inversionStrength{5}, lapseRate{1e-4};
    auto func = vf::gpu::profileRampanelliAndZardi(surfaceTemperature, inversionBaseHeight, inversionThickness, inversionStrength, lapseRate);
    const real theta00 = inversionStrength + surfaceTemperature - inversionBaseHeight * lapseRate;
    EXPECT_THAT(func(c0o1, c0o1, c0o1), RealNear(surfaceTemperature, 1e-10));
    EXPECT_THAT(func(c1o1, c0o1, c0o1), RealNear(surfaceTemperature, 1e-10));
    EXPECT_THAT(func(c0o1, c1o1, c0o1), RealNear(surfaceTemperature, 1e-10));

    const real zMid = inversionBaseHeight + inversionThickness*c1o2;
    const real tempMid = surfaceTemperature+inversionStrength*c1o2;
    EXPECT_THAT(func(c0o1, c0o1, zMid), RealNear(tempMid, 1e-2));
    EXPECT_THAT(func(c1o1, c0o1, zMid), RealNear(tempMid, 1e-2));
    EXPECT_THAT(func(c0o1, c1o1, zMid), RealNear(tempMid, 1e-2));
    
    const real zHigh{inversionBaseHeight * c2o1};
    const real tempHigh{theta00 + lapseRate * zHigh}; 
    EXPECT_THAT(func(c0o1, c0o1, zHigh), RealNear(tempHigh, 1e-2));
    EXPECT_THAT(func(c1o1, c0o1, zHigh), RealNear(tempHigh, 1e-2));
    EXPECT_THAT(func(c0o1, c1o1, zHigh), RealNear(tempHigh, 1e-2));
}


TEST(InterpolateProfile, interpolate)
{
    using namespace vf::basics::constant;
    std::vector<real> data{c1o1, c2o1, c3o1};
    std::vector<real> coords{c0o1, c1o1, c2o1};
    EXPECT_THAT(vf::gpu::interpolateProfile(-c1o1, coords, data), testing::Eq(c0o1));
    EXPECT_THAT(vf::gpu::interpolateProfile(c0o1, coords, data), testing::Eq(c1o1));
    EXPECT_THAT(vf::gpu::interpolateProfile(c1o2, coords, data), testing::Eq(c3o1*c1o2));
    EXPECT_THAT(vf::gpu::interpolateProfile(c2o1+c1o2, coords, data), testing::Eq(c3o1+c1o2));
}

TEST(InterpolateProfile, raise)
{
    using namespace vf::basics::constant;
    std::vector<real> coords{c2o1, c1o1, -c1o1};
    std::vector<real> data{c1o1, c2o1, c3o1};
    EXPECT_THROW(vf::gpu::interpolateProfile(-c1o1, coords, data), std::runtime_error);
    std::sort(coords.begin(), coords.end());
    EXPECT_THROW(vf::gpu::interpolateProfile(-c1o1, coords, {}), std::runtime_error);
    EXPECT_THROW(vf::gpu::interpolateProfile(-c1o1, {}, data), std::runtime_error);
}

TEST(InterpolateData, all)
{
    using namespace vf::basics::constant;
    std::vector<real> data{c1o1, c2o1, c3o1};
    std::vector<real> coords{c0o1, c1o1, c2o1};
    auto func = vf::gpu::interpolateDataAD(coords, data);
    EXPECT_THAT(func(c0o1, c0o1, -c1o1), testing::Eq(c0o1));
    EXPECT_THAT(func(c1o1, c0o1, -c1o1), testing::Eq(c0o1));
    EXPECT_THAT(func(c0o1, c1o1, -c1o1), testing::Eq(c0o1));
    EXPECT_THAT(func(c0o1, -c1o1, -c1o1), testing::Eq(c0o1));
    EXPECT_THAT(func(c0o1, c1o1, c0o1), testing::Eq(c1o1));
    EXPECT_THAT(func(c1o1, c0o1, c0o1), testing::Eq(c1o1));
    EXPECT_THAT(func(-c1o1, c0o1, c0o1), testing::Eq(c1o1));
    EXPECT_THAT(func(c1o1, c1o1, c0o1), testing::Eq(c1o1));
    EXPECT_THAT(func(c0o1, c0o1, c1o2), testing::Eq(c3o1*c1o2));
    EXPECT_THAT(func(c1o1, c2o1, c1o2), testing::Eq(c3o1*c1o2));
    EXPECT_THAT(func(c1o1, c0o1, c1o2), testing::Eq(c3o1*c1o2));
    EXPECT_THAT(func(c0o1, c2o1, c2o1+c1o2), testing::Eq(c3o1+c1o2));
    EXPECT_THAT(func(c1o1, c0o1, c2o1+c1o2), testing::Eq(c3o1+c1o2));

}