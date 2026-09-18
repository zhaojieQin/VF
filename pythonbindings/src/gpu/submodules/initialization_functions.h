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
//! \author Henry Korb
//=======================================================================================
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <basics/config/ConfigurationFile.h>
#include <basics/constants/NumericConstants.h>
#include <gpu/core/Parameter/Parameter.h>
#include <gpu/core/PreCollisionInteractor/PreCollisionInteractor.h>
#include <gpu/core/PreProcessor/InitializationFunctions.h>

using namespace vf::basics::constant;

namespace gpu_bindings::initialization_functions
{
    namespace py = pybind11;
    using namespace vf::gpu;
    inline void makeModule(py::module_& parentModule)
    {
        py::module_ mod = parentModule.def_submodule("initialization_functions");
        mod.def("uniform", uniform, py::arg("density"), py::arg("velocity_x"), py::arg("velocity_y"), py::arg("velocity_z"), py::arg("velocity_ratio"));
        mod.def("log_law", logLaw, py::arg("friction_velocity"), py::arg("von_karman_constant"), py::arg("roughness_length"), py::arg("velocity_ratio"));
        mod.def("perturbed_log_law", perturbedLogLaw, py::arg("friction_velocity"), py::arg("von_karman_constant"), py::arg("roughness_length"), py::arg("length"), py::arg("width"), py::arg("height"), py::arg("boundary_layer_height"), py::arg("velocity_ratio"));
        mod.def("profile_Zilitenkevich", profileZilitenkevich, py::arg("friction_velocity"), py::arg("von_karman_constant"), py::arg("roughness_length"), py::arg("inversion_height"), py::arg("merging_region_thickness"), py::arg("geostrophic_wind_speed"), py::arg("geostrophic_wind_direction"), py::arg("velocity_ratio"));
        mod.def("taylor_green_vortex", TaylorGreenVortex, py::arg("velocity_LB"));
        mod.def("uniform_AD", uniformAD, py::arg("temperature"));
        mod.def("constant_gradient", constantGradient, py::arg("temperature_surface"), py::arg("gradient"));
        mod.def("two_layers", twoLayers, py::arg("temperature_surface"), py::arg("inversion_height"), py::arg("gradient"));
        mod.def("profile_Rampanelli_and_Zardi", profileRampanelliAndZardi, py::arg("surface_temperature"), py::arg("inversion_base_height"), py::arg("inversion_thickness"), py::arg("inversion_strength"), py::arg("lapse_rate"));
        mod.def("interpolate_data_AD", [](const py::array_t<real>& coordinates, const py::array_t<real>& temperatures){
            auto arr_to_vec = [](const py::array_t<real>& arr)
            {
                real* ptr = (real*)arr.request().ptr;
                return std::vector<real>(ptr, ptr+arr.request().size);
            };
            return interpolateDataAD(arr_to_vec(coordinates), arr_to_vec(temperatures));
        }, py::arg("coordinates"), py::arg("temperatures"));
        mod.def("add_random_perturbation", addRandomPerturbation, py::arg("unperturbed_function"), py::arg("perturbation_strength"), py::arg("maximum_height_of_perturbation"));
    }
}