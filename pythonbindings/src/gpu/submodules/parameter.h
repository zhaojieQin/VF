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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <basics/config/ConfigurationFile.h>
#include <basics/constants/NumericConstants.h>
#include <gpu/core/Parameter/Parameter.h>
#include <gpu/core/PreCollisionInteractor/PreCollisionInteractor.h>
#include <gpu/core/PreProcessor/InitializationFunctions.h>

using namespace vf::basics::constant;

namespace gpu_bindings::parameter
{
    namespace py = pybind11;
    using namespace vf::gpu;

    inline void makeModule(py::module_& parentModule)
    {
        py::class_<Parameter, std::shared_ptr<Parameter>>(parentModule, "Parameter")
            .def(py::init<int, int, std::optional<const vf::basics::ConfigurationFile*>>(), py::arg("number_of_processes"), py::arg("my_ID"), py::arg("config_data"))
            .def(py::init<int, int>(), py::arg("number_of_processes"), py::arg("my_ID"))
            .def(py::init<const vf::basics::ConfigurationFile*>(), py::arg("config_data"))
            .def("set_forcing", &Parameter::setForcing, py::arg("forcing_x"), py::arg("forcing_y"), py::arg("forcing_z"))
            .def("set_quadric_limiters", &Parameter::setQuadricLimiters, py::arg("quadric_limiter_p"), py::arg("quadric_limiter_m"), py::arg("quadric_limiter_d"))
            .def("set_diff_on", &Parameter::setDiffOn, py::arg("is_diff"))
            .def("set_buoyancy_enabled", &Parameter::setBuoyancyEnabled, py::arg("buoyancy_enabled"))
            .def("set_turbulent_prandtl_number", &Parameter::setTurbulentPrandtlNumber, py::arg("turbulent_prandtl_number"))
            .def("set_max_level", &Parameter::setMaxLevel, py::arg("number_of_levels"))
            .def("set_timestep_end", &Parameter::setTimestepEnd, py::arg("tend"))
            .def("get_timestep_end", &Parameter::getTimestepEnd)
            .def("set_timestep_out", &Parameter::setTimestepOut, py::arg("tout"))
            .def("set_timestep_start_out", &Parameter::setTimestepStartOut, py::arg("t_start_out"))
            .def("set_timestep_of_coarse_level", &Parameter::setTimestepOfCoarseLevel, py::arg("timestep"))
            .def("set_calc_turbulence_intensity", &Parameter::setCalcTurbulenceIntensity, py::arg("calc_velocity_and_fluctuations"))
            .def("set_output_path", &Parameter::setOutputPath, py::arg("o_path"))
            .def("set_output_prefix", &Parameter::setOutputPrefix, py::arg("o_prefix"))
            .def("set_print_files", &Parameter::setPrintFiles, py::arg("print_files"))
            .def("set_viscosity_LB", &Parameter::setViscosityLB, py::arg("viscosity"))
            .def("set_velocity_LB", &Parameter::setVelocityLB, py::arg("velocity"))
            .def("set_diffusivity_LB", &Parameter::setDiffusivity, py::arg("diffusivity"))
            .def("set_viscosity_ratio", &Parameter::setViscosityRatio, py::arg("viscosity_ratio"))
            .def("set_velocity_ratio", &Parameter::setVelocityRatio, py::arg("velocity_ratio"))
            .def("set_density_ratio", &Parameter::setDensityRatio, py::arg("density_ratio"))
            .def("set_buoyancy_factor", &Parameter::setBuoyancyFactor, py::arg("buoyancy_factor"))
            .def("set_reference_temperature", &Parameter::setReferenceTemperature, py::arg("reference_temperature"))
            .def("set_devices", &Parameter::setDevices, py::arg("devices"))
            .def("set_max_dev", &Parameter::setMaxDev, py::arg("max_dev"))
            .def("set_is_body_force", &Parameter::setIsBodyForce, py::arg("is_body_force"))
            .def("set_use_streams", &Parameter::setUseStreams, py::arg("use_streams"))
            .def("configure_main_kernel", &Parameter::configureMainKernel, py::arg("kernel"))
            .def("set_AD_kernel", &Parameter::setADKernel, py::arg("ad_kernel"))
            .def("set_outflow_pressure_correction_factor", &Parameter::setOutflowPressureCorrectionFactor, py::arg("correction_factor"))
            .def("set_initial_condition", py::overload_cast<const std::function<std::tuple<real, real, real, real>(real, real, real)>&>(&Parameter::setInitialCondition), py::arg("init_func"))
            .def("set_initial_condition_AD", &Parameter::setInitialConditionAD, py::arg("init_func"))
            .def("set_initial_local_reference_temperature", &Parameter::setInitialLocalReferenceTemperature, py::arg("init_func"))
            .def("add_interactor", &Parameter::addInteractor, py::arg("interactor"))
            .def("add_sampler", &Parameter::addSampler, py::arg("sampler"))
            .def("get_output_path", &Parameter::getOutputPath)
            .def("get_output_prefix", &Parameter::getOutputPrefix)
            .def("get_velocity", &Parameter::getVelocity)
            .def("get_viscosity", &Parameter::getViscosity)
            .def("get_velocity_ratio", &Parameter::getVelocityRatio)
            .def("get_viscosity_ratio", &Parameter::getViscosityRatio)
            .def("get_density_ratio", &Parameter::getDensityRatio)
            .def("get_force_ratio", &Parameter::getForceRatio)
            .def("get_SGS_constant", &Parameter::getSGSConstant)
            .def("get_is_body_force", &Parameter::getIsBodyForce);
    }
} // namespace parameter