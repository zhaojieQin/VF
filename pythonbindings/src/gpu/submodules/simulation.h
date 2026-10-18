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
#include <pybind11/numpy.h>
#include <gpu/core/Calculation/Simulation.h>
#include <gpu/core/Kernel/KernelFactory/KernelFactory.h>
#include <gpu/core/PreProcessor/PreProcessorFactory/PreProcessorFactory.h>
#include <gpu/core/DataStructureInitializer/GridProvider.h>
#include <gpu/core/Parameter/Parameter.h>
#include <gpu/core/Cuda/CudaMemoryManager.h>
#include <gpu/core/Output/DataWriter.h>
#include "gpu/core/BoundaryConditions/BoundaryConditionFactory.h"
#include "gpu/core/TurbulenceModels/TurbulenceModelFactory.h"
#include "gpu/core/GridScaling/GridScalingFactory.h"
#include <logger/Logger.h>
#include <cmath>

namespace gpu_bindings::simulation
{
    namespace py = pybind11;
    using namespace vf::gpu;

    inline void makeModule(py::module_ &parentModule)
    {
        // missing setFactories and setDataWriter, not possible to wrap these functions as long as they take unique ptr arguments
        py::class_<Simulation>(parentModule, "Simulation")
        .def(py::init<std::shared_ptr<Parameter>,
                      std::shared_ptr<GridBuilder>,
                      BoundaryConditionFactory*>(),
             py::arg("parameter"), py::arg("grid_builder"), py::arg("bc_factory"))
        .def(py::init<std::shared_ptr<Parameter>,
                      std::shared_ptr<GridBuilder>,
                      BoundaryConditionFactory*,
                      GridScalingFactory*>(),
             py::arg("parameter"), py::arg("grid_builder"), py::arg("bc_factory"), py::arg("grid_scaling_factory"))

        .def(py::init<std::shared_ptr<Parameter>,
                      std::shared_ptr<CudaMemoryManager>,
                      std::shared_ptr<GridBuilder>,
                      BoundaryConditionFactory*>(),
             py::arg("parameter"), py::arg("memory_manager"), py::arg("grid_builder"), py::arg("bc_factory"))
        .def(py::init<std::shared_ptr<Parameter>,
                      std::shared_ptr<CudaMemoryManager>,
                      std::shared_ptr<GridBuilder>,
                      BoundaryConditionFactory*,
                      GridScalingFactory*>(),
             py::arg("parameter"), py::arg("memory_manager"), py::arg("grid_builder"), py::arg("bc_factory"),
             py::arg("scaling_factory"))

        .def(py::init<std::shared_ptr<Parameter>,
                      std::shared_ptr<GridBuilder>,
                      BoundaryConditionFactory*,
                      std::shared_ptr<TurbulenceModelFactory>>(),
             py::arg("parameter"), py::arg("grid_builder"), py::arg("bc_factory"), py::arg("tm_factory"))
        .def(py::init<std::shared_ptr<Parameter>,
                      std::shared_ptr<GridBuilder>,
                      BoundaryConditionFactory*,
                      std::shared_ptr<TurbulenceModelFactory>,
                      GridScalingFactory*>(),
             py::arg("parameter"), py::arg("grid_builder"), py::arg("bc_factory"), py::arg("tm_factory"),
             py::arg("grid_scaling_factory"))

        
        .def(py::init<std::shared_ptr<Parameter>,
                      std::shared_ptr<CudaMemoryManager>,
                      std::shared_ptr<GridBuilder>,
                      BoundaryConditionFactory*,
                      std::shared_ptr<TurbulenceModelFactory>>(),
             py::arg("parameter"), py::arg("memory_manager"), py::arg("grid_builder"), py::arg("bc_factory"), py::arg("tm_factory"))
        .def(py::init<std::shared_ptr<Parameter>,
                      std::shared_ptr<CudaMemoryManager>,
                      std::shared_ptr<GridBuilder>,
                      BoundaryConditionFactory*,
                      std::shared_ptr<TurbulenceModelFactory>,
                      GridScalingFactory*>(),
             py::arg("parameter"), py::arg("memory_manager"), py::arg("grid_builder"), py::arg("bc_factory"), py::arg("tm_factory"),
             py::arg("grid_scaling_factory"))
        .def("run", &Simulation::run)
        .def("init_timers", &Simulation::initTimers)
        .def("calculate_timestep", &Simulation::calculateTimestep, py::arg("time_step"))
        .def("finalize", &Simulation::finalize)
        .def("addKineticEnergyAnalyzer", &Simulation::addKineticEnergyAnalyzer, py::arg("t_analyse"))
        .def("addEnstrophyAnalyzer", &Simulation::addEnstrophyAnalyzer, py::arg("t_analyse"))
        .def("get_velocity_bc_coordinates", [](Simulation& sim, int level) {
            auto para = sim.getParameter();
            uint n_bc = para->getParH(level)->velocityBC.numberOfBCnodes;

            py::array_t<real> coords(std::vector<py::ssize_t>{static_cast<py::ssize_t>(n_bc), static_cast<py::ssize_t>(3)});
            sim.getVelocityBCCoordinates(level, coords.mutable_data(), n_bc);
            return coords;
        }, py::arg("level") = 0, "Get physical coordinates (x,y,z) in meters of all velocity BC nodes")
        .def("update_velocity_bc", [](Simulation& sim, py::array_t<real> u, py::array_t<real> v, py::array_t<real> w, int level) {
            auto para = sim.getParameter();
            uint n_bc = para->getParH(level)->velocityBC.numberOfBCnodes;

            if (u.size() != n_bc || v.size() != n_bc || w.size() != n_bc) {
                throw std::runtime_error("Velocity array size mismatch: expected " + std::to_string(n_bc) +
                                       " but got u=" + std::to_string(u.size()) +
                                       ", v=" + std::to_string(v.size()) +
                                       ", w=" + std::to_string(w.size()));
            }

            sim.updateVelocityBC(level, u.data(), v.data(), w.data(), n_bc);
        }, py::arg("u"), py::arg("v"), py::arg("w"), py::arg("level") = 0,
           "Update velocity BC with physical velocities (m/s). Automatically converts to LBM units and copies to GPU.")
        .def("get_surface_layer_bc_coordinates", [](Simulation& sim, int level) {
            auto para = sim.getParameter();
            uint n_bc = para->getParH(level)->surfaceLayerBC.numberOfBCnodes;

            py::array_t<real> coords(std::vector<py::ssize_t>{static_cast<py::ssize_t>(n_bc), static_cast<py::ssize_t>(3)});
            sim.getSurfaceLayerBCCoordinates(level, coords.mutable_data(), n_bc);
            return coords;
        }, py::arg("level") = 0, "Get physical coordinates (x,y,z) in meters of all surface layer BC nodes")
        .def("update_surface_layer_heat_flux", [](Simulation& sim, py::array_t<real> hfx, int level) {
            auto para = sim.getParameter();
            uint n_bc = para->getParH(level)->surfaceLayerBC.numberOfBCnodes;

            if ((uint)hfx.size() != n_bc) {
                throw std::runtime_error("hfx array size mismatch: expected " + std::to_string(n_bc) +
                                       " but got " + std::to_string(hfx.size()));
            }

            sim.updateSurfaceLayerHeatFlux(level, hfx.data(), n_bc);
        }, py::arg("hfx"), py::arg("level") = 0,
           "Update surface layer heat flux at each surface layer BC node and copy to GPU. "
           "Values are stored as-is: pass the kinematic heat flux in lattice units, "
           "i.e. H[W/m^2] / (rho*cp) * dt/dx (same normalisation as set_surface_layer_boundary_condition).")
        .def("get_freestream_bc_coordinates", [](Simulation& sim, int level) {
            auto para = sim.getParameter();
            uint n_bc = para->getParH(level)->freestreamBC.numberOfBCnodes;

            py::array_t<real> coords(std::vector<py::ssize_t>{static_cast<py::ssize_t>(n_bc), static_cast<py::ssize_t>(3)});
            sim.getFreestreamBCCoordinates(level, coords.mutable_data(), n_bc);
            return coords;
        }, py::arg("level") = 0, "Get physical coordinates (x,y,z) in meters of all freestream BC nodes")
        .def("update_freestream_bc", [](Simulation& sim, py::array_t<real> u, py::array_t<real> v, py::array_t<real> w, int level) {
            auto para = sim.getParameter();
            uint n_bc = para->getParH(level)->freestreamBC.numberOfBCnodes;

            if (u.size() != n_bc || v.size() != n_bc || w.size() != n_bc) {
                throw std::runtime_error("Freestream velocity array size mismatch: expected " + std::to_string(n_bc) +
                                       " but got u=" + std::to_string(u.size()) +
                                       ", v=" + std::to_string(v.size()) +
                                       ", w=" + std::to_string(w.size()));
            }

            sim.updateFreestreamBC(level, u.data(), v.data(), w.data(), n_bc);
        }, py::arg("u"), py::arg("v"), py::arg("w"), py::arg("level") = 0,
           "Update freestream BC with physical velocities (m/s). Per-node inflow/outflow switching done on GPU.")
        .def("get_ad_dirichlet_bc_coordinates", [](Simulation& sim, int level) {
            auto para = sim.getParameter();
            uint n_bc = para->getParH(level)->AdvectionDiffusionDirichletBC.numberOfBCnodes;

            py::array_t<real> coords(std::vector<py::ssize_t>{static_cast<py::ssize_t>(n_bc), static_cast<py::ssize_t>(3)});
            sim.getADDirichletBCCoordinates(level, coords.mutable_data(), n_bc);
            return coords;
        }, py::arg("level") = 0, "Get physical coordinates (x,y,z) in meters of all advection-diffusion Dirichlet BC nodes")
        .def("update_ad_dirichlet_bc", [](Simulation& sim, py::array_t<real> concentration, int level) {
            auto para = sim.getParameter();
            uint n_bc = para->getParH(level)->AdvectionDiffusionDirichletBC.numberOfBCnodes;

            if ((uint)concentration.size() != n_bc) {
                throw std::runtime_error("AD Dirichlet concentration array size mismatch: expected " + std::to_string(n_bc) +
                                       " but got " + std::to_string(concentration.size()));
            }

            sim.updateADDirichletBC(level, concentration.data(), n_bc);
        }, py::arg("concentration"), py::arg("level") = 0,
           "Update advection-diffusion Dirichlet values (temperature deviation T - T_ref in K) and copy to GPU.");
    }
}