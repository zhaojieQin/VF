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
           "Update advection-diffusion Dirichlet values (temperature deviation T - T_ref in K) and copy to GPU.")
        .def("set_cell_perturbation_cells", [](Simulation& sim, py::array_t<int, py::array::c_style | py::array::forcecast> cell_table,
                                               real x0, real y0, real z0, real cell_dx, real cell_dy, real dz,
                                               uint number_of_cells, int level) {
            if (cell_table.ndim() != 3)
                throw std::runtime_error("cell_table must be a 3-D int array indexed [iz, iy, ix]");
            const uint nzc = static_cast<uint>(cell_table.shape(0));
            const uint nyc = static_cast<uint>(cell_table.shape(1));
            const uint nxc = static_cast<uint>(cell_table.shape(2));
            return sim.setCellPerturbationCells(level, x0, y0, z0, cell_dx, cell_dy, dz, cell_table.data(), nxc, nyc,
                                                nzc, number_of_cells);
        }, py::arg("cell_table"), py::arg("x0"), py::arg("y0"), py::arg("z0"), py::arg("cell_dx"), py::arg("cell_dy"),
           py::arg("dz"), py::arg("number_of_cells"), py::arg("level") = 0,
           "Cell perturbation method: group fluid nodes into cells once. cell_table[iz, iy, ix] = cell id (-1: none), "
           "ix = floor((x-x0)/cell_dx), iy = floor((y-y0)/cell_dy), iz = round((z-z0)/dz), coordinates in m. "
           "Returns the number of perturbed fluid nodes.")
        .def("apply_cell_perturbation", [](Simulation& sim, py::object theta_prime, py::object u_prime, py::object v_prime) {
            const uint n = sim.getNumberOfPerturbationCells();
            auto as_array = [n](const py::object& obj, const char* name) -> py::array_t<real, py::array::c_style | py::array::forcecast> {
                if (obj.is_none())
                    return py::array_t<real, py::array::c_style | py::array::forcecast>();
                auto arr = py::array_t<real, py::array::c_style | py::array::forcecast>::ensure(obj);
                if (!arr || (uint)arr.size() != n)
                    throw std::runtime_error(std::string("apply_cell_perturbation: ") + name + " must have " +
                                             std::to_string(n) + " values (one per cell)");
                return arr;
            };
            auto theta = as_array(theta_prime, "theta_prime");
            auto u = as_array(u_prime, "u_prime");
            auto v = as_array(v_prime, "v_prime");
            if (u_prime.is_none() != v_prime.is_none())
                throw std::runtime_error("apply_cell_perturbation: give both u_prime and v_prime, or neither");
            sim.applyCellPerturbation(theta_prime.is_none() ? nullptr : theta.data(),
                                      u_prime.is_none() ? nullptr : u.data(),
                                      v_prime.is_none() ? nullptr : v.data());
        }, py::arg("theta_prime"), py::arg("u_prime") = py::none(), py::arg("v_prime") = py::none(),
           "Cell perturbation method: seed one perturbation per cell into the pre-collision distributions, "
           "theta_prime in K (AD concentration units), u_prime/v_prime in lattice units (dx/dt). None skips that part.")
        .def("set_relaxation_zone", [](Simulation& sim, real x_min, real x_max, real y_min, real y_max, real z_max,
                                       real width, real top_width, uint faces, int level) {
            return sim.setRelaxationZone(level, x_min, x_max, y_min, y_max, z_max, width, top_width, faces);
        }, py::arg("x_min"), py::arg("x_max"), py::arg("y_min"), py::arg("y_max"), py::arg("z_max"), py::arg("width"),
           py::arg("top_width") = 0, py::arg("faces") = 15u, py::arg("level") = 0,
           "Lateral relaxation zone: every fluid node within `width` (m) of a selected face is nudged towards a target "
           "state with weight cos^2(pi d / (2 width)). faces bit mask: 1=MX, 2=PX, 4=MY, 8=PY, 16=PZ (uses top_width). "
           "Returns the number of relaxed fluid nodes.")
        .def("get_relaxation_zone_nodes", [](Simulation& sim) {
            const uint n = sim.getNumberOfRelaxationZoneNodes();
            py::array_t<real> coords(std::vector<py::ssize_t>{static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(3)});
            py::array_t<real> weights(static_cast<py::ssize_t>(n));
            sim.getRelaxationZoneNodes(coords.mutable_data(), weights.mutable_data());
            return py::make_tuple(coords, weights);
        }, "Relaxation-zone nodes: (coords (n, 3) in m, ramp weights (n,)), in the order expected by push_relaxation_targets.")
        .def("push_relaxation_targets", [](Simulation& sim, py::object vx, py::object vy, py::object vz, py::object theta) {
            const uint n = sim.getNumberOfRelaxationZoneNodes();
            auto as_array = [n](const py::object& obj, const char* name) -> py::array_t<real, py::array::c_style | py::array::forcecast> {
                auto arr = py::array_t<real, py::array::c_style | py::array::forcecast>::ensure(obj);
                if (!arr || (uint)arr.size() != n)
                    throw std::runtime_error(std::string("push_relaxation_targets: ") + name + " must have " +
                                             std::to_string(n) + " values (one per zone node)");
                return arr;
            };
            auto u = as_array(vx, "vx");
            auto v = as_array(vy, "vy");
            auto w = as_array(vz, "vz");
            py::array_t<real, py::array::c_style | py::array::forcecast> th;
            if (!theta.is_none())
                th = as_array(theta, "theta");
            sim.pushRelaxationTargets(u.data(), v.data(), w.data(), theta.is_none() ? nullptr : th.data());
        }, py::arg("vx"), py::arg("vy"), py::arg("vz"), py::arg("theta") = py::none(),
           "Push the next target frame (velocity in lattice units dx/dt, theta in K as the AD concentration); "
           "the previous frame becomes the t0 frame of the time interpolation.")
        .def("apply_relaxation", [](Simulation& sim, real alpha, real rate, bool relax_velocity, bool relax_temperature) {
            sim.applyRelaxation(alpha, rate, relax_velocity, relax_temperature);
        }, py::arg("alpha"), py::arg("rate"), py::arg("relax_velocity") = true, py::arg("relax_temperature") = true,
           "Relax the zone towards (1-alpha)*t0 + alpha*t1 with rate dt/tau at the face (call between time steps).");
    }
}