#!/usr/bin/env python3
"""
test_time_varying_inlet.py - Test time-varying velocity BC based on actuator_line setup

This script is based on actuator_line.py but replaces the periodic/precursor inlet
with a time-varying velocity boundary condition.
"""

import numpy as np
from pathlib import Path
from mpi4py import MPI
from pyfluids import basics, gpu, logger, parallel

# ========== Configuration ==========
sim_name = "time_varying_inlet"
config_file = Path(__file__).parent / "Python/actuator_line/configActuatorLine.cfg"
output_path = Path(__file__).parent / Path("output/time_varying_inlet")
output_path.mkdir(parents=True, exist_ok=True)

def get_velocity(z, u_star, kappa, z0):
    """Logarithmic velocity profile"""
    return u_star / kappa * np.log(z / z0 + 1)

# ========== Initialize ==========
logger.Logger.initialize_logger()
communicator = parallel.MPICommunicator.get_instance()

config = basics.ConfigurationFile()
config.load(str(config_file))

para = gpu.Parameter(communicator.get_number_of_processes(), communicator.get_process_id(), config)
bc_factory = gpu.BoundaryConditionFactory()

# ========== Read parameters from config ==========
boundary_layer_height = config.get_float_value("boundaryLayerHeight", 1000)
z0 = config.get_float_value("z0", 0.1)
u_star = config.get_float_value("u_star", 0.4)
kappa = config.get_float_value("vonKarmanConstant", 0.4)
viscosity = config.get_float_value("viscosity", 1.56e-5)
mach = config.get_float_value("Ma", 0.1)
nodes_per_height = config.get_uint_value("nz", 64)
density = config.get_float_value("Density", 1.225)

# Time parameters
t_start_out = config.get_float_value("tStartOut")
t_out = config.get_float_value("tOut")
t_end = config.get_float_value("tEnd")

# ========== Domain setup ==========
length = np.array([6, 4, 1]) * boundary_layer_height
dx = boundary_layer_height / nodes_per_height
velocity = 0.5 * get_velocity(boundary_layer_height, u_star, kappa, z0)
dt = dx * mach / (np.sqrt(3) * velocity)

velocity_ratio = dx / dt
velocity_LB = velocity / velocity_ratio
viscosity_LB = viscosity / (velocity_ratio * dx)
pressure_gradient = u_star * u_star / boundary_layer_height
pressure_gradient_LB = pressure_gradient * (dt * dt) / dx

logger.vf_log_info(f"=== Simulation Parameters ===")
logger.vf_log_info(f"Domain: {length[0]:.1f} x {length[1]:.1f} x {length[2]:.1f} m")
logger.vf_log_info(f"Grid: dx = {dx:.4f} m, nodes_per_height = {nodes_per_height}")
logger.vf_log_info(f"Time: dt = {dt:.6f} s, t_end = {t_end:.1f} s")
logger.vf_log_info(f"Velocity: {velocity:.4f} m/s, LB = {velocity_LB:.6f}")
logger.vf_log_info(f"Viscosity: {viscosity:.6e} m²/s, LB = {viscosity_LB:.8e}")
logger.vf_log_info(f"Pressure gradient: {pressure_gradient:.6e}, LB = {pressure_gradient_LB:.8e}")

# Check stability
omega = 1.0 / (3.0 * viscosity_LB + 0.5)
logger.vf_log_info(f"Omega = {omega:.8f} (stable if 0.5 < omega < 1.99)")

# ========== Setup Parameter object ==========
para.set_output_path(str(output_path))
para.set_output_prefix(sim_name)
para.set_print_files(True)

para.set_forcing(pressure_gradient_LB, 0, 0)
para.set_velocity_LB(velocity_LB)
para.set_viscosity_LB(viscosity_LB)
para.set_velocity_ratio(velocity_ratio)
para.set_viscosity_ratio(dx * dx / dt)
para.set_density_ratio(density)

para.configure_main_kernel(gpu.kernel.compressible.K17CompressibleNavierStokes)

para.set_timestep_start_out(0)  # Start output from beginning
para.set_timestep_out(int(500.0 / dt))  # Output every 100 seconds
para.set_timestep_end(int(t_end / dt))
para.set_is_body_force(True)

# ========== Turbulence model ==========
tm_factory = gpu.TurbulenceModelFactory(para)
tm_factory.read_config_file(config)

# ========== Grid setup ==========
grid_scaling_factory = gpu.GridScalingFactory()
grid_scaling_factory.set_scaling_factory(gpu.GridScaling.ScaleCompressible)

grid_dimensions = gpu.grid_generator.GridDimensions(0, length[0], 0, length[1], 0, length[2], dx)
grid_builder = gpu.grid_generator.MultipleGridBuilderFacade(grid_dimensions)

# IMPORTANT: set_periodic must come BEFORE create_grids
# We disable periodic in X direction to use velocity inlet BC
grid_builder.set_periodic_boundary_condition(False, True, False)
grid_builder.create_grids(communicator.get_process_id())

# ========== Boundary conditions (AFTER create_grids) ==========

# Define logarithmic velocity profile for inlet (returns LBM units)
def velocity_profile_inlet(x, y, z):
    u_phys = get_velocity(max(z, z0), u_star, kappa, z0)
    u_lb = u_phys / velocity_ratio
    return (u_lb, 0.0, 0.0)

# Set velocity BC on inlet (MX = minus-X side)
grid_builder.set_velocity_boundary_condition(
    gpu.SideType.MX,
    velocity_LB,  # reference value (unused when profile is provided)
    0.0,
    0.0,
    velocity_profile=velocity_profile_inlet
)

# Outlet: pressure BC
grid_builder.set_pressure_boundary_condition(gpu.SideType.PX, 0)

# Bottom: stress BC (wall model)
sampling_offset = 2
grid_builder.set_stress_boundary_condition(gpu.SideType.MZ, 0, 0, 1, sampling_offset, z0, kappa, dx)

# Top: slip BC
grid_builder.set_slip_boundary_condition(gpu.SideType.PZ, 0, 0, -1)

# Configure BC kernels
bc_factory.set_velocity_boundary_condition(gpu.VelocityBC.VelocityInterpolatedCompressible)
bc_factory.set_pressure_boundary_condition(gpu.PressureBC.OutflowNonReflective)
bc_factory.set_stress_boundary_condition(gpu.StressBC.StressBounceBackCompressible)
bc_factory.set_slip_boundary_condition(gpu.SlipBC.SlipTurbulentViscosityCompressible)

para.set_outflow_pressure_correction_factor(0.0)

# ========== Initial condition ==========
para.set_initial_condition(
    gpu.initialization_functions.log_law(u_star, kappa, z0, velocity_ratio)
)

# ========== Create simulation ==========
cuda_memory_manager = gpu.CudaMemoryManager(para)

sim = gpu.Simulation(
    para,
    cuda_memory_manager,
    grid_builder.get_grid_builder(),
    bc_factory,
    tm_factory,
    grid_scaling_factory
)

# ========== Get BC node coordinates for time-varying update ==========
logger.vf_log_info("Getting inlet BC node coordinates...")
inlet_coords = sim.get_velocity_bc_coordinates(level=0)
n_bc = inlet_coords.shape[0]
logger.vf_log_info(f"Found {n_bc} inlet BC nodes")

# Prepare base velocity profile for time-varying BC
u_base = np.zeros(n_bc)
v_base = np.zeros(n_bc)
w_base = np.zeros(n_bc)

for i in range(n_bc):
    z = inlet_coords[i, 2]
    u_base[i] = get_velocity(max(z, z0), u_star, kappa, z0)  # physical units (m/s)

logger.vf_log_info(f"Base velocity range: [{u_base.min():.2f}, {u_base.max():.2f}] m/s")

# ========== Time-varying BC parameters ==========
oscillation_amplitude = 0.2  # 20% amplitude
oscillation_period = 100.0   # seconds

logger.vf_log_info(f"Time-varying BC: amplitude={oscillation_amplitude*100:.0f}%, period={oscillation_period:.1f}s")
logger.vf_log_info(f"BC update interval: every timestep")

# ========== Run simulation ==========
sim.init_timers()

for t in range(1, para.get_timestep_end() + 1):
    # Update inlet BC every timestep
    sim_time = t * dt
    oscillation_factor = 1.0 + oscillation_amplitude * np.sin(2.0 * np.pi * sim_time / oscillation_period)
    u_bc = u_base * oscillation_factor
    v_bc = v_base
    w_bc = w_base

    sim.update_velocity_bc(u_bc, v_bc, w_bc, level=0)

    sim.calculate_timestep(t)

sim.finalize()
MPI.Finalize()

logger.vf_log_info("=== Simulation completed successfully ===")
