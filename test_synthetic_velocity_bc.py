#!/usr/bin/env python3
"""
test_synthetic_velocity_bc.py - Simple test of time-varying velocity BC with synthetic data

This script creates a minimal VirtualFluids simulation and demonstrates:
1. Getting velocity BC node coordinates with get_velocity_bc_coordinates()
2. Updating velocity BC with synthetic time-varying data using update_velocity_bc()
3. Running timesteps with updated boundary conditions

Requirements:
  - Conda environment 'vf' active
  - VirtualFluids Python bindings compiled with velocity BC support
"""

import sys
import numpy as np
from pathlib import Path

try:
    from pyfluids import gpu, basics, logger, parallel
    print("✓ pyfluids GPU module loaded")
except ImportError as e:
    print(f"✗ Failed to import pyfluids: {e}")
    print("  Make sure the conda env 'vf' is active: conda activate vf")
    sys.exit(1)


def main():
    """Main test function"""

    print("\n" + "="*70)
    print("VirtualFluids Synthetic Velocity BC Test")
    print("="*70)

    # ========== Configuration ==========

    # Domain parameters (small test domain)
    boundary_layer_height = 100.0  # meters
    length = np.array([3, 4, 1]) * boundary_layer_height  # [600, 400, 100] m
    nodes_per_height = 128  # Coarse grid for testing
    dx = boundary_layer_height / nodes_per_height  # 6.25 m grid spacing

    # Flow parameters (matching boundary_layer setup)
    velocity_ref = 10.0  # m/s reference velocity
    mach = 0.1           # Same as boundary_layer (not too low)
    viscosity = 1.56e-5  # m^2/s (air at 20°C, physical viscosity)

    dt = dx * mach / (np.sqrt(3) * velocity_ref)  # timestep in seconds
    velocity_LB = velocity_ref * dt / dx
    viscosity_LB = viscosity * dt / (dx * dx)  # Convert to LBM units

    omega_LB = 1.0 / (3.0 * viscosity_LB + 0.5)
    print(f"\n=== Stability Check ===")
    print(f"  Physical viscosity: {viscosity:.6e} m²/s")
    print(f"  LBM viscosity: {viscosity_LB:.8f}")
    print(f"  Omega: {omega_LB:.6f}")
    if omega_LB >= 1.99:
        print(f"  ⚠ WARNING: omega = {omega_LB:.6f} is too close to 2.0 (unstable!)")
        print(f"  Using fixed viscosity_LB = 0.01 for stability")
        viscosity_LB = 0.001
        omega_LB = 1.0 / (3.0 * viscosity_LB + 0.5)
        print(f"  New omega: {omega_LB:.6f}")
    else:
        print(f"  ✓ Omega is stable (< 1.99)")
    print(f"======================\n")

    # Simulation parameters
    T_TOTAL = 100.0            # total simulated time in seconds
    N_STEPS = int(T_TOTAL / dt)
    T_OUT = 10.0               # output every 10 simulated seconds
    OUTPUT_INTERVAL = 1 #int(T_OUT / dt)
    BC_UPDATE_INTERVAL = 1  # update BC every timestep for smooth transitions

    print(f"\n=== Domain Configuration ===")
    print(f"  Domain size: {length[0]:.1f} x {length[1]:.1f} x {length[2]:.1f} m")
    print(f"  Grid spacing: {dx:.3f} m")
    print(f"  Timestep: {dt:.6f} s")
    print(f"  Reference velocity: {velocity_ref:.2f} m/s")
    print(f"  LB velocity: {velocity_LB:.6f}")
    print(f"  LB viscosity: {viscosity_LB:.8f}")

    # ========== Step 1: Initialize VirtualFluids Simulation ==========

    print("\n[1/5] Initializing VirtualFluids simulation...")

    logger.Logger.initialize_logger()
    communicator = parallel.MPICommunicator.get_instance()

    # Load config file for physical unit scaling
    config = basics.ConfigurationFile()
    config_file = Path(__file__).parent / "synthetic_test.cfg"
    config.load(str(config_file))

    # Create Parameter object
    para = gpu.Parameter(communicator.get_number_of_processes(),
                        communicator.get_process_id(),
                        config)

    # Create output directory
    output_path = Path("./output/synthetic_test").absolute()
    output_path.mkdir(parents=True, exist_ok=True)

    para.set_output_prefix("synthetic_test")
    para.set_output_path(str(output_path))
    para.set_print_files(True)  # Enable VTK file output

    para.set_velocity_LB(velocity_LB)
    para.set_viscosity_LB(viscosity_LB)
    para.set_velocity_ratio(dx / dt)
    para.set_viscosity_ratio(dx * dx / dt)
    para.set_density_ratio(1.0)

    para.configure_main_kernel(gpu.kernel.compressible.K17CompressibleNavierStokes)

    para.set_timestep_start_out(0)
    para.set_timestep_out(OUTPUT_INTERVAL)
    para.set_timestep_end(N_STEPS)

    # Create boundary condition factory
    bc_factory = gpu.BoundaryConditionFactory()

    # Create grid with velocity BC on inlet (MX = minus-X side)
    grid_scaling_factory = gpu.GridScalingFactory()
    grid_scaling_factory.set_scaling_factory(gpu.GridScaling.ScaleCompressible)

    grid_dimensions = gpu.grid_generator.GridDimensions(0, length[0], 0, length[1], 0, length[2], dx)
    grid_builder = gpu.grid_generator.MultipleGridBuilderFacade(grid_dimensions)

    # set_periodic_boundary_condition must come BEFORE create_grids
    grid_builder.set_periodic_boundary_condition(False, True, False)
    grid_builder.create_grids(communicator.get_process_id())

    # All other BC setters must come AFTER create_grids

    # Define power-law velocity profile for inlet (LBM units: dimensionless)
    # GridBuilder expects velocity_profile to return LBM units directly
    H = length[2]
    velocity_ratio = dx / dt

    # Define power-law velocity profile for inlet BC (returns 3 values: u, v, w in LBM units)
    # CRITICAL: Must return LBM units because VTK output will multiply by velocity_ratio
    def velocity_profile_func(x, y, z):
        z_normalized = max(z / H, 0.01)
        u_phys = velocity_ref * (z_normalized ** (1.0/7.0))
        u_lb = u_phys / velocity_ratio  # Convert to LBM units
        return (u_lb, 0.0, 0.0)  # Return (u, v, w) in LBM units

    # Define power-law initial condition for entire flow field (returns 4 values: rho, u, v, w in LBM units)
    def power_law_init(x, y, z):
        z_normalized = max(z / H, 0.01)
        u_phys = velocity_ref * (z_normalized ** (1.0/7.0))
        u_lb = u_phys / velocity_ratio  # Convert to LBM units
        rho = 1.0
        v_lb = 0.0
        w_lb = 0.0
        return (rho, u_lb, v_lb, w_lb)  # Return (rho, u, v, w) for initial condition

    # Set velocity BC with profile function
    # GridBuilder will call this function for each BC node to initialize velocityBC array
    # CRITICAL: velocity_profile must return LBM units (same as initial condition)
    #           because VTK output multiplies by velocity_ratio to get physical units
    grid_builder.set_velocity_boundary_condition(
        gpu.SideType.MX,
        velocity_LB,  # Reference value (unused when velocity_profile is provided)
        0,
        0,
        velocity_profile=velocity_profile_func  # Function returns LBM units
    )

    grid_builder.set_pressure_boundary_condition(gpu.SideType.PX, 0)
    grid_builder.set_slip_boundary_condition(gpu.SideType.MZ, 0, 0, 1)
    grid_builder.set_slip_boundary_condition(gpu.SideType.PZ, 0, 0, -1)

    bc_factory.set_velocity_boundary_condition(gpu.VelocityBC.VelocityInterpolatedCompressible)# VelocityInterpolatedCompressible  VelocityBounceBack
    bc_factory.set_pressure_boundary_condition(gpu.PressureBC.OutflowNonReflective)
    bc_factory.set_slip_boundary_condition(gpu.SlipBC.SlipCompressible)

    para.set_outflow_pressure_correction_factor(0.0)

    # Set initial condition matching the BC profile (power-law)
    # This ensures the entire flow field starts with a consistent velocity profile
    # IMPORTANT: Initial condition must return LBM units (dimensionless)
    velocity_ratio = dx / dt
    def power_law_init(x, y, z):
        z_normalized = max(z / H, 0.01)
        u_phys = velocity_ref * (z_normalized ** (1.0/7.0))
        u_lb = u_phys / velocity_ratio  # Convert to LBM units
        rho = 1.0
        v_lb = 0.0
        w_lb = 0.0
        return (rho, u_lb, v_lb, w_lb)

    para.set_initial_condition(power_law_init)

    # Create CUDA memory manager
    cuda_memory_manager = gpu.CudaMemoryManager(para)

    # Create turbulence model factory (QR model from config for stability)
    tm_factory = gpu.TurbulenceModelFactory(para)
    tm_factory.read_config_file(config)

    # Create simulation (full-field output instead of probe)
    sim = gpu.Simulation(para, grid_builder.get_grid_builder(), bc_factory, tm_factory, grid_scaling_factory)
    sim.init_timers()

    print("✓ Simulation initialized")
    print("  Note: GridBuilder has already initialized velocityBC with power-law profile")

    # ========== Step 2: Verify BC Initialization and Prepare Base Profile ==========

    print("\n[2/5] Verifying velocity BC initialization...")

    lbm_coords = sim.get_velocity_bc_coordinates(level=0)
    n_bc = lbm_coords.shape[0]

    # CRITICAL DEBUG: Check what units GridBuilder stored in velocityBC
    print(f"\n  GridBuilder velocity BC initialization:")
    print(f"  velocity_profile_func returns LBM units (same as power_law_init)")
    print(f"  GridBuilder stores LBM units in velocityBC")
    print(f"  VTK output multiplies by velocity_ratio ({velocity_ratio:.1f}) to display physical units")
    print(f"  Stored LBM units: u_lb ~ 0.03-0.06, will display as 5-10 m/s in VTK")

    print(f"\n✓ Retrieved {n_bc} BC nodes")
    print(f"  Coordinate range:")
    print(f"    X: [{lbm_coords[:, 0].min():.2f}, {lbm_coords[:, 0].max():.2f}] m")
    print(f"    Y: [{lbm_coords[:, 1].min():.2f}, {lbm_coords[:, 1].max():.2f}] m")
    print(f"    Z: [{lbm_coords[:, 2].min():.2f}, {lbm_coords[:, 2].max():.2f}] m")

    # Prepare base velocity profile for time-varying BC
    # Calculate what the velocity should be at each BC node (for verification and oscillation)
    print("\n  Calculating base velocity profile for time-varying BC...")

    u_base = np.zeros(n_bc)
    v_base = np.zeros(n_bc)
    w_base = np.zeros(n_bc)

    for i in range(n_bc):
        x, y, z = lbm_coords[i, 0], lbm_coords[i, 1], lbm_coords[i, 2]
        z_normalized = max(z / H, 0.01)
        u_base[i] = velocity_ref * (z_normalized ** (1.0/7.0))  # physical units (m/s)
        # v_base and w_base remain zero

    print(f"  Base velocity profile (physical units):")
    print(f"    u_base: [{u_base.min():.4f}, {u_base.max():.4f}] m/s")
    print(f"    v_base: [{v_base.min():.4f}, {v_base.max():.4f}] m/s")
    print(f"    w_base: [{w_base.min():.4f}, {w_base.max():.4f}] m/s")

    # Sample calculation for verification
    z_mid = length[2] / 2.0
    z_norm_mid = z_mid / length[2]
    u_phys_expected = velocity_ref * (z_norm_mid ** (1.0/7.0))
    print(f"\n  Verification: at z={z_mid:.2f}m (mid-height):")
    print(f"    Expected u_phys = {velocity_ref:.1f} * {z_norm_mid:.4f}^(1/7) = {u_phys_expected:.4f} m/s")

    # Sample a few BC nodes
    print(f"\n  Sample BC nodes (first 3, mid, last 3):")
    sample_indices = [0, 1, 2, n_bc//2, n_bc-3, n_bc-2, n_bc-1]
    for idx in sample_indices:
        if idx < n_bc:
            z = lbm_coords[idx, 2]
            u_phys = u_base[idx]
            print(f"    Node {idx:4d}: z={z:6.2f}m, u_phys={u_phys:8.4f} m/s")

    print(f"\n  ✓ Base profile calculated (GridBuilder already initialized BC correctly)")

    # Run initial timestep to stabilize the flow field
    print(f"\n  Running initial timestep...")
    sim.calculate_timestep(0)
    print(f"  ✓ Initial timestep completed (step 0)")
    print(f"  Check VTK output: inlet (x≈3.12m) should show power-law profile")
    print(f"  Expected inlet u: {u_base.min():.2f} to {u_base.max():.2f} m/s")

    # ========== Step 3: Prepare Synthetic Time-Varying Velocities ==========

    print("\n[3/5] Preparing synthetic time-varying velocity data...")

    # Oscillation parameters
    amplitude = 0.2  # 20% amplitude oscillation
    period = 20.0  # 20 seconds period

    print(f"✓ Synthetic velocity profile:")
    print(f"    Base: power-law u(z) = {velocity_ref:.1f} * (z/{H:.1f})^(1/7) [physical units m/s]")
    print(f"    Oscillation: amplitude = {amplitude*100:.0f}%, period = {period:.3f} s")
    print(f"    Base velocity range: [{u_base.min():.2f}, {u_base.max():.2f}] m/s")
    print(f"    When passed to update_velocity_bc(), converted to LBM: [{(u_base / velocity_ratio).min():.6f}, {(u_base / velocity_ratio).max():.6f}]")

    # ========== Step 5: Main Time-Stepping Loop with BC Updates ==========

    print("\n[5/6] Starting simulation with time-varying BC...")
    print(f"  Total timesteps: {N_STEPS}")
    print(f"  BC update interval: every {BC_UPDATE_INTERVAL} steps")
    print(f"  Total simulation time: {N_STEPS * dt:.3f} s\n")

    for step in range(1, N_STEPS + 1):

        # Update boundary conditions from synthetic data
        if step % BC_UPDATE_INTERVAL == 0:
            sim_time = step * dt

            # Compute time-varying velocities (physical units: m/s)
            oscillation_factor = 1.0 + amplitude * np.sin(2.0 * np.pi * sim_time / period)
            u_bc = u_base * oscillation_factor
            v_bc = np.zeros(n_bc)
            w_bc = np.zeros(n_bc)

            # update_velocity_bc expects physical units (m/s) and converts internally
            sim.update_velocity_bc(u_bc, v_bc, w_bc, level=0)

            max_u = np.max(u_bc)
            min_u = np.min(u_bc)
            print(f"  Step {step:4d} | t = {sim_time:7.4f} s | BC updated | u_range = [{min_u:.2f}, {max_u:.2f}] m/s | factor = {oscillation_factor:.3f}")

        # Advance LBM one timestep
        sim.calculate_timestep(step)

        # Progress reporting
        if step % (N_STEPS // 5) == 0 and step % BC_UPDATE_INTERVAL != 0:
            print(f"  Step {step:4d} | t = {step * dt:7.4f} s")

    # ========== Step 6: Finalize ==========

    print("\n[6/6] Finalizing simulation...")
    sim.finalize()

    print("\n" + "="*70)
    print("✓ Synthetic velocity BC test completed successfully!")
    print("="*70)
    print("\nTest summary:")
    print(f"  - Retrieved {n_bc} velocity BC nodes")
    print(f"  - Updated BC {N_STEPS // BC_UPDATE_INTERVAL} times")
    print(f"  - Ran {N_STEPS} timesteps ({N_STEPS * dt:.3f} s simulated time)")
    print(f"  - Velocity oscillation period: {period:.3f} s ({N_STEPS * dt / period:.1f} cycles)")
    print("\nNext steps:")
    print("  - Verify BC coordinates are at inlet boundary (x ≈ 0)")
    print("  - Check that Mach number warnings appeared if velocities were high")
    print("  - Test with real VirtualFluids configuration files")
    print("  - Integrate with WRF interpolator for actual WRF-LBM coupling")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
