#!/usr/bin/env python3
"""
诊断 BC 节点索引顺序问题

检查 velocityBC 数组索引 (nodeIndex) 和物理坐标 (Z) 的对应关系
"""

import sys
import numpy as np
from pathlib import Path

try:
    from pyfluids import gpu, basics, logger, parallel
    print("✓ pyfluids GPU module loaded")
except ImportError as e:
    print(f"✗ Failed to import pyfluids: {e}")
    sys.exit(1)


def main():
    print("\n" + "="*70)
    print("BC Node Order Diagnostic")
    print("="*70)

    # 简化的配置（使用 test_synthetic_velocity_bc.py 中的参数）
    boundary_layer_height = 100.0
    length = np.array([6, 4, 1]) * boundary_layer_height
    nodes_per_height = 64
    dx = boundary_layer_height / nodes_per_height

    velocity_ref = 10.0
    mach = 0.1
    viscosity = 1.56e-5
    dt = dx * mach / (np.sqrt(3) * velocity_ref)
    velocity_LB = velocity_ref * dt / dx
    viscosity_LB = viscosity * dt / (dx * dx)

    print(f"\nDomain: {length[0]}x{length[1]}x{length[2]} m, dx={dx:.3f}m")

    # 初始化 VirtualFluids
    logger.Logger.initialize_logger()
    communicator = parallel.MPICommunicator.get_instance()

    config = basics.ConfigurationFile()
    config_file = Path(__file__).parent / "synthetic_test.cfg"
    config.load(str(config_file))

    para = gpu.Parameter(communicator.get_number_of_processes(),
                        communicator.get_process_id(),
                        config)

    output_path = Path("./output/bc_diagnostic").absolute()
    output_path.mkdir(parents=True, exist_ok=True)
    para.set_output_path(str(output_path))
    para.set_velocity_LB(velocity_LB)
    para.set_viscosity_LB(viscosity_LB)
    para.set_velocity_ratio(dx / dt)
    para.configure_main_kernel(gpu.kernel.compressible.K17CompressibleNavierStokes)

    # 创建网格
    bc_factory = gpu.BoundaryConditionFactory()
    grid_scaling_factory = gpu.GridScalingFactory()
    grid_scaling_factory.set_scaling_factory(gpu.GridScaling.ScaleCompressible)

    grid_dimensions = gpu.grid_generator.GridDimensions(0, length[0], 0, length[1], 0, length[2], dx)
    grid_builder = gpu.grid_generator.MultipleGridBuilderFacade(grid_dimensions)

    grid_builder.set_periodic_boundary_condition(False, True, False)
    grid_builder.create_grids(communicator.get_process_id())

    # 设置速度边界条件（MX 侧，x=0 入口）
    grid_builder.set_velocity_boundary_condition(gpu.SideType.MX, velocity_LB, 0, 0)
    grid_builder.set_pressure_boundary_condition(gpu.SideType.PX, 0)
    grid_builder.set_slip_boundary_condition(gpu.SideType.MZ, 0, 0, 1)
    grid_builder.set_slip_boundary_condition(gpu.SideType.PZ, 0, 0, -1)

    bc_factory.set_velocity_boundary_condition(gpu.VelocityBC.VelocityInterpolatedCompressible)
    bc_factory.set_pressure_boundary_condition(gpu.PressureBC.OutflowNonReflective)
    bc_factory.set_slip_boundary_condition(gpu.SlipBC.SlipCompressible)

    def dummy_init(x, y, z):
        return (1.0, velocity_LB, 0.0, 0.0)
    para.set_initial_condition(dummy_init)

    cuda_memory_manager = gpu.CudaMemoryManager(para)
    tm_factory = gpu.TurbulenceModelFactory(para)
    tm_factory.read_config_file(config)

    sim = gpu.Simulation(para, grid_builder.get_grid_builder(), bc_factory, tm_factory, grid_scaling_factory)

    print("\n" + "="*70)
    print("Step 1: Get BC node coordinates")
    print("="*70)

    coords = sim.get_velocity_bc_coordinates(level=0)
    n_bc = coords.shape[0]

    print(f"\nTotal BC nodes: {n_bc}")
    print(f"X range: [{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}] m")
    print(f"Y range: [{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}] m")
    print(f"Z range: [{coords[:, 2].min():.2f}, {coords[:, 2].max():.2f}] m")

    print("\n" + "="*70)
    print("Step 2: Check nodeIndex vs Z coordinate relationship")
    print("="*70)

    # 检查 nodeIndex 和 Z 坐标的关系
    z_coords = coords[:, 2]

    # 按 Z 坐标排序
    sorted_indices = np.argsort(z_coords)

    print("\n--- First 10 BC nodes (by nodeIndex) ---")
    print("nodeIndex |   X (m)  |   Y (m)  |   Z (m)  | Z-rank")
    print("-" * 60)
    for i in range(min(10, n_bc)):
        z_rank = np.where(sorted_indices == i)[0][0]
        print(f"  {i:4d}    | {coords[i, 0]:8.2f} | {coords[i, 1]:8.2f} | {coords[i, 2]:8.2f} | {z_rank:4d}/{n_bc}")

    print("\n--- Last 10 BC nodes (by nodeIndex) ---")
    print("nodeIndex |   X (m)  |   Y (m)  |   Z (m)  | Z-rank")
    print("-" * 60)
    for i in range(max(0, n_bc-10), n_bc):
        z_rank = np.where(sorted_indices == i)[0][0]
        print(f"  {i:4d}    | {coords[i, 0]:8.2f} | {coords[i, 1]:8.2f} | {coords[i, 2]:8.2f} | {z_rank:4d}/{n_bc}")

    print("\n--- First 10 BC nodes (by Z coordinate) ---")
    print("nodeIndex |   X (m)  |   Y (m)  |   Z (m)  | Z-rank")
    print("-" * 60)
    for z_rank in range(min(10, n_bc)):
        i = sorted_indices[z_rank]
        print(f"  {i:4d}    | {coords[i, 0]:8.2f} | {coords[i, 1]:8.2f} | {coords[i, 2]:8.2f} | {z_rank:4d}/{n_bc}")

    print("\n--- Last 10 BC nodes (by Z coordinate) ---")
    print("nodeIndex |   X (m)  |   Y (m)  |   Z (m)  | Z-rank")
    print("-" * 60)
    for z_rank in range(max(0, n_bc-10), n_bc):
        i = sorted_indices[z_rank]
        print(f"  {i:4d}    | {coords[i, 0]:8.2f} | {coords[i, 1]:8.2f} | {coords[i, 2]:8.2f} | {z_rank:4d}/{n_bc}")

    print("\n" + "="*70)
    print("Step 3: Check if nodeIndex order matches Z coordinate order")
    print("="*70)

    # 检查 nodeIndex 是否按 Z 排序
    is_z_sorted = np.all(np.diff(z_coords) >= 0)
    is_z_reverse_sorted = np.all(np.diff(z_coords) <= 0)

    if is_z_sorted:
        print("✓ BC nodes are sorted by Z coordinate (ascending)")
        print("  → nodeIndex=0 corresponds to lowest Z (bottom)")
        print("  → nodeIndex=n_bc-1 corresponds to highest Z (top)")
    elif is_z_reverse_sorted:
        print("✗ BC nodes are REVERSE sorted by Z coordinate (descending)")
        print("  → nodeIndex=0 corresponds to highest Z (top)")
        print("  → nodeIndex=n_bc-1 corresponds to lowest Z (bottom)")
        print("  ⚠️ This will cause velocity profile inversion!")
    else:
        print("✗ BC nodes are NOT sorted by Z coordinate")
        print("  → nodeIndex has no simple relationship with Z coordinate")
        print("  ⚠️ This will cause scrambled velocity profile!")

        # 计算排序相关性
        spearman_corr = np.corrcoef(np.arange(n_bc), z_coords)[0, 1]
        print(f"\n  Spearman correlation (nodeIndex vs Z): {spearman_corr:.4f}")
        if spearman_corr > 0.9:
            print("  → Strong positive correlation (nearly sorted)")
        elif spearman_corr < -0.9:
            print("  → Strong negative correlation (nearly reverse sorted)")
        else:
            print("  → Weak correlation (random order)")

    print("\n" + "="*70)
    print("Step 4: Simulate velocity assignment")
    print("="*70)

    # 模拟按 Z 坐标赋值的情况
    H = length[2]
    velocity_ratio = dx / dt

    print("\nMethod 1: Assign velocity by nodeIndex (WRONG if nodes not Z-sorted)")
    u_by_index = np.zeros(n_bc)
    for i in range(n_bc):
        z = coords[i, 2]
        z_norm = max(z / H, 0.01)
        u_phys = velocity_ref * (z_norm ** (1.0/7.0))
        u_by_index[i] = u_phys

    print(f"  u[nodeIndex=0] = {u_by_index[0]:.4f} m/s at Z={coords[0, 2]:.2f}m")
    print(f"  u[nodeIndex={n_bc-1}] = {u_by_index[n_bc-1]:.4f} m/s at Z={coords[n_bc-1, 2]:.2f}m")

    print("\nMethod 2: Assign velocity by Z-sorted order (CORRECT)")
    u_by_z = np.zeros(n_bc)
    for z_rank in range(n_bc):
        i = sorted_indices[z_rank]
        z = coords[i, 2]
        z_norm = max(z / H, 0.01)
        u_phys = velocity_ref * (z_norm ** (1.0/7.0))
        u_by_z[i] = u_phys

    # 找到最低和最高 Z 的节点
    i_bottom = sorted_indices[0]
    i_top = sorted_indices[-1]
    print(f"  u[nodeIndex={i_bottom} (bottom)] = {u_by_z[i_bottom]:.4f} m/s at Z={coords[i_bottom, 2]:.2f}m")
    print(f"  u[nodeIndex={i_top} (top)] = {u_by_z[i_top]:.4f} m/s at Z={coords[i_top, 2]:.2f}m")

    # 检查两种方法是否一致
    if np.allclose(u_by_index, u_by_z):
        print("\n✓ Both methods produce identical results")
        print("  → BC node order is correct, no fix needed")
    else:
        print("\n✗ Methods produce DIFFERENT results!")
        print("  → BC node order is incorrect")
        print("\nDifference statistics:")
        diff = u_by_index - u_by_z
        print(f"  Mean difference: {np.mean(diff):.6f} m/s")
        print(f"  Max difference:  {np.max(np.abs(diff)):.6f} m/s")
        print(f"  RMS difference:  {np.sqrt(np.mean(diff**2)):.6f} m/s")

    print("\n" + "="*70)
    print("Recommendation")
    print("="*70)

    if is_z_sorted or np.allclose(u_by_index, u_by_z):
        print("✓ No fix needed - BC nodes are already correctly ordered")
    elif is_z_reverse_sorted:
        print("⚠️ BC nodes are reverse sorted")
        print("\nFix: Reverse the velocity arrays before calling update_velocity_bc:")
        print("  sim.update_velocity_bc(u_bc[::-1], v_bc[::-1], w_bc[::-1], level=0)")
    else:
        print("⚠️ BC nodes are in random order")
        print("\nFix: Sort velocity arrays by Z coordinate:")
        print("  sorted_indices = np.argsort(coords[:, 2])")
        print("  u_bc_sorted = np.zeros_like(u_bc)")
        print("  for z_rank in range(n_bc):")
        print("      i = sorted_indices[z_rank]")
        print("      # Compute velocity for this Z position")
        print("      u_bc_sorted[i] = ...")
        print("  sim.update_velocity_bc(u_bc_sorted, v_bc_sorted, w_bc_sorted, level=0)")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
