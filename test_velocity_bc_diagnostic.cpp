// Temporary diagnostic patch for update_velocity_bc
// Add this code to pythonbindings/src/gpu/submodules/simulation.h:164
// Right AFTER line 164: cudaMgr->cudaCopyVeloBC(level);

// Add these lines for diagnostic output:
VF_LOG_INFO("=== DIAGNOSTIC: update_velocity_bc ===");
VF_LOG_INFO("velocityRatio (C++) = {:.10f}", velocityRatio);
VF_LOG_INFO("n_bc = {}", n_bc);

// Print first 3 and last 3 BC nodes
std::vector<int> sample_indices = {0, 1, 2, static_cast<int>(n_bc/2),
                                    static_cast<int>(n_bc-3), static_cast<int>(n_bc-2), static_cast<int>(n_bc-1)};
VF_LOG_INFO("Sample BC nodes (Host side after write):");
for (int idx : sample_indices) {
    if (idx >= 0 && idx < static_cast<int>(n_bc)) {
        VF_LOG_INFO("  Node {:4d}: Vx_host = {:.8f} (LBM)", idx, para->getParH(level)->velocityBC.Vx[idx]);
    }
}

// OPTIONAL: Read back from GPU to verify (requires cudaMemcpy Device->Host)
// This is expensive but necessary to diagnose if GPU buffer differs from host
std::vector<real> Vx_device_readback(n_bc);
cudaMemcpy(Vx_device_readback.data(), para->getParD(level)->velocityBC.Vx,
           sizeof(real) * n_bc, cudaMemcpyDeviceToHost);

VF_LOG_INFO("Sample BC nodes (Device side after cudaCopyVeloBC):");
for (int idx : sample_indices) {
    if (idx >= 0 && idx < static_cast<int>(n_bc)) {
        VF_LOG_INFO("  Node {:4d}: Vx_device = {:.8f} (LBM), diff = {:.2e}",
                    idx, Vx_device_readback[idx],
                    std::abs(Vx_device_readback[idx] - para->getParH(level)->velocityBC.Vx[idx]));
    }
}
