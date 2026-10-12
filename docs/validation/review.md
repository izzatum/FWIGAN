# Review scope and architecture decisions

Independent task reviews and a whole-branch review accepted the delivered research-toolkit scope through source commit `d31d28c6572197ad07df3b76f3a7d7f7cad9679a`. The reviewed interfaces cover physical tensor contracts, the raw DeepWave adapter, separate FWI/FWIGAN strategies, gradient isolation, full continuation snapshots, CLI and notebook clients, and the measured CPU benchmark. All material implementation findings were fixed and received scoped re-review. No merge, shared-branch push, or hosted CI execution is implied.

The final scientific correction at `7885070` calibrates the known Marmousi source to the inversion grid. The subsequent `d31d28c` correction changes only an absent-device RNG test fixture: it explicitly mocks CUDA unavailability, matching the opaque-history case it tests. A causal CPU surrogate reproduced the original 800/816-byte mismatch before correction, and the corrected case and five related checks passed. No runtime code, numerical tolerances or real CUDA checks changed.

## Decisions recorded during implementation

1. **Reject float32 coordinates that the native backend would truncate into a different grid cell.** Silently snapping them would violate the physical-coordinate contract and solver parity. This means some fractional-metre acquisitions require a future separately validated adapter.
2. **Calibrate the stored Marmousi known-source coefficient by the two-dimensional cell-area ratio.** The exact `(3/30)^2 = 0.01` factor restores the point-forcing amplitude on the inversion grid while preserving original fine-grid generation and the legacy FWIGAN 10 Hz initializer. Imported coefficients or other source discretizations require their own calibration; finite-difference, PML and near-field discrepancies remain.

The accepted architecture and numerical behavior are supported by [the validation record](validation.md). Four CUDA-specific checks passed, but the corrected full GPU suite and GPU benchmark are blocked by provider provisioning. Full Marmousi equivalence/convergence, hosted CI and a modern backend remain unverified. The complete local review briefs, findings, fixes and decision ledger are preserved outside the checkout at `/workspace/fwigan-validation/`.
