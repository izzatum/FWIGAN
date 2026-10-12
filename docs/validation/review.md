# Review scope and decisions

Independent task reviews and whole-branch review accepted the toolkit's tensor contracts, raw DeepWave adapter, separate FWI/FWIGAN strategies, gradient isolation, continuation snapshots, CLI/notebook clients and CPU benchmark through the calibrated source at `7885070`. Subsequent scoped reviews approved the absent-device test fixture at `d31d28c`, benchmark metadata API correction at `2172137`, AMP calibration guard at `286e036`, field-labelled numerical diagnostics and narrow CUDA velocity-change ULP floor at `7339f63`, and opt-in untimed repeatability diagnostic at `f54f251`. These implementation reviews do **not** approve the failed CUDA equivalent-work comparison or merging the draft PR.

Two scientific decisions remain explicit:

1. Float32 physical coordinates that native DeepWave would truncate into a different grid cell are rejected; silently snapping them would violate geometry parity.
2. The stored 30 m Marmousi known-source coefficient is calibrated by `(3/30)^2 = 0.01`. Fine-grid 7 Hz generation and the FWIGAN 10 Hz initializer remain unchanged. Imported coefficients, surface/colocated geometry and different PML widths need separate validation.

The CUDA velocity-change floor applies to **one recovered float32 model ULP per cell only**; every other benchmark field retains its previous criterion. The actual RTX 4090 benchmark at `7339f63` still failed 44 comparison paths, including 522 of 2,800 velocity-change cells after the floor. The opt-in `f54f251` diagnostic was approved for controlled investigation and passed CPU-focused tests; its GPU run has not occurred. The [validation record](validation.md) gives the raw outcomes and [failure JSON](gpu-benchmark-failure.json) preserves the field diagnostics.

Both current hosted CPU checks at `f54f251` pass 215 tests with four optional CUDA skips each. The earlier full RTX 4090 suite at `7339f63` passed 215/215, including all four CUDA checks. There is no accepted all-workload CUDA timing/memory, full Marmousi convergence or modern-backend comparison. Pull request #1 remains a draft. Full local review briefs and numerical source assessments are retained in the validation workspace outside this checkout.
