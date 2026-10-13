# Full-grid 120-epoch Marmousi GPU notebook workflow comparison

**Evidence class:** Full-grid 120-epoch Marmousi GPU. This is the completed full-grid, 120-epoch Marmousi GPU comparison.
**Bundle:** `/workspace/fwigan-validation/120epoch-gpu-retry-20261013T0753Z.artifacts`. **Shared traces:** [30, 60, 2000] [shots, receivers, samples]; inverse grid 100 × 310 cells at 30 × 30 m; dt=0.003 s. Last completed epoch 120; completed epoch values by case: [120].
**Shared observation SHA-256:** `20b9d72e97b0ab2e08b02d10534f6665aa9b407bdb011039e9d9c0715ba65db1`.

Workflow status: `completed`; requested horizon: `120`; device: `cuda`; baseline notebook commit: `c7ec47ed89ba1380ad7a02bbc4526b93675e76f3`.
Preparation: `{"path": "/tmp/fwigan-whole-workflow/workflow/shared_observations.pt", "sha256": "20b9d72e97b0ab2e08b02d10534f6665aa9b407bdb011039e9d9c0715ba65db1", "data_fingerprint": "e6a97fcfe6278cba0acddd75d5890800eafe68e45927fbd61ae1e4a6d496dfa8", "shape": [30, 60, 2000], "fine_shape": [1000, 3100], "coarse_shape": [100, 310], "seconds": 87.01930181006901, "simulation_mode": "all_shots", "shot_block_probe": null, "raw_sha256": "7d3f351e4c14e2e302257988122b9d3bb131f7f3d948ac2b9a96808e4acd21f2", "raw_strides": [60, 1, 1800], "peak_memory": {"allocated": 0, "reserved": 2797600768, "peak_allocated": 2756718080, "peak_reserved": 2797600768}, "parent_memory_after_release": {"allocated": 0, "reserved": 0}}`.
Initial preflight/full-run pair checks: `FWIGAN_preflight=matched`, `FWIGAN=matched`, `FWIGAN_Noisy_preflight=matched`, `FWIGAN_Noisy=matched`.
Notebook execution cases: `FWIGAN-new=completed`, `FWIGAN-old=completed`, `FWIGAN_Noisy-new=completed`, `FWIGAN_Noisy-old=completed`, `FWI-new=completed`.

| Case | Last epoch | Raw residual MSE (pressure-unit²) | Clean normalized residual MSE | Model RMSE (m/s) | Source cosine | Source norm / unscaled 7 Hz | Source norm / calibrated coarse | Signed σ | Physical actual / skipped | Critic actual / skipped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Original clean | 120 | 12.22 | 0.002001 | 432.3 m/s | 0.9403 | 0.8618 | 86.18 | 1 | 716 / 4 | 4309 / 11 |
| Package clean | 120 | 12.25 | 0.001952 | 412.9 m/s | 0.9429 | 0.8635 | 86.35 | 1 | 716 / 4 | 4310 / 10 |
| Original noisy | 120 | 11.31 | 0.005773 | 464.6 m/s | 0.9011 | 0.861 | 86.1 | 0.04317 | 714 / 6 | 4306 / 14 |
| Package noisy | 120 | 11.74 | 0.003844 | 462.5 m/s | 0.9222 | 0.861 | 86.1 | 0.03861 | 715 / 5 | 4307 / 13 |

Raw residual MSE compares raw predicted and shared observed pressures. `normalized_clean_mse` compares legacy-normalized predicted and **clean** shared observed traces even for noisy cases; it excludes the once-injected noisy adversarial target. Sigma is signed and in normalized-data units; 0.05 is the configured observation-noise standard deviation, not an assured recovery target.

| Case | Synchronized training-only seconds | Notebook wall time (s) |
| --- | ---: | ---: |
| Original clean | 572.3 | 597.8 |
| Package clean | 298 | 630 |
| Original noisy | 567.9 | 591.1 |
| Package noisy | 302.2 | 672.4 |

Training-only seconds exclude evaluation and checkpoints; notebook wall time includes the executed notebook cells and their lifecycle/I/O. A shorter training-only bar does not establish a shorter whole-notebook run.

### Conventional FWI companion

The package-only FWI run reached epoch 120 on the same shared raw data with the fixed calibrated inverse-grid 7 Hz source. Raw residual MSE: 0.001411 pressure-unit²; coarse model RMSE: 477.9 m/s; training-only time: 66.86 s; notebook wall time: 96.49 s. No original-notebook FWI counterpart was run, and this companion is not an adversarial-versus-FWI convergence verdict.

The fine-grid 7 Hz observation source is unscaled; the known coarse inverse-grid source is 0.01 × that waveform. Both amplitude references are identified in the source figure. FWIGAN begins from an unscaled 10 Hz initializer. Source cosine correlation measures shape, while the two norm ratios measure amplitude against different references. The waveform-shape panel shows the first 0.6 s for pulse readability; amplitude norms use the full 6 s waveform.

Figures generated from actual bundle arrays and metric rows: `velocity-models.png`/`.svg`, `velocity-differences.png`/`.svg`, `metric-histories.png`/`.svg`, `source-and-sigma.png`/`.svg`, `work-and-resources.png`/`.svg`.
Velocity maps use one common m/s scale; difference maps use one symmetric m/s scale. Metric lines connect observed milestone values only. Cumulative CUDA allocator peaks are process-live snapshots, not per-epoch increments. Training time excludes evaluation and checkpoint I/O when present in both old and package rows.

These are selected-horizon outcomes, not full 401-epoch convergence or general method superiority. Old and package paths need not be bit-identical on CUDA; AMP skips, native reductions and scheduler behavior may differ. No source-identifiability claim follows from a normalized-trace adversarial objective.

Figure-provenance JSON: `figure-provenance.json`.
