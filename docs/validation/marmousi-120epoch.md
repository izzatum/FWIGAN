# Full-grid Marmousi notebook comparison: 120 epochs

Five notebooks completed a fresh 120-epoch run on one NVIDIA A40: original and package FWIGAN with clean and noisy targets, plus a package-only conventional FWI companion. The four GAN runs began with matched physical parameters, critic, normalized target, and Python/NumPy/Torch/CUDA random states. By epoch 120, all four GAN model errors were lower than their shared **478.003 m/s** initial RMSE. The package clean run reached **412.915 m/s**, versus **432.333 m/s** for the original clean run; the noisy pair reached **462.523** and **464.579 m/s**, respectively. These are single measured trajectories at a selected horizon, not a convergence or method-superiority result.

![Coarse truth, shared initial model, and four measured epoch-120 velocity fields on one m/s scale](../assets/marmousi-120epoch-velocity-models.png)

The [SVG velocity models](../assets/marmousi-120epoch-velocity-models.svg) share one physical scale. The [package-minus-original difference maps](../assets/marmousi-120epoch-velocity-differences.png) ([SVG](../assets/marmousi-120epoch-velocity-differences.svg)) use one symmetric m/s scale for both pairs. Directly comparing the final 100 × 310 tensors gives clean difference RMS **138.654 m/s**, maximum absolute **848.687 m/s**; noisy difference RMS **177.647 m/s**, maximum absolute **826.742 m/s**. The models remain materially different even though the starts matched.

## What changed through the longer run

The following are measured coarse-model RMSE values, in m/s. All four rows start at 478.003 m/s at epoch 0. The [metric-history figure](../assets/marmousi-120epoch-metric-histories.png) ([SVG](../assets/marmousi-120epoch-metric-histories.svg)) also shows actual epochs 2, 5, 10, and 15 and connects only evaluated milestones.

| GAN case | Epoch 20 | 40 | 60 | 80 | 100 | 120 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original clean | 478.250 | 478.276 | 477.578 | 459.128 | 442.859 | 432.333 |
| Package clean | 478.536 | 474.851 | 470.182 | 444.728 | 425.180 | 412.915 |
| Original noisy | 482.107 | 488.558 | 492.630 | 485.823 | 481.175 | 464.579 |
| Package noisy | 482.447 | 492.802 | 491.318 | 485.510 | 477.546 | 462.523 |

Clean targets improved steadily after roughly epoch 40. The noisy models initially moved farther from coarse truth, peaked near 493 m/s around epochs 40–60, then recovered by epoch 120. Lower trace mismatch alone would have hidden that reversal. The clean legacy-normalized residual MSE went from **0.028876** at initialization to **0.002001/0.001952** for original/package clean and **0.005773/0.003844** for original/package noisy. For noisy cases this metric compares against the **clean** shared traces, excluding the one-time noise injected into the adversarial target.

| GAN case at epoch 120 | Raw residual MSE (pressure-unit²) | Clean normalized MSE | Model RMSE (m/s) | Source cosine | Source norm / unscaled 7 Hz | Source norm / calibrated coarse 7 Hz | Signed σ | Physical actual / skipped | Critic actual / skipped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Original clean | 12.22475 | 0.002001 | 432.333 | 0.94034 | 0.86179 | 86.179 | 1.0 | 716 / 4 | 4309 / 11 |
| Package clean | 12.24957 | 0.001952 | 412.915 | 0.94292 | 0.86347 | 86.347 | 1.0 | 716 / 4 | 4310 / 10 |
| Original noisy | 11.30671 | 0.005773 | 464.579 | 0.90106 | 0.86096 | 86.096 | +0.04317 | 714 / 6 | 4306 / 14 |
| Package noisy | 11.73921 | 0.003844 | 462.523 | 0.92221 | 0.86104 | 86.104 | +0.03861 | 715 / 5 | 4307 / 13 |

Raw residual MSE compares raw predicted and shared observed pressures. The fine-grid 7 Hz observation source is unscaled; its known coarse inverse-grid coefficient is **0.01 ×** that waveform. GAN source norms are roughly 0.86 of the unscaled reference, or 86 times the calibrated coarse reference. Source cosine measures waveform shape, and neither ratio establishes amplitude identifiability under legacy per-trace normalization. The [source and signed-noise figure](../assets/marmousi-120epoch-source-and-sigma.png) ([SVG](../assets/marmousi-120epoch-source-and-sigma.svg)) separates these references; its waveform preview shows 0.6 s while norms use the full 6 s. Signed σ is in normalized-data units. The noisy target's configured standard deviation is **0.05**, not an assured recovery target.

The package-only conventional FWI companion used the fixed calibrated coarse-grid 7 Hz source and the same raw observations. Its own raw residual MSE fell from **0.00246932** at epoch 0 to **0.00141110 pressure-unit²** at epoch 120; its model RMSE was **477.869 m/s**, compared with **478.003 m/s** initially. There is no original-notebook FWI counterpart, and its raw objective and fixed source make a GAN-versus-FWI convergence ranking inappropriate here.

## Shared experiment and earlier 20-epoch evidence

Fine-grid Marmousi has **1000 × 3100** cells at **3 m** spacing; inversion uses **100 × 310** cells at **30 m**. All-shot acquisition has **30 shots**, **60 receivers**, and **2000 samples** at **0.003 s**. Six five-shot groups and six critic cycles govern the GAN runs. The GAN source initializer is an unscaled 10 Hz waveform. The original notebooks came from `c7ec47ed89ba1380ad7a02bbc4526b93675e76f3`; package source was `788e39f0bc2e363d679961bf056f5c503c184e80`. The environment reported Torch **1.13.1+cu117**, DeepWave **0.0.9**, CUDA build **11.7**, and an NVIDIA A40. Original numerical cells 8, 10, and 12 retained their frozen hashes, recorded in [result.json](marmousi-120epoch/result.json).

The 120-epoch bundle's raw trace SHA-256, `7d3f351e4c14e2e302257988122b9d3bb131f7f3d948ac2b9a96808e4acd21f2`, and shared-observation tensor SHA-256, `20b9d72e97b0ab2e08b02d10534f6665aa9b407bdb011039e9d9c0715ba65db1`, equal those in the [earlier 20-epoch report](marmousi-20epoch.md). This run **started afresh from matched initial states**; it did not continue the earlier checkpoints. At epoch 20 within this 120-epoch run, clean model RMSE was **478.250/478.536 m/s** for original/package versus **479.761/481.715 m/s** in the separate earlier run. Noisy values were **482.107/482.447 m/s** here versus **482.470/482.876 m/s** earlier. The shared raw data permit that comparison, while the different trajectories show why the 120-epoch curves must not be spliced onto the earlier run.

## Measured work, time, and limits

| GAN case | Synchronized training-only time (s) | Whole executed-notebook wall time (s) |
| --- | ---: | ---: |
| Original clean | 572.292 | 597.836 |
| Package clean | 298.004 | 629.952 |
| Original noisy | 567.883 | 591.112 |
| Package noisy | 302.225 | 672.375 |

Training-only timing excludes evaluation and checkpoint I/O. Package training-only time was lower in each observed pair, while whole-notebook wall time was **higher** in each pair. The driver exercised package resume and per-epoch checkpoint writing, so these measurements do not show a whole-workflow speedup. The [work and resource figure](../assets/marmousi-120epoch-work-and-resources.png) ([SVG](../assets/marmousi-120epoch-work-and-resources.svg)) shows actual/skipped optimizer calls and process-live cumulative CUDA allocator peaks, not per-epoch memory increments. FWI training-only time was **66.860 s** and its whole notebook wall time **96.493 s**.

Matched initial conditions do not guarantee matching CUDA trajectories: native reductions, AMP skips, and critic/scheduler paths can diverge. The public evidence covers **120 of the original study's 401 epochs** and does not establish full convergence, general method superiority, source identifiability, or final-field parity. The old notebook's optimizer-only checkpoint does not include its unregistered physical tensors. The package's full resumable checkpoint was **omitted** from the returned bundle because its **674,542,764 compressed bytes** exceed the conservative **400 MiB** allowance (source checkpoint **741,029,932 bytes**). Final physical exports are present in the verified local bundle, but the public small evidence cannot resume this exact run.

## Evidence, reproducibility, and cost

The [result](marmousi-120epoch/result.json), [artifact manifest](marmousi-120epoch/artifact_manifest.json), [figure provenance](marmousi-120epoch/figure-provenance.json), [independent audit summary](marmousi-120epoch/independent-audit-summary.json), and [filtered controller summary](marmousi-120epoch/control-summary.json) identify the run and verified inputs. Published [metrics](marmousi-120epoch/FWIGAN-old-metrics.json) include original clean, [package clean](marmousi-120epoch/FWIGAN-new-metrics.json), [original noisy](marmousi-120epoch/FWIGAN_Noisy-old-metrics.json), [package noisy](marmousi-120epoch/FWIGAN_Noisy-new-metrics.json), and [FWI](marmousi-120epoch/FWI-new-metrics.json). All five [executed notebooks](marmousi-120epoch/FWIGAN-old-executed.ipynb) are also published ([package clean](marmousi-120epoch/FWIGAN-new-executed.ipynb), [original noisy](marmousi-120epoch/FWIGAN_Noisy-old-executed.ipynb), [package noisy](marmousi-120epoch/FWIGAN_Noisy-new-executed.ipynb), [FWI](marmousi-120epoch/FWI-new-executed.ipynb)). Every case completed the requested epoch. [Frozen notebook driver](marmousi-120epoch/notebook_workflow_comparison.py), [figure renderer](marmousi-120epoch/render_workflow_comparison.py), and [GPU controller snapshot](marmousi-120epoch/run_workflow_comparison.py) are retained for audit. The controller imports outside-only helpers and is not a standalone launch command.

The verified local archive is **73,862,706 bytes**, SHA-256 `ca0ecaf8e62c059276e42180541a3d4dc2e34389d5a6b132c751be9ef2fdd38e`; large tensors, full checkpoints, and the archive remain outside Git. The renderer's original and published hashes, original/published SVG hashes, exact PNG hashes, and XML drawing-value comparison are recorded in figure provenance. Published SVGs had line-end whitespace normalized without changing drawing values.

The first A40 attempt ended during environment build after a cloud executor outage; the user terminated pod `h843dkfh09gge5`. It produced **no 120-epoch result**. Its compute estimate is bounded above by **$0.261987** from creation to first verified absence. The completed retry on pod `0q2d9l3fzhxa8v` lasted **3066.692 s**, estimating **$0.502597** at the reported **$0.59/hour**. Together those attempts estimate **at most $0.764584** of compute, excluding unmeasured transport and other charges. The ledger's post-run guarded accounting applies a **$0.659/hour** uplifted rate to observed lifetime; its separate retry reservation used a **$1.84/hour** rate cap plus contingency. Neither is a billed rate. The [budget ledger](marmousi-120epoch/budget-ledger.json), [first-attempt cleanup](marmousi-120epoch/interrupted-attempt-cleanup.json), and [retry cleanup](marmousi-120epoch/independent-cleanup.json) record the accounting and independent proof: each pod GET returned 404 and a fresh account list showed no remaining owned pod. Cost numbers are estimates, not an invoice.
