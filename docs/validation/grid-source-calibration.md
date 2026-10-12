# Raw source coefficient calibration across grids

DeepWave 0.0.9 injects a point-source coefficient into one grid cell as `coefficient × vp² × inner_dt²`, without dividing by cell area. In 2D, equal coefficients therefore represent integrated forcing proportional to cell area. A wavelet observed on the original 3 m Marmousi grid needs an inverse-grid coefficient `source_30m = source_3m × (3/30)² = 0.01 × source_3m` to represent matching point forcing on the 30 m grid. The 7 Hz frequency and peak time remain unchanged.

Marmousi preparation preserves the exact original unscaled 7 Hz fine-grid observation generation. Its returned `ObservationData.source` stores the known coefficient wavelet for the inversion grid, with the deterministic 0.01 factor, for both FWI and FWIGAN. Conventional FWI fixes that known coefficient. FWIGAN's `initial_source` retains the original unscaled 10 Hz guess, so this correction does not change its source training path, noise draws or normalization. No empirical fitted amplitude is used in production. Source and resulting pressure amplitudes here are in arbitrary units. Same-grid tiny preparation needs factor one; imported observation artifacts are not automatically recalibrated and must provide a correct source coefficient for their inversion grid.

The backend evidence is `scalar/scalar.py` constructing `vp2dt2` and resampling source time samples, then the pinned C++ `scalar_cpu.cpp` `add_sources` adding `source_amplitudes[s] * model[i]` to one cell. No spatial area normalization occurs in that path. The executable probe uses the native pinned backend; the regression test separately propagates the actual coefficient returned by package preparation through the public forward adapter.

## Reproduction and measured evidence

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 timeout 60s python examples/calibrate_grid_source.py --output runs/grid-source-calibration.json
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q tests/test_cli.py -k marmousi
```

The committed [actual CPU result](grid-source-calibration.json) includes Python/Torch/DeepWave versions, script and backend Python SHA256 hashes, complete geometry, timesteps, finite/no-gradient checks and measured errors. This repository copy ran in **13.34 seconds**, using Python 3.10.22, Torch 1.13.1+cpu and DeepWave 0.0.9 with two CPU threads. No GPU or full inversion was run.

The probe uses a homogeneous 2100 m/s model, 256 samples at 0.003 s, a 7 Hz Ricker source peaking at 1/7 s, an interior source at `[z=300,x=900]` m and three same-depth receivers at offsets 300, 450 and 600 m. Coarse `[40,70]` at 30 m and fine `[391,691]` at 3 m share physical sample extents 1170×2070 m. PML widths are physically equal at 300 m (10 coarse cells, 100 fine cells). Fine propagation internally uses dt0.0006 s (ratio5); coarse uses dt0.003 s (ratio1).

| All-sample metric | Actual value |
| --- | ---: |
| Coarse/fine raw trace norm ratio, equal coefficients | 100.3091118 |
| Relative L2 error, equal coefficients | 99.3094532 |
| Relative L2 error, theoretical 0.01 trace gain | 0.02622535 |
| Diagnostic least-squares coarse-to-fine gain | 0.00996581388 |
| Relative L2 error, fitted gain (diagnostic only) | 0.02600018 |
| Cosine similarity | 0.99966194 |

The deterministic area ratio agrees with the diagnostic fitted gain within 0.35%. Direct-arrival ±0.1 s windows give relative L2 0.02614636 with the exact 0.01 correction. The production source uses geometry, not the diagnostic best fit. The real public-adapter regression simulates with the actual prepared known coefficient and requires relative L2 below 0.05, while the unscaled coefficient must expose error above 90. Focused tests also prove unscaled fine-source generation, returned calibrated known source, unchanged 10 Hz initializer and unchanged raw observation values for both methods.

## Scope of the evidence

This is an interior homogeneous **off-source** check isolating the dominant source cell-area issue, not complete Marmousi raw equivalence or convergence evidence. Residual differences include spatial/temporal discretization and PML sampling. Product Marmousi preparation uses a shared fixed PML cell count, which means different physical widths on the two grids. Its source/receiver depths are zero and some positions coincide; near-source sampled Green's functions and boundary behavior can retain additional grid dependence. The area factor does not remove those effects. No raw trace normalization, backend physics, objective, optimizer or imported-data policy changes follow from this correction.
