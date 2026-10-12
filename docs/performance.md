# Performance evidence and measurement

The product benchmark compares actual `FWIGANStrategy` with the frozen original notebook training cell in `tests/reference/notebook_cells.json`. This intentional repository reference is recorded in benchmark provenance; the executable example does not substitute another optimized notebook for the package.

Run from the repository root after setup, with other CPU jobs idle:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python examples/benchmark.py --device cpu --suite tiny --output runs/benchmark-cpu.json
```

The tiny suite covers clean and noisy six-cycle FWIGAN work. Fresh identical seeds establish common state, then warmup requires stable scales and real physical/critic optimizer steps. Both branches start timed windows from the same physical tensors, critic, named Adam groups/moments, schedulers, scalers and RNG. Checkpoint I/O, bootstrap, warmup and separate DeepWave instrumentation are excluded from timing. Successful-step deltas reject incomparable AMP windows.

The JSON records repeated raw timing samples and medians, synchronized end-to-end wall time, solver forward/adjoint counts, update counts, physical/critic/Adam/loss errors and software/device provenance. Timing is an artifact, with no unstable CI speed threshold. The reference tests continue to establish numerical parity after notebook thinning.

## CUDA measurement

CUDA is optional and requires a compatible Torch 1.13.1/DeepWave 0.0.9 build. No GPU is automatically allocated. On a prepared GPU:

```bash
python -m pytest -q tests/test_cuda.py
python examples/benchmark.py --device cuda --suite all --output runs/benchmark-cuda.json
```

After the all-suite benchmark successfully writes real CUDA results, render its measured timings, adjoints and absolute/incremental allocated/reserved peaks:

```bash
python examples/plot_gpu_benchmark.py --input runs/benchmark-cuda.json --output-dir docs/assets
```

This renderer requires real GPU identity, all three workloads, repeated timing windows and non-null memory measurements. It rejects CPU, tiny-only and incomplete JSON; no GPU figure has been generated here.

The all suite adds a representative **synthetic constant-model** workload on a 100×310 inversion grid, nt2000, five shots, 60 receivers and six critic cycles. It is not a Marmousi convergence run. This workload is GPU-only; it is intentionally excluded from routine CPU training.

For a separately prepared CUDA environment, install the CUDA Torch build before compiling the legacy extension. A CUDA 11.7 toolchain and Torch `1.13.1+cu117` match the legacy stack. On Ada hardware, the legacy build can target Ampere PTX with `TORCH_CUDA_ARCH_LIST="8.6+PTX"`; run the CUDA integration suite afterward to establish package compatibility on the actual device.

```bash
python -m pip install setuptools==69.5.1 wheel==0.45.1
python -m pip install torch==1.13.1+cu117 --index-url https://download.pytorch.org/whl/cu117
python -m pip install numpy==1.23.5 scipy==1.10.1
TORCH_CUDA_ARCH_LIST="8.6+PTX" MAX_JOBS=2 python -m pip install --no-build-isolation deepwave==0.0.9
python -m pip install -e '.[test,plot,notebook]'
```

Each timing window uses CUDA synchronization at its boundaries, so includes host and solver transfers. Absolute and incremental peak allocated/reserved memory are recorded with each branch's live initial allocation; the other branch's model and optimizers are released and common snapshots remain on CPU. Capacity claims require real device peaks, not estimates.

The retained float32 wavefield estimate is `2000 × 3 × 5 × 124 × 334 × 4 = 4,969,920,000 bytes` (4.63 GiB). Both branches retain this store because they reuse the same gradient-enabled solve. Detachment removes unnecessary adjoints, not the dominant wavefield. Critic activations, cuDNN workspaces, optimizer memory and allocator reservation add overhead.

## Work reduction and remaining backend costs

At six critic cycles the original loop makes one forward and seven adjoints per group; the optimized strategy makes one forward and one adjoint. Critic forward and gradient-penalty work remain. Reducing seven adjoints to one does not imply a sevenfold total speedup. Tiny CPU timing cannot establish full-scale CUDA speedup.

DeepWave 0.0.9 still performs dynamic velocity checks, geometry reductions, CPU NumPy/SciPy resampling of source amplitudes and receiver gradients, and transfers back to CUDA. AMP does not halve the float32 solver wavefields. Update counters track actual optimizer calls without replacing skipped AMP attempts with successful work.

Useful knobs are bounded epoch/group budgets, FWI batch size, the existing equal FWIGAN shot groups, reporting/evaluation cadence, CPU thread limits and explicit CUDA AMP. Changing shot grouping changes critic channels and the preserved objective. New solvers, normalization, critics, mixed-precision physics, PML or compiler transformations need independent scientific validation.

## Reproducible figures

`python examples/illustrate.py` repeats the real 24 epoch bounded CPU FWI case and regenerates `docs/assets/tiny-*.png/.svg`, numerical arrays and metadata with config, fingerprint, units, library versions and solver counts. `--reuse-cache` redraws existing arrays explicitly. The shared 2000 m/s lower bound equals the initializer and is a material prior. Truth and inversion use the same grid; these plots establish a small executable path and data-fit improvement.

CPU benchmark JSON and plots are added only after real successful measurement. CUDA results remain unmeasured until the GPU suite actually executes; skipped CPU CUDA tests are not GPU evidence.

## Recorded tiny CPU result

The committed [JSON samples](assets/tiny-cpu-benchmark.json) were collected serially with two CPU threads, Python 3.10, Torch 1.13.1+cpu and DeepWave 0.0.9. Each window contains two warmed steps, repeated three times with alternating branch order, nt128, two shots per group, 60 receivers and six critic cycles. No other owned CPU test job ran during these samples; this shared host can still have unrelated contention.

| Variant | Original median seconds/group | Package median seconds/group |
| --- | ---: | ---: |
| Clean | 0.3793 | 0.1715 |
| Noisy | 0.3573 | 0.1735 |

![Actual CPU timing samples and solver adjoint counts](assets/tiny-cpu-benchmark.png)

Physical values/changes, critic parameters, both Adam states, physical gradients, generator losses, scalers and schedulers agree exactly on CPU; all RNG streams match. Only critic reporting averages allow tiny rounding differences (Python scalar mean versus float32 mean). Both branches execute one physical and six critic updates per group and retain the same 18,481,152-byte wavefield. The original makes seven adjoints; the package makes one. These tiny measurements establish neither GPU speedup nor Marmousi performance.

Recreate the plot from the emitted artifact with `python examples/plot_benchmark.py --input docs/assets/tiny-cpu-benchmark.json`.
