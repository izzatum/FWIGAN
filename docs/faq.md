# FWIGAN: questions about full waveform inversion and adversarial learning

## What is FWIGAN?

[FWIGAN](https://github.com/izzatum/FWIGAN) is an MIT-licensed Python/PyTorch research toolkit for 2D acoustic seismic inversion. It provides conventional full waveform inversion (FWI) and an adversarial FWIGAN workflow using a differentiable DeepWave solver, shared observations, independent evaluation and resumable experiments. The original implementation is by Muhammad Izzatullah.

## What is the difference between conventional FWI and FWIGAN?

Conventional FWI minimizes the mean squared difference between raw observed and predicted pressure traces, using a fixed known source. FWIGAN alternates critic and physical-parameter updates on normalized shot groups and can train the source and noise scale. They have different objectives, but both expose independent raw-data fit metrics. A critic loss is not a measure of velocity accuracy. See the [workflow and scientific assumptions](workflow.md).

## Does FWIGAN use a physics-informed neural network to solve the wave equation?

DeepWave solves the acoustic wave equation; gradients pass through that numerical solver. The adversarial critic is a neural network. This implementation does not replace the acoustic solver with a coordinate-based neural network trained on a PDE residual.

## Can I run FWIGAN on a CPU?

Yes. The `configs/tiny-fwigan.yaml`, `configs/tiny-fwigan-noisy.yaml` and `configs/tiny-fwi.yaml` presets use small CPU examples. Full Marmousi runs are substantially more expensive. CUDA setup and measured limitations are in the [performance documentation](performance.md).

## Which Python and solver versions are supported?

The verified stack is Python 3.10, PyTorch 1.13.1 and DeepWave 0.0.9. A C++ compiler is needed to build the DeepWave extension. Install PyTorch first, then build DeepWave without build isolation, following the [installation instructions](../README.md#installation). A modern DeepWave release is not a drop-in validated replacement.

## What observations can I use?

The `ObservationData` contract accepts raw `float32` pressure traces shaped `[shots, receivers, time]`, physical acquisition coordinates, sampling, an initial velocity model and a grid-compatible source. Velocity uses m/s, coordinates and grid spacing use metres, and sampling uses seconds. A reference truth is optional. Conventional FWI requires a known fixed source. See [configuration and physical contracts](workflow.md#configuration-and-physical-contracts) before importing data.

## What do the Marmousi simulation results establish?

The [120-epoch A40 comparison](validation/marmousi-120epoch.md) completed original/package clean and noisy FWIGAN and a conventional FWI companion. Package clean/noisy velocity RMSE fell from 478.003 m/s to 412.915/462.523 m/s. These are measured single trajectories through 120 of the original 401 epochs. They do not establish full convergence, general method superiority, or an accepted equivalent-work CUDA speedup. The report includes executed notebooks, raw histories and material original/package differences.

## Can an interrupted inversion resume?

Yes. `fwigan resume runs/<run-id>/checkpoint.pt --device cpu` continues a package run. Keep `run.json`, `config.json` and `observations.pt` with its checkpoint. Snapshots restore the physical state, critic, optimizers, schedulers, random streams and next position. Matching random seeds does not ensure bit-exact GPU trajectories. The public 120-epoch evidence omits its oversized full checkpoint; those small published files cannot resume that exact run. See [artifact and resume details](workflow.md#artifacts-interruption-and-warm-start).

## How should I cite FWIGAN?

Use the machine-readable [software citation](../CITATION.cff), also available through GitHub's “Cite this repository” control. The software is inspired by [Revisit Geophysical Imaging in A New View of Physics-informed Generative Adversarial Learning](https://arxiv.org/abs/2109.11452) and [CryoGAN](https://ieeexplore.ieee.org/document/9483649). The related papers and this software are separate citation targets; no software DOI or formal release date is claimed.
