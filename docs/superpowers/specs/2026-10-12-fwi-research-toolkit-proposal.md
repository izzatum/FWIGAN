# FWI research toolkit: architecture proposal and refactor roadmap

Status: approved architecture. Implementation is underway; the current iteration covers phases 1–4 and the gradient-isolation optimization from phase 6. Method revisions and backend modernization remain separately validated milestones.

## Goal and scope

Build a reproducible research toolkit supporting conventional full-waveform inversion (FWI) and the existing adversarial FWIGAN experiments. The user selected this research-toolkit scope. Success means that both methods can use the same documented acquisition, acoustic forward model, observations, preprocessing, and evaluation, while experiments can be configured, run, resumed, and compared without depending on notebook execution order.

The initial physical scope is 2D, constant-density acoustic propagation with DeepWave and PyTorch automatic differentiation. Start with the bundled Marmousi tensors and small synthetic CPU examples. Generic tensor-based observations must be usable with an initial model and acquisition metadata, without a true velocity model. Field-data processing, SEG-Y ingestion, elastic/anisotropic modeling, distributed execution, and additional propagation engines are future projects.

CPU validation establishes executable behavior and selected numerical contracts. It does not establish full Marmousi convergence, GPU performance, or the scientific superiority of FWIGAN.

## Findings from the current repository

The repository has two Colab notebooks, bundled velocity tensors, and illustrations, with no Python package, dependency manifest, CLI, or test suite. Notebook source differs in just three cells: clean/noisy configuration and checkpoint names (cell 14), the sigma optimizer group (cell 16), and the observations passed to training (cell 22). These should become experiment presets sharing one implementation.

Confirmed issues and important scientific choices:

- In cell 10, velocity, source, and sigma are ordinary tensors rather than registered parameters or buffers. A direct probe found an empty Generator state_dict and parameter list. Cell 22 saves that state_dict, so its generator checkpoint omits the inverted quantities. The separate save_updates function is not called by the training loop.
- Cell 8 loads checkpoints onto CUDA and overwrites every optimizer-group learning rate with one supplied value. Generator reload passes 5, including to groups initially configured with source LR 1e-3 and noisy sigma LR 1e-2. Epoch, scheduler, scaler, observations, and RNG state are not restored.
- Cell 12 reads notebook globals for batching, critic cycles, device, and epoch. The caller cannot fully control its behavior through its arguments.
- The critic treats shots as channels and produces one score for an entire shot group. It is not currently a per-shot discriminator. Five pooling stages and a hard-coded dense dimension restrict geometry: a probe at 64 time samples passed with 60 receivers but failed with 31 and 64 receivers.
- Normalization is abs(max(trace)), not max(abs(trace)). A probe with values [-2,-1,0.5] produced normalized maximum magnitude 4. Correcting this changes the numerical method and needs a separately recorded comparison.
- Independent normalization of each predicted trace removes positive source-amplitude scale in a linear acoustic model. Source amplitude is therefore not identifiable from this objective alone. The unconstrained source also allows long-duration energy that can trade off against velocity changes.
- Gaussian sigma is unconstrained, and positive/negative sigma induce the same Gaussian distribution. Noise is added after normalization, so sigma describes normalized-data units rather than physical recording amplitude. The Laplace branch is broken; a direct probe raises an undefined-sigma NameError. Gaussian noise is the initial supported model.
- Truth and inversion use 3 m and 30 m grids, respectively. This reduces one source of inverse crime but does not establish that all inverse crimes are eliminated. Acquisition, interpolation, boundaries, and resampling still need explicit checks.
- Physical grid endpoints differ: the last truth grid point is (2997, 9297) m and the last inverse grid point is (2970, 9270) m at origin zero. Model comparison and resampling must use physical coordinates. DeepWave 0.0.9 defaults to ten absorbing-boundary cells, corresponding to 30 m versus 300 m on these grids; boundary sensitivity and grid dispersion should be measured.
- DT=0.003 is used as seconds, although its comment says milliseconds. Paths and units are currently implicit, and notebook installation requests unpinned DeepWave despite using its removed Propagator API.

The onboarding environment has already executed both original notebook variants on small synthetic CPU inputs, including forward propagation, gradients, critic and generator updates, plotting, and saving/loading update tensors. It has not validated the original checkpoint-resume path or a full inversion.

## Architecture choice

Three practical options:

1. Extract reusable functions and keep notebook-controlled experiments. This is the smallest change but leaves configuration, resumability, and reproducible comparison fragmented.
2. Create a modular Python research package with CLI and notebook clients. Recommended: a shared physics/data core and small explicit inversion strategies support both methods, with one experiment lifecycle and artifact contract.
3. Build a plugin platform around arbitrary engines and methods. This adds abstractions and maintenance before the repository has validated a second engine or a broader set of workflows.

Use option 2. Prefer pure functions for numerical transformations, dataclasses for validated configuration/data contracts, PyTorch modules for trainable state, and a few Protocol interfaces at actual variation points. One process and one composition root are sufficient.

## Component boundaries and patterns

| Component | Responsibility | Pattern and rationale |
| --- | --- | --- |
| GridSpec, Acquisition, Observations | Carry physical geometry, trace metadata, masks, and optional reference truth | Explicit data contracts prevent hidden unit/layout assumptions |
| InversionParameters | Own velocity/source/noise quantities and enabled trainable groups | PyTorch parameter/buffer ownership enables device movement and serialization |
| ForwardModel | Predict raw receiver traces with gradients through velocity and source | Adapter isolates DeepWave API/layout/coordinate conventions |
| Preprocessing | Filter, window, weight, and optionally normalize traces | Composition of small functions; explicit shared settings for observed/predicted data |
| FWI method and FWIGAN method | Define objective and their respective optimization/update schedules | Strategy; alternation belongs to FWIGAN rather than to a generic scalar-loss interface |
| ExperimentRunner | Prepare data, run continuation stages, evaluate, checkpoint, resume, and export | Dependency injection and a common lifecycle, with composition rather than a deep trainer hierarchy |
| CheckpointStore and metric/plot functions | Persist run state and report results | Small persistence and reporting components; named runner hooks where needed |
| CLI and notebooks | Construct experiments and inspect their results | One public API used by both entry points |

The runner receives explicit dependencies. A simple factory at the CLI/API composition root constructs the configured backend and method; there is no need for a plugin registry, service locator, event bus, or generic repository framework.

### Numerical contracts

- Velocity shape: [nz, nx]. Lengths in metres, velocity in m/s, time in seconds, frequency in Hz. Include model origin and spacing explicitly.
- Acquisition coordinates: [shots, points, 2], ordered [z, x] in physical metres. Validate in-domain points and convert to backend grid coordinates inside the adapter. Reject unsupported sub-grid coordinates rather than silently rounding them.
- Boundary configuration: explicitly declare absorbing-layer widths, top-boundary treatment, and numerical accuracy. Stability and waveform accuracy are separate checks; assess grid dispersion over the wavelet's bandwidth and test surface/collocated source-receiver injection.
- Source amplitudes: [shots, sources, time], with explicit shared-wavelet broadcasting.
- Canonical receiver traces: [shots, receivers, time]. Convert the legacy backend's time-major outputs at the adapter boundary.
- Propagation returns unnormalized deterministic waveforms. Observation noise and adversarial stochastic simulation are separate components.
- A forward contract can be `predict(velocity, source, acquisition, grid, sampling) -> receiver_traces`; it must preserve gradients for enabled parameters.
- An inversion strategy owns a `step(batch, state, context) -> metrics` operation. The FWI strategy updates physical parameters against a data loss; the FWIGAN strategy owns critic cycles, gradient penalty, stochastic noise when enabled, and generator updates.
- Both methods use the same acquisition and recorded observations for a comparison. Preprocessing must be explicit. New conventional runs use amplitude-preserving preprocessing by default; the existing per-trace normalization is retained only in a named legacy preset until a controlled experiment changes it.
- Preserve the legacy critic's shot-stack semantics through an adapter during extraction. A later per-shot critic consumes [shots, 1, time, receivers] and supports validated shapes. This changes the modeled distribution and must be an opt-in method revision.

### Proposed package layout

```text
pyproject.toml
src/fwigan/
    config.py                 # validated configuration and presets
    domain.py                 # grid, sampling, acquisition, observations
    parameters.py             # registered trainable state and constraints
    physics/
        deepwave.py           # backend adapter and coordinate/layout mapping
    data/
        io.py                 # observations/model tensors with metadata
        synthetic.py          # truth simulation and seeded Gaussian noise
        preprocessing.py      # filters, windows, weights, normalization
    inversion/
        fwi.py                # conventional FWI optimization strategy
        adversarial.py        # FWIGAN alternating updates and losses
        critic.py             # legacy critic and validated later revision
        regularization.py     # physical/source priors
    experiment.py             # lifecycle, continuation stages, batching
    checkpoints.py            # snapshots, resume, warm starts
    evaluation.py             # residual/data/model metrics
    plotting.py               # figures from results
    cli.py                    # composition root
configs/                      # legacy clean/noisy and conventional presets
examples/                     # bounded executable examples
notebooks/                    # package-based demonstrations
tests/                        # contracts, gradients, resume, smoke inversions
```

These are responsibility boundaries, not a requirement to create every file before it has working behavior. Extract the smallest runnable vertical slice first.

## Intended workflow

Validate config and physical geometry → load observations or generate a reproducible synthetic dataset → initialize enabled parameters → run configured frequency stages and shot batches → evaluate independently of the optimization loss → save a portable run snapshot → export physical results and figures.

Proposed CLI examples:

```bash
fwigan simulate --config configs/marmousi-data.yaml
fwigan invert --config configs/marmousi-fwi.yaml
fwigan invert --config configs/marmousi-fwigan.yaml
fwigan resume runs/<run-id>/checkpoint.pt
fwigan evaluate runs/<run-id>
```

Each run owns a unique output directory with the resolved configuration, dataset fingerprint/reference, version/device metadata, metrics, snapshots, and final velocity/source/noise exports. Simulation and inversion are separate commands: inversion must also work on recorded observations with no truth model.

Resume restores the same experiment. Warm start deliberately starts a new experiment from selected physical quantities. A complete snapshot includes registered inversion state, optional critic, optimizers, schedulers, scalers, completed stage/epoch/batch position, shot order, RNG states, metrics, schema version, configuration, and data identity. Store the observed noise realization or reference an immutable verified artifact. Write snapshots atomically and load onto the selected device. Parameter exports include actual physical velocity, not just latent bounded coordinates.

## Refactor roadmap and acceptance criteria

### 1. Characterize and lock the existing experiments

Document tensor layouts, acquisition units, preprocessing, optimizer groups, stochastic draw order, and update order. Capture tiny deterministic reference outputs with the existing Python 3.10/PyTorch 1.13.1/DeepWave 0.0.9 environment. Package the reference fixtures and tests inside the repository; onboarding-only helpers are not the long-term test interface.

Acceptance: fixed-input traces, normalization, losses, and velocity/source gradients have recorded comparison tolerances; legacy critic geometry limits are explicit; clean/noisy cases are covered. Numerical equivalence means agreement within justified tolerances, not universal bitwise equality.

### 2. Extract a runnable package and experiment entry point

Move duplicate logic into shared modules, introduce canonical contracts and the DeepWave adapter, add packaging/dependency metadata, and represent the two notebooks as clean/noisy legacy presets. Remove global-variable and absolute-path dependencies. Make notebooks call the same API as the CLI.

Acceptance: both legacy presets run on a bounded CPU example through the public API/CLI. After layout conversion, traces, losses, and gradients agree with phase 1. A loaded-observations run does not require reference truth. No algorithm or dependency upgrade is bundled into extraction.

### 3. Repair parameter ownership and experiment state

Register trainable tensors and fixed buffers, define separate optimizer groups, add portable atomic snapshots, and implement explicit resume and warm-start behavior. Preserve existing numerical parameterization for the legacy preset while designing validated constrained parameterizations for new runs.

Acceptance: velocity, source, sigma, and critic state round-trip; learning rates, schedulers, RNG, observation realization, and run position restore. An interrupted clean and noisy CPU run agrees with its uninterrupted counterpart within justified tolerances. CUDA checks run only when supported. Incompatible config/data snapshots fail clearly.

### 4. Establish a conventional FWI baseline

Implement a velocity-only inversion against raw-amplitude receiver data with a known fixed source, weighted L2/MSE data objective, explicit loss reduction, and configurable velocity constraints/regularization. Use Adam initially. Validate source/receiver geometry, finite inputs, time sampling, stability settings, and parameter bounds. Truth-derived model metrics are optional; data metrics are always available.

Acceptance: a small controlled model improves held-out or full-shot data fit and, when truth is available, model error. Directional finite-difference checks establish velocity/source gradient behavior on a small suitable problem. Batched residual gradients agree with full-shot gradients under the chosen reduction. Initial inversion cannot accidentally use ground-truth velocity.

### 5. Add frequency continuation and controlled joint inversion

Introduce explicit low-to-high frequency stages with the same stage filter applied to predicted and observed data, preserving differentiability of the prediction path. Define optimizer carry/reset policies in config. Enable source and positive Gaussian-noise estimation as separate ablations, with documented source support/bandwidth and amplitude constraints or priors. Conventional noise inference uses a Gaussian likelihood with both variance-weighted residuals and the log-variance normalization term; scalar L2 alone cannot estimate sigma. Distinguish statistical noise from forward-model discrepancy. Validate rather than silently enabling every inversion variable.

Acceptance: filters honor sampling/Nyquist constraints and operate consistently; a tiny stage transition and resumed transition agree. All shots are covered, including incomplete final batches. Tests validate enabled/frozen parameter groups and positive velocity/noise constraints. Compare velocity-only, velocity+source, and velocity+source+noise under the same observations; document identifiability limits.

### 6. Improve FWIGAN as an explicit method revision

Use the shared physical core and observations while preserving the legacy method as a reference. Detach simulated data during critic updates so the wave solver is not differentiated unnecessarily. Freeze critic parameters during physical-parameter updates while preserving the input gradient. Give the revised critic a documented per-shot input contract and validated shape handling, and support an optional weighted FWI/adversarial hybrid objective.

Acceptance: critic steps change only critic parameters and physical steps change only enabled physical parameters. Gradient-penalty computation remains valid; incomplete batches and selected acquisition sizes work or fail validation clearly. Benchmark legacy and revised methods on the same datasets and independent metrics, including multiple random seeds and forward/adjoint counts, runtime, and peak memory. Compare model errors on a common physical grid. Critic scores are not treated as velocity accuracy or evidence of superiority.

### 7. Modernize the backend and finish reproducibility

Validate a current supported DeepWave version behind the adapter in a separate pinned environment. Check location conversions, time/source conventions, boundary conditions, numerical defaults, traces, and gradients; differences can be expected and require explanations and calibrated tolerances. Switch the default only after this benchmark passes. Retain a reproducible legacy reference environment for comparisons.

Add fast CPU CI, documented optional GPU integration checks, a bounded synthetic inversion example, clean/noisy conventional and adversarial example configs, result comparison commands, and user documentation. Refresh cloud installation/startup configuration for the resulting package once its commands are tested.

Acceptance: a clean install runs the documented CPU example and meaningful tests; modern-backend tests meet explicit numerical criteria; GPU behavior is tested where hardware is available. Full Marmousi experiments are a separately resourced benchmark, with recorded config, data identity, convergence/fit metrics, memory, and runtime. No CPU-only readiness claim is presented as full scientific reproduction.

## Implementation sequencing

Phases 1–3 establish a trustworthy reusable version of the existing method and are the first deliverable. Phase 4 adds the scientifically interpretable baseline. Phases 5–6 add controlled method improvements. Phase 7 separates dependency modernization from algorithm and architecture changes.

The detailed task-by-task implementation plan should be derived from the reviewed architecture and contracts. Tests should target physical gradients, experiment continuity, update isolation, batching/geometry, and inversion behavior rather than merely mirroring helper implementations. Each migration phase should leave a runnable, reviewable repository.
