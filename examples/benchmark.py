#!/usr/bin/env python3
"""Paired original-notebook/FWIGANStrategy benchmark; no timing assertions.

Requires an intentional source checkout containing tests/reference. After copying
to examples, the checkout is found automatically; otherwise pass --repo.
--suite all adds a synthetic representative clean workload and requires CUDA.
Each sample reloads one warmed CPU snapshot; branches never coexist on CUDA.
"""
import argparse
import contextlib
import copy
import gc
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time
import types
import warnings

os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import torch
import deepwave


def portable(value):
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: portable(v) for k, v in value.items()}
    if isinstance(value, list): return [portable(v) for v in value]
    if isinstance(value, tuple): return tuple(portable(v) for v in value)
    return copy.deepcopy(value)


def rng_state(device):
    return portable(dict(python=random.getstate(), numpy=np.random.get_state(),
                         torch=torch.get_rng_state(),
                         cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else []))


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']: torch.cuda.set_rng_state_all(state['cuda'])


def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(value)


class CountedAdam(torch.optim.Adam):
    """Count actual successful calls without querying CUDA scalers per cycle."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs); self.completed_steps = 0

    def step(self, closure=None):
        result = super().step(closure); self.completed_steps += 1
        return result


def source_moments(state, native):
    """Transform only source-shaped Adam moments; scalar step is unchanged."""
    result = portable(state)
    for group in result['param_groups']:
        if group['name'] == 'source':
            for key in group['params']:
                for field in ('exp_avg', 'exp_avg_sq'):
                    if key in result['state']:
                        value = result['state'][key][field]
                        result['state'][key][field] = value.permute(2, 0, 1).contiguous() if native else value.permute(1, 2, 0).contiguous()
    return result


class Original:
    """Unmodified frozen cell12, with device/config/bootstrap supplied explicitly.

    reference.strategy.make_reference hardcodes CPU and uses config.batch_size
    for channels. This adapter shares its cell execution/QuietProgress contract,
    uses actual shots/config.batches, and enables the real CUDA scalers.
    """
    def __init__(self, data, config, observed, frozen, helper, device):
        self.device, self.config = device, config
        self.data = data.to(device); self.observed = observed.to(device)
        # Frozen autocast defaults to enabled=True. Supply an execution facade
        # so CPU AMP is explicitly disabled without editing the frozen source.
        amp = types.SimpleNamespace(autocast=lambda **kwargs: torch.cuda.amp.autocast(enabled=device.type == 'cuda', **kwargs))
        facade = types.SimpleNamespace(**{k: getattr(torch, k) for k in dir(torch)})
        facade.cuda = types.SimpleNamespace(amp=amp)
        ns = dict(torch=facade, nn=torch.nn, np=np, dw=deepwave,
                  tqdm=helper.QuietProgress, BATCHES=config.batches,
                  CRITIC_CYCLE=config.critic_cycles, DEVICE=device, epoch=0)
        path = 'notebooks/FWIGAN_Noisy.ipynb' if config.noise else 'notebooks/FWIGAN.ipynb'
        for index in ('8', '10', '12'):
            exec(compile(frozen['notebooks'][path][index]['source'], f'frozen:{path}:cell{index}', 'exec'), ns)
        source = data.source if data.initial_source is None else data.initial_source
        shots = data.traces.shape[0] // config.batches
        self.generator = ns['Generator'](data.initial_velocity.clone(), torch.tensor(config.sigma_init),
            source.permute(2, 0, 1).contiguous().clone(), shots, data.grid.spacing,
            data.sampling.dt, config.noise, device)
        self.critic = ns['Critic'](shots, data.sampling.nt, data.traces.shape[1]).to(device)
        groups = [dict(name='velocity', params=[self.generator.model], lr=config.velocity_lr),
                  dict(name='source', params=[self.generator.source], lr=config.source_lr)]
        if config.noise: groups.append(dict(name='sigma', params=[self.generator.sigma], lr=config.sigma_lr))
        self.physical_optimizer = CountedAdam(groups, betas=config.adam_betas)
        self.critic_optimizer = CountedAdam([dict(name='critic', params=list(self.critic.parameters()), lr=config.critic_lr)], betas=config.adam_betas)
        for name, period in (('physical', config.physical_scheduler_step_size), ('critic', config.critic_scheduler_step_size)):
            setattr(self, name+'_scheduler', torch.optim.lr_scheduler.StepLR(getattr(self, name+'_optimizer'), period, gamma=config.scheduler_gamma))
            setattr(self, name+'_scaler', torch.cuda.amp.GradScaler(enabled=device.type == 'cuda'))
        self.train_fn = ns['train_fn']
        self.counters = dict(physical_updates=0, critic_updates=0, physical_skipped=0, critic_skipped=0)

    def run(self):
        before = counters(self)
        gl, cl = [], []
        self.train_fn(0, self.observed.permute(0, 2, 1).unsqueeze(0), self.generator, self.critic,
            self.physical_optimizer, self.critic_optimizer, self.physical_scheduler, self.critic_scheduler,
            gl, cl, self.physical_scaler, self.critic_scaler, self.generator.source,
            self.data.acquisition.sources, self.data.acquisition.receivers)
        for index, name in enumerate(('physical', 'critic')):
            actual = counters(self)[index]-before[index]
            expected = self.config.batches*(1 if name == 'physical' else self.config.critic_cycles)
            self.counters[name+'_updates'] += actual
            self.counters[name+'_skipped'] += expected-actual
        return [dict(generator_loss=g, critic_loss=c) for g, c in zip(gl, cl)]

    def physical(self, gradients=False):
        result = {}
        for name, native in (('velocity', self.generator.model), ('source', self.generator.source), ('sigma', self.generator.sigma)):
            value = native.grad if gradients else native
            if value is not None: result[name] = portable(value.permute(1, 2, 0) if name == 'source' else value)
        return result

    def load(self, snapshot):
        self.counters = copy.deepcopy(snapshot['counters'])
        for name, tensor in snapshot['physical'].items():
            target = getattr(self.generator, 'model' if name == 'velocity' else name)
            value = tensor.permute(2, 0, 1).contiguous() if name == 'source' else tensor
            with torch.no_grad(): target.copy_(value.to(self.device))
        self.critic.load_state_dict(snapshot['critic'])
        for name in ('physical', 'critic'):
            state = snapshot[name+'_optimizer']
            if name == 'physical': state = source_moments(state, native=True)
            getattr(self, name+'_optimizer').load_state_dict(state)
            getattr(self, name+'_scheduler').load_state_dict(snapshot[name+'_scheduler'])
            getattr(self, name+'_scaler').load_state_dict(snapshot[name+'_scaler'])


class Optimized:
    def __init__(self, data, config, observed, device):
        from fwigan.parameters import InversionParameters
        from fwigan.physics import LegacyDeepwave
        from fwigan.inversion import FWIGANStrategy
        source = data.source if data.initial_source is None else data.initial_source
        params = InversionParameters(data.initial_velocity, source, torch.tensor(config.sigma_init), config.train_source, config.train_sigma).to(device)
        self.strategy = FWIGANStrategy(params, LegacyDeepwave(), data, config, observed=observed)

    def __getattr__(self, name): return getattr(self.strategy, name)

    def run(self):
        metrics = [self.strategy.step(slice(b, None, self.config.batches)) for b in range(self.config.batches)]
        self.strategy.finish_epoch()
        return [dict(generator_loss=m['generator_loss'], critic_loss=m['critic_loss']) for m in metrics]

    def physical(self, gradients=False):
        return {name: portable(value.grad if gradients else value)
                for name in ('velocity', 'source', 'sigma')
                for value in [getattr(self.parameters, name)] if not gradients or value.grad is not None}

    def load(self, snapshot):
        self.parameters.load_state_dict(snapshot['physical'])
        state = self.strategy.state_dict()
        for name in ('critic', 'observed', 'physical_optimizer', 'critic_optimizer',
                     'physical_scheduler', 'critic_scheduler', 'physical_scaler', 'critic_scaler'):
            state[name] = snapshot[name]
        state['counters'] = copy.deepcopy(snapshot['counters'])
        self.strategy.load_state_dict(state)


def snapshot(branch, device):
    result = dict(physical=branch.physical(), critic=portable(branch.critic.state_dict()),
                  observed=portable(branch.observed), rng=rng_state(device), gradients=branch.physical(True))
    # These totals keep scheduler guards valid after restoring the common state.
    result['counters'] = copy.deepcopy(branch.counters)
    for name in ('physical', 'critic'):
        result[name+'_optimizer'] = portable(getattr(branch, name+'_optimizer').state_dict())
        if name == 'physical' and isinstance(branch, Original):
            result[name+'_optimizer'] = source_moments(result[name+'_optimizer'], native=False)
        result[name+'_scheduler'] = portable(getattr(branch, name+'_scheduler').state_dict())
        result[name+'_scaler'] = portable(getattr(branch, name+'_scaler').state_dict())
    return result


def counters(branch):
    return [getattr(branch, name+'_optimizer').completed_steps for name in ('physical', 'critic')]


def assert_steps(before, after, epochs, config):
    actual = [a-b for a, b in zip(after, before)]
    expected = [epochs*config.batches, epochs*config.batches*config.critic_cycles]
    if actual != expected: raise RuntimeError(f'incomparable window: actual optimizer updates {actual}, expected {expected}; AMP skipped updates')
    return dict(physical=actual[0], critic=actual[1])


def errors(left, right):
    if torch.is_tensor(left):
        diff = (left-right).abs()
        return dict(max_absolute=float(diff.max()), max_relative=float((diff / right.abs().clamp_min(1e-12)).max()))
    if isinstance(left, dict):
        if left.keys() != right.keys(): raise ValueError('numerical state keys differ')
        return {str(k): errors(left[k], right[k]) for k in left}
    if isinstance(left, (list, tuple)):
        if len(left) != len(right): raise ValueError('numerical state lengths differ')
        return [errors(l, r) for l, r in zip(left, right)]
    return dict(max_absolute=abs(left-right)) if isinstance(left, (float, int)) else dict(equal=left == right)


def require_close(left, right, rtol, atol):
    if torch.is_tensor(left): torch.testing.assert_close(left, right, rtol=rtol, atol=atol)
    elif isinstance(left, dict):
        if left.keys() != right.keys(): raise AssertionError('comparison keys differ')
        for key in left: require_close(left[key], right[key], rtol, atol)
    elif isinstance(left, (list, tuple)):
        for l, r in zip(left, right): require_close(l, r, rtol, atol)
    elif isinstance(left, (float, int)):
        if not abs(left-right) <= atol+rtol*abs(right): raise AssertionError((left, right))
    elif left != right: raise AssertionError((left, right))


@contextlib.contextmanager
def instrument(branch):
    scalar = importlib.import_module('deepwave.scalar.scalar')
    counts = dict(solver_forward=0, solver_adjoint=0, critic_forward=0, saved_wavefield_bytes=[])
    f, b, alloc = scalar.PropagatorFunction.forward, scalar.PropagatorFunction.backward, scalar._allocate_wavefields
    def wrap(fn, key):
        def call(*args, **kwargs):
            counts[key] += 1; return fn(*args, **kwargs)
        return call
    def allocation(*args, **kwargs):
        result = alloc(*args, **kwargs)
        counts['saved_wavefield_bytes'].append(result[1].numel()*result[1].element_size())
        return result
    def critic(*_): counts['critic_forward'] += 1
    scalar.PropagatorFunction.forward = staticmethod(wrap(f, 'solver_forward'))
    scalar.PropagatorFunction.backward = staticmethod(wrap(b, 'solver_adjoint'))
    scalar._allocate_wavefields = allocation
    hook = branch.critic.register_forward_hook(critic)
    try: yield counts
    finally:
        scalar.PropagatorFunction.forward = staticmethod(f)
        scalar.PropagatorFunction.backward = staticmethod(b)
        scalar._allocate_wavefields = alloc; hook.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--suite', choices=('tiny', 'all'), default='tiny')
    parser.add_argument('--variant', choices=('both', 'clean', 'noisy'), default='both')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=2, help='epochs per tiny timed window')
    parser.add_argument('--representative-steps', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--cycles', type=int, default=6, help='tiny cycles; representative always uses six')
    parser.add_argument('--batches', type=int, default=1, help='strided groups; tiny has two shots per group')
    parser.add_argument('--tiny-nt', type=int, default=128)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-warmup', type=int, default=24)
    parser.add_argument('--rtol', type=float, default=2e-4, help='CUDA numerical tolerance (recorded)')
    parser.add_argument('--atol', type=float, default=2e-5, help='CUDA numerical tolerance (recorded)')
    parser.add_argument('--exclusive-cpu', action='store_true', help='caller certifies no overlapping CPU jobs')
    args = parser.parse_args()
    for key in ('steps', 'representative_steps', 'repeats', 'cycles', 'batches', 'tiny_nt', 'threads', 'max_warmup'):
        if getattr(args, key) < 1: parser.error(f'--{key.replace("_", "-")} must be positive')
    if args.rtol < 0 or args.atol < 0 or not np.isfinite([args.rtol, args.atol]).all(): parser.error('tolerances must be finite and nonnegative')
    if args.suite == 'all' and args.device != 'cuda': parser.error('--suite all requires CUDA; representative CPU training is intentionally excluded')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available(): parser.error('CUDA is unavailable; no GPU provisioning is performed')
    repo = args.repo.resolve()
    fixture = repo/'tests/reference/notebook_cells.json'
    if not fixture.is_file(): parser.error('use --repo to select a checkout containing tests/reference/notebook_cells.json')
    sys.path.insert(0, str(repo/'src'))
    spec = importlib.util.spec_from_file_location('_fwigan_benchmark_reference', repo/'tests/reference/__init__.py', submodule_search_locations=[str(repo/'tests/reference')])
    reference = importlib.util.module_from_spec(spec); sys.modules[spec.name] = reference; spec.loader.exec_module(reference)
    helper = importlib.import_module('_fwigan_benchmark_reference.strategy')
    from fwigan.config import ExperimentConfig
    from fwigan.data import make_tiny_data, legacy_normalize, to_critic_layout
    from fwigan.domain import GridSpec, SamplingSpec, Acquisition, ObservationData
    from fwigan.physics import LegacyDeepwave
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(args.threads)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    def sync():
        if device.type == 'cuda': torch.cuda.synchronize(device)
    def release():
        gc.collect()
        if device.type == 'cuda': sync(); torch.cuda.empty_cache()
    files = [fixture, repo/'tests/reference/strategy.py', repo/'src/fwigan/inversion/adversarial.py', repo/'src/fwigan/physics/deepwave.py']
    report = dict(schema=1, device=str(device), suite=args.suite, seed=args.seed,
        threads=args.threads, torch=torch.__version__, deepwave=getattr(deepwave, '__version__', '0.0.9'),
        python=platform.python_version(), cuda_build=torch.version.cuda,
        timing_scope='synchronized wall time of complete train epochs; excludes construction, warmup, checkpoint transfer and instrumentation',
        cpu_contention='caller certified exclusive' if args.exclusive_cpu else 'uncertain/shared host; indicative CPU timing only',
        provenance={str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        frozen_revision=reference.FROZEN['revision'], numerical_tolerance=dict(rtol=args.rtol if args.device == 'cuda' else 0., atol=args.atol if args.device == 'cuda' else 0., cpu_critic_loss_atol=2e-6, cpu_critic_loss_rtol=2e-7), workloads={})
    report['benchmark_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['cuda_numerics'] = dict(cudnn_deterministic=torch.backends.cudnn.deterministic,
                                 cudnn_benchmark=torch.backends.cudnn.benchmark,
                                 cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                                 matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    if device.type == 'cuda':
        report['gpu'] = dict(name=torch.cuda.get_device_name(device), cudnn=torch.backends.cudnn.version(), free_total_bytes=list(torch.cuda.mem_get_info()))
    workloads = [('tiny-'+name, name == 'noisy', False) for name in ('clean', 'noisy') if args.variant in ('both', name)]
    if args.suite == 'all': workloads.append(('representative-clean', False, True))
    for label, noisy, representative in workloads:
        seed(args.seed + int(noisy))
        if representative:
            grid, sampling = GridSpec((100, 310), 30.), SamplingSpec(2000, .003)
            xs = torch.zeros(5, 1, 2); xs[..., 0] = 90.; xs[:, 0, 1] = torch.linspace(15, 285, 5).round()*30
            xr = torch.zeros(5, 60, 2); xr[..., 0] = 90.; xr[..., 1] = (torch.arange(60)+3)*30
            acquisition = Acquisition(xs, xr)
            source = deepwave.wavelets.ricker(7., 2000, .003, 1/7).reshape(1, 1, -1)
            initial_source = deepwave.wavelets.ricker(10., 2000, .003, 1/7).reshape(1, 1, -1)
            # GPU simulation, then release its device tensors before training.
            with torch.no_grad():
                traces = LegacyDeepwave().predict(torch.full(grid.shape, 2200., device=device), source.to(device), acquisition.to(device), grid, sampling).cpu()
            data = ObservationData(traces, acquisition, grid, sampling, source, torch.full(grid.shape, 2100.), initial_source=initial_source)
        else: data = make_tiny_data(seed=args.seed, nt=args.tiny_nt, shots=2*args.batches)
        config = ExperimentConfig(device=args.device, batches=1 if representative else args.batches,
            batch_size=data.traces.shape[0]//(1 if representative else args.batches), critic_cycles=6 if representative else args.cycles,
            amp=args.device == 'cuda', noise=noisy, sigma_init=.08)
        # Preserve native trace strides for legacy fake and observation noise RNG.
        observed = to_critic_layout(legacy_normalize(data.traces))
        if noisy: observed = observed + .05*torch.randn_like(observed)
        observed = observed.squeeze(0).permute(0, 2, 1)
        def make(kind, common=None):
            branch = Original(data, config, observed, reference.FROZEN, helper, device) if kind == 'original' else Optimized(data, config, observed, device)
            if common is not None: branch.load(common); restore_rng(common['rng'])
            return branch
        branch = make('original'); warmup = []; stable = 0
        for _ in range(args.max_warmup):
            before = counters(branch); scales = [getattr(branch, n+'_scaler').get_scale() for n in ('physical', 'critic')]
            branch.run(); sync()
            actual = [a-b for a, b in zip(counters(branch), before)]
            after_scales = [getattr(branch, n+'_scaler').get_scale() for n in ('physical', 'critic')]
            stable = stable+1 if actual == [config.batches, config.batches*config.critic_cycles] and scales == after_scales else 0
            warmup.append(dict(actual_updates=actual, scales=after_scales))
            if stable >= 2: break
        if stable < 2: raise RuntimeError(f'{label}: no two consecutive stable successful original warmup epochs')
        common = snapshot(branch, device); branch = None; release()
        # Warm optimized kernels separately from the same calibrated state.
        branch = make('optimized', common)
        optimized_warmup = []
        for _ in range(2):
            before = counters(branch); scales = [getattr(branch, n+'_scaler').get_scale() for n in ('physical', 'critic')]
            branch.run(); sync(); updates = assert_steps(before, counters(branch), 1, config)
            after_scales = [getattr(branch, n+'_scaler').get_scale() for n in ('physical', 'critic')]
            if scales != after_scales: raise RuntimeError(f'{label}: optimized calibration changes AMP scale')
            optimized_warmup.append(dict(actual_updates=updates, scales=after_scales))
        branch = None; release()
        windows = {k: [] for k in ('original', 'optimized')}; comparisons = []; count_reports = {}
        steps = args.representative_steps if representative else args.steps
        for repetition in range(args.repeats):
            finals = {}
            for kind in (('original', 'optimized') if repetition % 2 == 0 else ('optimized', 'original')):
                branch = make(kind, common); before = counters(branch); sync()
                memory = None
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(device)
                    memory = dict(baseline_allocated_bytes=torch.cuda.memory_allocated(device), baseline_reserved_bytes=torch.cuda.memory_reserved(device))
                start = time.perf_counter()
                losses = [branch.run() for _ in range(steps)]
                sync(); elapsed = time.perf_counter()-start
                if memory is not None:
                    memory.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device), peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
                    memory['peak_allocated_increment_bytes'] = memory['peak_allocated_bytes']-memory['baseline_allocated_bytes']
                    memory['peak_reserved_increment_bytes'] = memory['peak_reserved_bytes']-memory['baseline_reserved_bytes']
                updates = assert_steps(before, counters(branch), steps, config)
                final = snapshot(branch, device); final['losses'] = losses
                final['physical_changes'] = {name: value-common['physical'][name] for name, value in final['physical'].items()}
                final['physical_change_l2'] = {name: float(value.norm()) for name, value in final['physical_changes'].items()}
                for name in ('velocity', 'source'):
                    if not torch.isfinite(final['physical_changes'][name]).all() or final['physical_change_l2'][name] == 0:
                        raise AssertionError(f'{label}/{kind}: physical {name} update must be finite and nonzero')
                finals[kind] = final
                windows[kind].append(dict(seconds=elapsed, seconds_per_batch=elapsed/(steps*config.batches), actual_updates=updates, memory=memory))
                branch = None; release()
            left, right = finals['original'], finals['optimized']
            fields = ('physical', 'physical_changes', 'critic', 'physical_optimizer', 'critic_optimizer', 'gradients', 'physical_scheduler', 'critic_scheduler', 'physical_scaler', 'critic_scaler')
            for field in fields: require_close(left[field], right[field], args.rtol if device.type == 'cuda' else 0., args.atol if device.type == 'cuda' else 0.)
            for l_epoch, r_epoch in zip(left['losses'], right['losses']):
                for l_metrics, r_metrics in zip(l_epoch, r_epoch):
                    # Scalar Python averaging in the original and float32
                    # on-device mean in production round the critic metric
                    # differently. The CPU generator and all updates stay exact.
                    require_close(l_metrics['generator_loss'], r_metrics['generator_loss'], args.rtol if device.type == 'cuda' else 0., args.atol if device.type == 'cuda' else 0.)
                    require_close(l_metrics['critic_loss'], r_metrics['critic_loss'], args.rtol if device.type == 'cuda' else 2e-7, args.atol if device.type == 'cuda' else 2e-6)
            cpu_rng_equal = torch.equal(left['rng']['torch'], right['rng']['torch'])
            cuda_rng_equal = all(torch.equal(l, r) for l, r in zip(left['rng']['cuda'], right['rng']['cuda']))
            python_rng_equal = left['rng']['python'] == right['rng']['python']
            numpy_rng_equal = all(np.array_equal(l, r) for l, r in zip(left['rng']['numpy'], right['rng']['numpy']))
            if not all((cpu_rng_equal, cuda_rng_equal, python_rng_equal, numpy_rng_equal)): raise AssertionError('branch RNG streams diverged')
            comparison = {field: errors(left[field], right[field]) for field in fields}
            comparison.update(losses=errors(left['losses'], right['losses']), torch_rng_equal=cpu_rng_equal, cuda_rng_equal=cuda_rng_equal,
                              python_rng_equal=python_rng_equal, numpy_rng_equal=numpy_rng_equal,
                              physical_change_l2={kind: finals[kind]['physical_change_l2'] for kind in finals})
            comparisons.append(comparison)
        # Real native forward/adjoint hooks are applied only in an untimed epoch.
        for kind in ('original', 'optimized'):
            branch = make(kind, common); before = counters(branch)
            with instrument(branch) as counts: branch.run(); sync()
            counts['actual_updates'] = assert_steps(before, counters(branch), 1, config)
            expected = dict(solver_forward=config.batches, solver_adjoint=config.batches*(config.critic_cycles+1 if kind == 'original' else 1), critic_forward=config.batches*(3*config.critic_cycles+1))
            if any(counts[k] != v for k, v in expected.items()): raise AssertionError((counts, expected))
            count_reports[kind] = counts; branch = None; release()
        medians = {kind: statistics.median(w['seconds_per_batch'] for w in samples) for kind, samples in windows.items()}
        report['workloads'][label] = dict(synthetic=True, geometry=dict(model=list(data.grid.shape), nt=data.sampling.nt, shots=data.traces.shape[0], receivers=data.traces.shape[1], spacing_m=data.grid.spacing, dt_s=data.sampling.dt),
            config=config.to_dict(), steps_per_window=steps, repetitions=args.repeats, warmup_original=warmup,
            warmup_optimized=optimized_warmup, windows=windows, median_seconds_per_batch=medians,
            original_over_optimized_time_ratio=medians['original']/medians['optimized'], numerical_comparisons=comparisons, untimed_counts=count_reports,
            limitation='synthetic training-loop evidence; not convergence or full Marmousi performance')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(workload=label, median_seconds_per_batch=medians, numerical_comparison='passed', counts=count_reports)), flush=True)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='User provided device_type.*CUDA is not available.*')
        main()
