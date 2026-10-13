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


class CalibrationError(RuntimeError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def calibrate_common(make, config, device, label, steps, max_warmup, sync, release):
    """Find a shared state whose next epochs succeed in both implementations."""
    expected = [config.batches, config.batches*config.critic_cycles]
    names = ('physical', 'critic')

    def scales(branch):
        return [getattr(branch, name+'_scaler').get_scale() for name in names]

    def epoch(branch):
        before, scales_before = counters(branch), scales(branch)
        branch.run(); sync()
        actual = [a-b for a, b in zip(counters(branch), before)]
        scales_after = scales(branch)
        return dict(actual_updates=actual, skipped_updates=[e-a for e, a in zip(expected, actual)],
                    scales_before=scales_before, scales_after=scales_after, scales=scales_after,
                    stable=actual == expected and scales_before == scales_after)

    needed = max(2, steps)
    branch = make('original'); warmup = []; stable = 0
    for _ in range(max_warmup):
        result = epoch(branch)
        warmup.append(result)
        stable = stable+1 if result['stable'] else 0
        if stable >= 2: break
    if stable < 2:
        raise CalibrationError(f'{label}: no two consecutive stable successful original warmup epochs '
                               f'within {max_warmup} epochs; last={warmup[-1]}',
                               dict(warmup_original=warmup, precheck_epochs=needed, precheck_attempts=[]))
    common = snapshot(branch, device); branch = None; release()

    attempts = []
    for attempt_number in range(1, max_warmup+1):
        candidate_scales = [common[name+'_scaler'].get('scale', 1.) for name in names]
        branches = []; lowered = {}; passed = True
        for kind in ('original', 'optimized'):
            branch = make(kind, common)
            epochs = []
            for _ in range(needed):
                result = epoch(branch); epochs.append(result)
                if not result['stable']: break
            final_scales = scales(branch)
            branch_result = dict(kind=kind, scales_before=epochs[0]['scales_before'],
                                 scales_after=final_scales,
                                 successful_updates=[sum(e['actual_updates'][i] for e in epochs) for i in range(2)],
                                 skipped_updates=[sum(e['skipped_updates'][i] for e in epochs) for i in range(2)],
                                 stable_epochs=sum(e['stable'] for e in epochs), epochs=epochs)
            branches.append(branch_result)
            passed &= len(epochs) == needed and all(e['stable'] for e in epochs)
            if any(after < before for after, before in zip(final_scales, candidate_scales)):
                observed = snapshot(branch, device)
                for index, name in enumerate(names):
                    if final_scales[index] < candidate_scales[index]:
                        previous = lowered.get(name)
                        if previous is None or final_scales[index] < previous['scale']:
                            lowered[name] = observed[name+'_scaler']
            branch = None; release()
        attempts.append(dict(number=attempt_number, common_scales=candidate_scales,
                             branches=branches, passed=passed))
        if passed:
            return common, dict(warmup_original=warmup, precheck_epochs=needed,
                                precheck_attempts=attempts)
        if not lowered:
            raise CalibrationError(f'{label}: joint calibration cannot advance after {attempt_number} attempts; '
                                   f'no naturally lowered AMP scaler; last={branches}',
                                   dict(warmup_original=warmup, precheck_epochs=needed,
                                        precheck_attempts=attempts))
        for name, state in lowered.items():
            common[name+'_scaler'] = state
    raise CalibrationError(f'{label}: joint calibration failed after {max_warmup} attempts; '
                           f'last={attempts[-1]}',
                           dict(warmup_original=warmup, precheck_epochs=needed,
                                precheck_attempts=attempts))


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


def json_safe(value):
    if isinstance(value, float): return value if np.isfinite(value) else str(value)
    if isinstance(value, dict): return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(item) for item in value]
    return value


def compare_value(path, left, right, rtol, atol, diagnostics, device_type, velocity_state=None):
    """Collect field-labelled comparisons; only CUDA velocity changes use a ULP floor."""
    if torch.is_tensor(left) or torch.is_tensor(right):
        result = dict(path=path, kind='tensor', shape=list(left.shape) if torch.is_tensor(left) else None,
                      other_shape=list(right.shape) if torch.is_tensor(right) else None,
                      dtype=str(left.dtype) if torch.is_tensor(left) else None,
                      other_dtype=str(right.dtype) if torch.is_tensor(right) else None,
                      rtol=rtol, atol=atol)
        diagnostics.append(result)
        if not torch.is_tensor(left) or not torch.is_tensor(right) or left.shape != right.shape or left.dtype != right.dtype:
            result.update(passed=False, reason='tensor type, shape or dtype mismatch')
            return
        special = path == 'physical_changes.velocity' and device_type == 'cuda'
        if special:
            if velocity_state is None or any(not torch.is_tensor(v) or v.dtype != torch.float32 or v.shape != left.shape
                                             for v in velocity_state) or left.dtype != torch.float32:
                result.update(passed=False, reason='velocity floor requires matching float32 common, finals and changes')
                return
            if not all(bool(torch.isfinite(v).all()) for v in (left, right, *velocity_state)):
                result.update(passed=False, reason='nonfinite velocity state')
                return
        if left.dtype == torch.bool:
            old_failed = left != right
            diff = old_failed.to(torch.int64)
            old_bound = None
            result.update(elements=left.numel(), old_failed=int(old_failed.sum()),
                          max_absolute=int(diff.max()) if diff.numel() else None, max_relative=None)
        else:
            diff = (left-right).abs()
            old_bound = atol + rtol*right.abs()
            old_failed = ~torch.isfinite(diff) | (diff > old_bound)
            finite_diff = diff[torch.isfinite(diff)]
            result.update(elements=left.numel(), old_failed=int(old_failed.sum()),
                          max_absolute=float(finite_diff.max()) if finite_diff.numel() else None,
                          max_relative=json_safe(float((diff/right.abs().clamp_min(1e-12)).max())) if diff.numel() else None)
        if special:
            common, final_original, final_optimized = velocity_state
            magnitude = torch.maximum(common.abs(), torch.maximum(final_original.abs(), final_optimized.abs()))
            next_value = torch.nextafter(magnitude, torch.full_like(magnitude, float('inf')))
            floor = next_value-magnitude
            if not bool(torch.isfinite(floor).all()) or not bool((floor > 0).all()):
                result.update(passed=False, reason='nonfinite or zero float32 velocity ULP')
                return
            rescued = old_failed & (diff <= torch.maximum(old_bound, floor))
            still_failed = old_failed & ~rescued
            result.update(rule='max(atol + rtol * abs(optimized_change), one local float32 model ULP)',
                          floor_min=float(floor.min()), floor_max=float(floor.max()),
                          rescued_by_ulp=int(rescued.sum()), still_failed=int(still_failed.sum()),
                          passed=not bool(still_failed.any()))
            samples = []
            for index in torch.nonzero(old_failed)[:8].tolist():
                key = tuple(index)
                samples.append(dict(index=index, common=float(common[key]),
                                    final_original=float(final_original[key]), final_optimized=float(final_optimized[key]),
                                    change_original=float(left[key]), change_optimized=float(right[key]),
                                    absolute_error=float(diff[key]), old_bound=float(old_bound[key]), floor=float(floor[key]),
                                    rescued=bool(rescued[key])))
            result['samples'] = samples
        else:
            try: torch.testing.assert_close(left, right, rtol=rtol, atol=atol)
            except AssertionError as error: result.update(passed=False, reason=str(error).splitlines()[0])
            else: result['passed'] = True
        return
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            diagnostics.append(dict(path=path, kind='mapping', passed=False,
                                    reason='keys differ', original_keys=[str(k) for k in left],
                                    optimized_keys=[str(k) for k in right]))
        for key in left:
            if key in right:
                compare_value(f'{path}.{key}', left[key], right[key], rtol, atol, diagnostics,
                              device_type, velocity_state)
        return
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            diagnostics.append(dict(path=path, kind='sequence', passed=False,
                                    reason=f'lengths differ: {len(left)} versus {len(right)}'))
        for index, (l_item, r_item) in enumerate(zip(left, right)):
            compare_value(f'{path}[{index}]', l_item, r_item, rtol, atol, diagnostics,
                          device_type, velocity_state)
        return
    try: require_close(left, right, rtol, atol)
    except (AssertionError, TypeError, ValueError): passed = False
    else: passed = True
    diagnostics.append(dict(path=path, kind='scalar', passed=passed,
                            original=json_safe(left), optimized=json_safe(right)))


def record_comparison(report, label, repetition, diagnostics, output):
    work = report['workloads'][label]
    work.setdefault('comparison_diagnostics', []).append(dict(repetition=repetition, fields=diagnostics))
    failed = [item['path'] for item in diagnostics if not item['passed']]
    if failed:
        work['status'] = 'numerical_failed'
        work['error'] = f'{label}: numerical comparison failed: {", ".join(failed)}'
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        raise AssertionError(work['error'])


class DiagnosticError(RuntimeError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


DIAGNOSTIC_FIELDS = ('physical', 'critic', 'observed', 'physical_optimizer',
                     'critic_optimizer', 'physical_scheduler', 'critic_scheduler',
                     'physical_scaler', 'critic_scaler', 'counters')


def state_fingerprint(value):
    digest = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(f'tensor:{tensor.dtype}:{tuple(tensor.shape)}:'.encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(f'array:{item.dtype}:{item.shape}:'.encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            digest.update(b'dict{')
            for key in sorted(item, key=str): visit(key); visit(item[key])
            digest.update(b'}')
        elif isinstance(item, (list, tuple)):
            digest.update(f'{type(item).__name__}:{len(item)}:['.encode())
            for child in item: visit(child)
            digest.update(b']')
        else:
            digest.update(f'{type(item).__name__}:{repr(item)};'.encode())
    visit(value)
    return digest.hexdigest()


def diagnostic_rng_equal(left, right):
    return dict(python=left['python'] == right['python'],
                numpy=all(np.array_equal(a, b) for a, b in zip(left['numpy'], right['numpy'])),
                torch=torch.equal(left['torch'], right['torch']),
                cuda=len(left['cuda']) == len(right['cuda']) and
                     all(torch.equal(a, b) for a, b in zip(left['cuda'], right['cuda'])))


def diagnostic_state(state):
    return {key: state[key] for key in DIAGNOSTIC_FIELDS} | {'rng': state['rng']}


def tensor_stats(values):
    result = {}
    for name, value in values.items():
        if value is None:
            result[name] = dict(present=False)
        else:
            finite = bool(torch.isfinite(value).all())
            result[name] = dict(present=True, shape=list(value.shape), finite=finite,
                                l2=json_safe(float(value.float().norm())),
                                max_absolute=json_safe(float(value.abs().max())) if value.numel() else 0.)
    return result


@contextlib.contextmanager
def capture_diagnostic_cycles(branch):
    """Observe losses and pre-step gradients without changing optimizer work."""
    events = {'physical': [], 'critic': []}
    originals = []
    for name in ('physical', 'critic'):
        scaler = getattr(branch, name+'_scaler')
        optimizer = getattr(branch, name+'_optimizer')
        scale = getattr(scaler, 'scale', None)
        step = getattr(optimizer, 'step', None)
        if callable(scale):
            def observed_scale(loss, *args, _name=name, _scale=scale, **kwargs):
                events[_name].append(dict(loss=json_safe(float(loss.detach())),
                                          step_called=False))
                return _scale(loss, *args, **kwargs)
            scaler.scale = observed_scale
            originals.append((scaler, 'scale', scale))
        if callable(step):
            def observed_step(*args, _name=name, _step=step, **kwargs):
                event = events[_name][-1] if events[_name] else None
                if event is not None:
                    if _name == 'physical':
                        event['gradients'] = branch.physical(True)
                    else:
                        event['gradients'] = {key: portable(parameter.grad) if parameter.grad is not None else None
                                              for key, parameter in branch.critic.named_parameters()}
                    event['gradient_stats'] = tensor_stats(event['gradients'])
                value = _step(*args, **kwargs)
                if event is not None:
                    event['step_called'] = True
                    event['post_weights'] = (branch.physical() if _name == 'physical'
                                             else portable(branch.critic.state_dict()))
                    event['weight_stats'] = tensor_stats(event['post_weights'])
                return value
            optimizer.step = observed_step
            originals.append((optimizer, 'step', step))
    try: yield events
    finally:
        for obj, attr, value in reversed(originals): setattr(obj, attr, value)


def diagnostic_cycle_summary(events):
    return {name: [dict(loss=event.get('loss'), step_called=event['step_called'],
                        gradient_stats=event.get('gradient_stats', {}) if index == 0 else None,
                        weight_stats=event.get('weight_stats', {}) if index == 0 else None)
                   for index, event in enumerate(cycles)]
            for name, cycles in events.items()}


def diagnostic_compare(left, right, common, device_type, rtol, atol, label, bit_exact=False):
    tolerance = (0., 0.) if bit_exact else (rtol, atol)
    diagnostics = []
    velocity_state = (common['physical']['velocity'], left['physical']['velocity'],
                      right['physical']['velocity'])
    for field in (*DIAGNOSTIC_FIELDS, 'gradients', 'physical_changes'):
        compare_value(field, left[field], right[field], *tolerance, diagnostics,
                      'cpu' if bit_exact else device_type, velocity_state)
    compare_value('metrics', left['metrics'], right['metrics'], *tolerance, diagnostics,
                  'cpu' if bit_exact else device_type)
    if not bit_exact and device_type == 'cpu':
        for entry in diagnostics:
            if entry['path'].endswith('.critic_loss') and entry['kind'] == 'scalar':
                entry['passed'] = abs(entry['original']-entry['optimized']) <= 2e-6+2e-7*abs(entry['optimized'])
    for stream, equal in diagnostic_rng_equal(left['rng'], right['rng']).items():
        diagnostics.append(dict(path=f'rng.{stream}', kind='rng', passed=equal))
    for name in ('critic', 'physical'):
        l_events, r_events = left['cycles'][name], right['cycles'][name]
        if len(l_events) != len(r_events):
            diagnostics.append(dict(path=f'{name}_cycles', kind='sequence', passed=False,
                                    reason='cycle counts differ'))
        for index, (l_event, r_event) in enumerate(zip(l_events, r_events)):
            compare_value(f'{name}_cycles[{index}].loss', l_event.get('loss'), r_event.get('loss'),
                          *tolerance, diagnostics, 'cpu' if bit_exact else device_type)
            if index == 0:
                for key in ('gradients', 'post_weights'):
                    compare_value(f'{name}_cycles[0].{key}', l_event.get(key), r_event.get(key),
                                  *tolerance, diagnostics, 'cpu' if bit_exact else device_type)
    return dict(label=label, bit_exact=bit_exact, failed_fields=[entry['path'] for entry in diagnostics if not entry['passed']],
                fields=diagnostics)


def diagnostic_replays(make, common, config, device, replays, sync, release, rtol=2e-4, atol=2e-5):
    """Untimed paired repeatability trace from one immutable calibrated state."""
    report = dict(status='diagnostic_only', mode='repeatability_diagnostic', epochs_per_replay=2,
                  replays=replays, common_fingerprint=state_fingerprint(diagnostic_state(common)),
                  runs=[], comparisons=[])
    pristine = portable(common)
    internal = {'original': [], 'optimized': []}
    for replay in range(replays):
        for kind in ('original', 'optimized'):
            branch = make(kind, common)
            start = snapshot(branch, device)
            start_fields = []
            for field in DIAGNOSTIC_FIELDS:
                compare_value(field, start[field], common[field], 0., 0., start_fields, 'cpu')
            rng_equal = diagnostic_rng_equal(start['rng'], common['rng'])
            valid = all(entry['passed'] for entry in start_fields) and all(rng_equal.values())
            run = dict(replay=replay, kind=kind, start_valid=valid,
                       start_fingerprint=state_fingerprint(diagnostic_state(start)),
                       start_failures=[entry['path'] for entry in start_fields if not entry['passed']],
                       start_rng_equal=rng_equal, epochs=[], common_unchanged=True)
            report['runs'].append(run)
            if not valid:
                report['status'] = 'diagnostic_invalid'
                run['start_diagnostics'] = [entry for entry in start_fields if not entry['passed']]
                raise DiagnosticError(f'{kind} loaded state differs from common snapshot', report)
            with capture_diagnostic_cycles(branch) as events:
                for epoch in range(2):
                    before = counters(branch)
                    scales_before = [getattr(branch, name+'_scaler').get_scale() for name in ('physical', 'critic')]
                    metric = branch.run(); sync()
                    after = counters(branch)
                    actual = [a-b for a, b in zip(after, before)]
                    expected = [config.batches, config.batches*config.critic_cycles]
                    state = snapshot(branch, device)
                    state['physical_changes'] = {key: value-common['physical'][key]
                                                 for key, value in state['physical'].items()}
                    state['metrics'] = metric
                    state['cycles'] = {name: events[name][epoch*(config.batches if name == 'physical' else config.batches*config.critic_cycles):
                                                     (epoch+1)*(config.batches if name == 'physical' else config.batches*config.critic_cycles)]
                                       for name in ('physical', 'critic')}
                    internal[kind].append(state)
                    run['epochs'].append(dict(epoch=epoch, actual_updates=actual,
                                              expected_updates=expected, full_work=actual == expected,
                                              scales_before=scales_before,
                                              scales_after=[getattr(branch, name+'_scaler').get_scale()
                                                            for name in ('physical', 'critic')],
                                              metrics=json_safe(metric),
                                              cycles=diagnostic_cycle_summary(state['cycles'])))
            branch = None; release()
            run['common_unchanged'] = state_fingerprint(diagnostic_state(common)) == report['common_fingerprint']
            if not run['common_unchanged']:
                report['status'] = 'diagnostic_invalid'
                run['common_fingerprint_after'] = state_fingerprint(diagnostic_state(common))
                raise DiagnosticError('common snapshot mutated during diagnostic run', report)
    for replay in range(replays):
        for epoch in range(2):
            index = replay*2+epoch
            report['comparisons'].append(diagnostic_compare(internal['original'][index], internal['optimized'][index],
                                                            pristine, device.type, rtol, atol,
                                                            f'cross.replay{replay}.epoch{epoch}'))
    for kind in ('original', 'optimized'):
        for replay in range(1, replays):
            for epoch in range(2):
                report['comparisons'].append(diagnostic_compare(internal[kind][epoch],
                                                                internal[kind][replay*2+epoch], pristine,
                                                                device.type, rtol, atol,
                                                                f'self.{kind}.replay0_vs_{replay}.epoch{epoch}',
                                                                bit_exact=True))
    report['comparison_failure_count'] = sum(bool(item['failed_fields']) for item in report['comparisons'])
    report['work_failure_count'] = sum(not epoch['full_work'] for run in report['runs']
                                       for epoch in run['epochs'])
    return report


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
    parser.add_argument('--diagnostic-replays', type=int, choices=(2, 3),
                        help='untimed paired two-epoch replays; diagnostic output only')
    parser.add_argument('--diagnostic-tf32-ablation', action='store_true',
                        help='in diagnostic mode, replay the same common state again with cuDNN TF32 disabled')
    args = parser.parse_args()
    if args.diagnostic_tf32_ablation and args.diagnostic_replays is None:
        parser.error('--diagnostic-tf32-ablation requires --diagnostic-replays')
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
    if args.diagnostic_replays:
        report['mode'] = 'repeatability_diagnostic'
        report['timing_scope'] = 'untimed diagnostic replays; no performance samples or speedup claim'
        report['cpu_contention'] = 'not applicable to untimed diagnostic replays'
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
        windows = {k: [] for k in ('original', 'optimized')}; comparisons = []; count_reports = {}
        steps = args.representative_steps if representative else args.steps
        try:
            common, calibration = calibrate_common(make, config, device, label, steps,
                                                   args.max_warmup, sync, release)
        except CalibrationError as error:
            report['workloads'][label] = dict(status='calibration_failed', calibration=error.diagnostics,
                                              error=str(error))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
            raise
        if args.diagnostic_replays:
            work = dict(status='diagnostic_only', synthetic=True, calibration=calibration,
                        config=config.to_dict(), geometry=dict(model=list(data.grid.shape), nt=data.sampling.nt,
                                                               shots=data.traces.shape[0], receivers=data.traces.shape[1],
                                                               spacing_m=data.grid.spacing, dt_s=data.sampling.dt),
                        profiles=[], limitation='untimed repeatability diagnostic; no validated performance or equivalence claim')
            report['workloads'][label] = work
            baseline_tf32 = torch.backends.cudnn.allow_tf32
            try:
                for profile_name in (('default', 'cudnn_tf32_off') if args.diagnostic_tf32_ablation else ('default',)):
                    if profile_name == 'cudnn_tf32_off': torch.backends.cudnn.allow_tf32 = False
                    precision = dict(name=profile_name,
                                     cudnn_deterministic=torch.backends.cudnn.deterministic,
                                     cudnn_benchmark=torch.backends.cudnn.benchmark,
                                     cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                                     matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                                     deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                                     device_capability=list(torch.cuda.get_device_capability(device)) if device.type == 'cuda' else None,
                                     cuda_build=torch.version.cuda)
                    driver_query = getattr(torch._C, '_cuda_getDriverVersion', None)
                    try:
                        precision['driver_version'] = driver_query() if device.type == 'cuda' and callable(driver_query) else None
                    except Exception:
                        precision['driver_version'] = None
                    try:
                        trace = diagnostic_replays(make, common, config, device, args.diagnostic_replays,
                                                   sync, release, args.rtol, args.atol)
                    except DiagnosticError as error:
                        trace = error.report
                        work['status'] = 'diagnostic_invalid'
                        work['error'] = str(error)
                        work['profiles'].append(dict(precision=precision, trace=trace))
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
                        raise
                    work['profiles'].append(dict(precision=precision, trace=trace))
            finally:
                torch.backends.cudnn.allow_tf32 = baseline_tf32
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
            print(json.dumps(dict(workload=label, status='diagnostic_only',
                                  profile_comparison_failures=[p['trace']['comparison_failure_count']
                                                               for p in work['profiles']])), flush=True)
            continue
        work = dict(status='running', synthetic=True,
                    geometry=dict(model=list(data.grid.shape), nt=data.sampling.nt,
                                  shots=data.traces.shape[0], receivers=data.traces.shape[1],
                                  spacing_m=data.grid.spacing, dt_s=data.sampling.dt),
                    config=config.to_dict(), steps_per_window=steps, repetitions=args.repeats,
                    warmup_original=calibration['warmup_original'], calibration=calibration,
                    warmup_optimized=[epoch for attempt in calibration['precheck_attempts']
                                      for branch in attempt['branches'] if branch['kind'] == 'optimized'
                                      for epoch in branch['epochs']],
                    windows=windows, numerical_comparisons=comparisons,
                    comparison_diagnostics=[], untimed_counts=count_reports,
                    limitation='synthetic training-loop evidence; not convergence or full Marmousi performance')
        report['workloads'][label] = work
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
            diagnostics = []
            tolerance = (args.rtol, args.atol) if device.type == 'cuda' else (0., 0.)
            velocity_state = (common['physical']['velocity'], left['physical']['velocity'], right['physical']['velocity'])
            for field in fields:
                compare_value(field, left[field], right[field], *tolerance, diagnostics,
                              device.type, velocity_state=velocity_state)
            if len(left['losses']) != len(right['losses']):
                diagnostics.append(dict(path='losses', kind='sequence', passed=False,
                                        reason='epoch counts differ'))
            for epoch_index, (l_epoch, r_epoch) in enumerate(zip(left['losses'], right['losses'])):
                if len(l_epoch) != len(r_epoch):
                    diagnostics.append(dict(path=f'losses[{epoch_index}]', kind='sequence', passed=False,
                                            reason='group counts differ'))
                for group_index, (l_metrics, r_metrics) in enumerate(zip(l_epoch, r_epoch)):
                    for name in ('generator_loss', 'critic_loss'):
                        path = f'losses[{epoch_index}][{group_index}].{name}'
                        if name not in l_metrics or name not in r_metrics:
                            diagnostics.append(dict(path=path, kind='mapping', passed=False,
                                                    reason='metric missing'))
                        else:
                            metric_tolerance = ((2e-7, 2e-6) if device.type == 'cpu' and name == 'critic_loss'
                                                else tolerance)
                            compare_value(path, l_metrics[name], r_metrics[name], *metric_tolerance,
                                          diagnostics, device.type)
            cpu_rng_equal = torch.equal(left['rng']['torch'], right['rng']['torch'])
            cuda_rng_equal = all(torch.equal(l, r) for l, r in zip(left['rng']['cuda'], right['rng']['cuda']))
            python_rng_equal = left['rng']['python'] == right['rng']['python']
            numpy_rng_equal = all(np.array_equal(l, r) for l, r in zip(left['rng']['numpy'], right['rng']['numpy']))
            for stream, equal in (('torch', cpu_rng_equal), ('cuda', cuda_rng_equal),
                                  ('python', python_rng_equal), ('numpy', numpy_rng_equal)):
                diagnostics.append(dict(path=f'rng.{stream}', kind='rng', passed=equal))
            def raw_errors(l_value, r_value):
                try: return json_safe(errors(l_value, r_value))
                except (AssertionError, RuntimeError, ValueError) as error: return dict(unavailable=str(error))
            comparison = {field: raw_errors(left[field], right[field]) for field in fields}
            comparison.update(losses=raw_errors(left['losses'], right['losses']), torch_rng_equal=cpu_rng_equal, cuda_rng_equal=cuda_rng_equal,
                              python_rng_equal=python_rng_equal, numpy_rng_equal=numpy_rng_equal,
                              physical_change_l2=json_safe({kind: finals[kind]['physical_change_l2'] for kind in finals}))
            comparisons.append(comparison)
            record_comparison(report, label, repetition, diagnostics, args.output)
        # Real native forward/adjoint hooks are applied only in an untimed epoch.
        for kind in ('original', 'optimized'):
            branch = make(kind, common); before = counters(branch)
            with instrument(branch) as counts: branch.run(); sync()
            counts['actual_updates'] = assert_steps(before, counters(branch), 1, config)
            expected = dict(solver_forward=config.batches, solver_adjoint=config.batches*(config.critic_cycles+1 if kind == 'original' else 1), critic_forward=config.batches*(3*config.critic_cycles+1))
            if any(counts[k] != v for k, v in expected.items()): raise AssertionError((counts, expected))
            count_reports[kind] = counts; branch = None; release()
        medians = {kind: statistics.median(w['seconds_per_batch'] for w in samples) for kind, samples in windows.items()}
        work.update(status='passed', median_seconds_per_batch=medians,
                    original_over_optimized_time_ratio=medians['original']/medians['optimized'])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(workload=label, median_seconds_per_batch=medians, numerical_comparison='passed', counts=count_reports)), flush=True)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='User provided device_type.*CUDA is not available.*')
        main()
