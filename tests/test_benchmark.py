"""Pinned CUDA startup contracts checked without allocating a GPU."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fwigan.checkpoints import rng_snapshot, restore_rng, seed_all
from fwigan.inversion.adversarial import gradient_penalty as production_gradient_penalty
from reference import gradient_penalty as frozen_gradient_penalty


class MetadataQueryReached(Exception):
    pass


def load_benchmark():
    path = Path(__file__).resolve().parents[1] / 'examples' / 'benchmark.py'
    spec = importlib.util.spec_from_file_location('benchmark_calibration_regression', path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    return benchmark


def test_frozen_and_production_penalties_consume_identical_seeded_beta_stream(monkeypatch):
    """A shared checkpoint must give both real penalty functions the same draws."""
    entry_rng = rng_snapshot()
    try:
        seed_all(42)
        common_rng = rng_snapshot()
        observed = torch.linspace(-1, 1, 2 * 128 * 60).reshape(1, 2, 128, 60)
        predicted = observed.flip(-1).clone()

        def critic(values):
            return values.square().flatten(1).sum(dim=1)

        def run(penalty, frozen):
            restore_rng(common_rng)
            betas = []
            real_rand = torch.rand

            def capture_rand(*args, **kwargs):
                value = real_rand(*args, **kwargs)
                betas.append(value.clone())
                return value

            with monkeypatch.context() as patch:
                patch.setattr(torch, 'rand', capture_rand)
                for _ in range(12):
                    if frozen:
                        penalty(critic, observed, predicted, device='cpu')
                    else:
                        penalty(critic, observed, predicted)
            return betas, rng_snapshot()

        frozen_betas, frozen_end = run(frozen_gradient_penalty, True)
        production_betas, production_end = run(production_gradient_penalty, False)

        assert len(frozen_betas) == len(production_betas) == 12
        assert all(beta.shape == (1, 1, 1, 1) for beta in frozen_betas + production_betas)
        assert all(torch.equal(left, right) for left, right in zip(frozen_betas, production_betas))
        assert not torch.equal(common_rng['torch'], frozen_end['torch'])
        for key in ('python', 'cudnn_deterministic', 'cudnn_benchmark'):
            assert frozen_end[key] == production_end[key]
        assert torch.equal(frozen_end['torch'], production_end['torch'])
        assert len(frozen_end['cuda']) == len(production_end['cuda'])
        assert all(torch.equal(left, right) for left, right in zip(frozen_end['cuda'], production_end['cuda']))
        assert frozen_end['numpy'][0] == production_end['numpy'][0]
        np.testing.assert_array_equal(frozen_end['numpy'][1], production_end['numpy'][1])
        assert frozen_end['numpy'][2:] == production_end['numpy'][2:]
    finally:
        restore_rng(entry_rng)


class FakeState:
    def __init__(self, **state):
        self.state = state

    def state_dict(self):
        return dict(self.state)

    def load_state_dict(self, state):
        self.state = dict(state)

    def get_scale(self):
        return self.state['scale']


class FakeOptimizer(FakeState):
    def __init__(self):
        super().__init__(moment=0)
        self.completed_steps = 0


class FakeBranch:
    """Only the GPU-dependent epoch is simulated; benchmark snapshots stay real."""
    def __init__(self, kind, thresholds, benchmark):
        self.kind, self.thresholds, self.benchmark = kind, thresholds, benchmark
        self.value = torch.tensor([1.0])
        self.critic = FakeState(weight=torch.tensor([2.0]))
        self.observed = torch.tensor([3.0])
        self.counters = dict(physical_updates=0, critic_updates=0,
                             physical_skipped=0, critic_skipped=0)
        for name in ('physical', 'critic'):
            setattr(self, name + '_optimizer', FakeOptimizer())
            setattr(self, name + '_scheduler', FakeState(epoch=0))
            setattr(self, name + '_scaler', FakeState(scale=65536.0, growth_tracker=0))

    def physical(self, gradients=False):
        return {} if gradients else {'velocity': self.value.clone()}

    def load(self, state):
        self.value = state['physical']['velocity'].clone()
        self.critic.load_state_dict(state['critic'])
        self.observed = state['observed'].clone()
        self.counters = dict(state['counters'])
        for name in ('physical', 'critic'):
            getattr(self, name + '_optimizer').load_state_dict(state[name + '_optimizer'])
            getattr(self, name + '_scheduler').load_state_dict(state[name + '_scheduler'])
            getattr(self, name + '_scaler').load_state_dict(state[name + '_scaler'])
            getattr(self, name + '_optimizer').completed_steps = self.counters[name + '_updates']

    def run(self):
        draw = torch.rand(())  # A real epoch advances the restored RNG stream.
        for name, count in (('physical', 1), ('critic', 6)):
            scaler = getattr(self, name + '_scaler')
            optimizer = getattr(self, name + '_optimizer')
            if scaler.get_scale() > self.thresholds[self.kind][name]:
                scaler.state['scale'] /= 2
                self.counters[name + '_skipped'] += count
            else:
                optimizer.completed_steps += count
                optimizer.state['moment'] += count
                getattr(self, name + '_scheduler').state['epoch'] += 1
                self.counters[name + '_updates'] += count
                self.value += draw * count


def fake_factory(benchmark, optimized_thresholds, starts, original_thresholds=None):
    thresholds = {'original': original_thresholds or {'physical': 8192, 'critic': 64},
                  'optimized': optimized_thresholds}

    def make(kind, common=None):
        branch = FakeBranch(kind, thresholds, benchmark)
        if common is not None:
            branch.load(common)
            benchmark.restore_rng(common['rng'])
            starts.append((kind, benchmark.snapshot(branch, torch.device('cpu')),
                           benchmark.rng_state(torch.device('cpu'))))
        return branch

    return make


def assert_same_common(left, right):
    for key in ('physical', 'critic', 'observed', 'physical_optimizer',
                'critic_optimizer', 'physical_scheduler', 'critic_scheduler',
                'physical_scaler', 'critic_scaler', 'counters'):
        if key == 'counters':
            assert left[key] == right[key]
        else:
            load_benchmark().require_close(left[key], right[key], 0, 0)
    for key in ('torch', 'cuda', 'python'):
        if key == 'torch':
            assert torch.equal(left['rng'][key], right['rng'][key])
        else:
            assert left['rng'][key] == right['rng'][key]
    assert left['rng']['numpy'][0] == right['rng']['numpy'][0]
    np.testing.assert_array_equal(left['rng']['numpy'][1], right['rng']['numpy'][1])
    assert left['rng']['numpy'][2:] == right['rng']['numpy'][2:]


def test_joint_calibration_uses_natural_backoff_and_one_verified_common():
    benchmark = load_benchmark()
    benchmark.seed(12)
    starts = []
    config = SimpleNamespace(batches=1, critic_cycles=6)
    make = fake_factory(benchmark, {'physical': 2048, 'critic': 32}, starts)

    common, diagnostics = benchmark.calibrate_common(
        make, config, torch.device('cpu'), 'fake', steps=3, max_warmup=24,
        sync=lambda: None, release=lambda: None)

    assert diagnostics['precheck_epochs'] == 3
    assert all(row['stable_epochs'] == 3 for row in diagnostics['precheck_attempts'][-1]['branches'])
    assert diagnostics['warmup_original'][0]['scales_before'] == [65536.0, 65536.0]
    assert any(row['skipped_updates'] != [0, 0] for row in diagnostics['warmup_original'])
    assert any(row['kind'] == 'optimized' and row['skipped_updates'] != [0, 0]
               for attempt in diagnostics['precheck_attempts'] for row in attempt['branches'])
    assert common['physical_scaler']['scale'] == 2048
    assert common['critic_scaler']['scale'] == 32
    assert len(starts) >= 4
    for original, optimized in zip(starts[::2], starts[1::2]):
        assert original[0] == 'original' and optimized[0] == 'optimized'
        assert_same_common(original[1], optimized[1])
        assert_same_common(original[1], {**original[1], 'rng': original[2]})
        assert_same_common(optimized[1], {**optimized[1], 'rng': optimized[2]})
    assert_same_common(common, starts[-2][1])
    assert_same_common(common, starts[-1][1])
    for kind in ('original', 'optimized'):
        branch = make(kind, common)
        before = benchmark.counters(branch)
        for _ in range(3):
            branch.run()
        assert benchmark.assert_steps(before, benchmark.counters(branch), 3, config) == {'physical': 3, 'critic': 18}


def test_joint_calibration_reports_bounded_failure_after_skips():
    benchmark = load_benchmark()
    benchmark.seed(12)
    make = fake_factory(benchmark, {'physical': 0, 'critic': 0}, [],
                        original_thresholds={'physical': 65536, 'critic': 65536})
    with pytest.raises(RuntimeError, match='fake.*joint calibration.*2 attempts') as failure:
        benchmark.calibrate_common(make, SimpleNamespace(batches=1, critic_cycles=6),
                                   torch.device('cpu'), 'fake', steps=1, max_warmup=2,
                                   sync=lambda: None, release=lambda: None)
    assert failure.value.diagnostics['precheck_epochs'] == 2
    assert len(failure.value.diagnostics['precheck_attempts']) == 2
    assert failure.value.diagnostics['precheck_attempts'][-1]['branches'][1]['skipped_updates'] == [1, 6]


def velocity_probe():
    common = torch.tensor([2100.], dtype=torch.float32)
    ulp = torch.nextafter(common, torch.full_like(common, float('inf'))) - common
    midpoint = ulp * 464.5
    perturb = torch.tensor([2.**-25], dtype=torch.float32)
    finals = []
    for gradient in (midpoint-perturb, midpoint+perturb):
        parameter = torch.nn.Parameter(common.clone())
        optimizer = torch.optim.SGD([parameter], lr=1.)
        parameter.grad = gradient
        optimizer.step()
        finals.append(parameter.detach().clone())
    return common, finals, ulp


def compare_probe(benchmark, path, common, finals, device_type='cuda'):
    diagnostics = []
    benchmark.compare_value(path, finals[0]-common, finals[1]-common,
                            2e-4 if device_type == 'cuda' else 0.,
                            2e-5 if device_type == 'cuda' else 0.,
                            diagnostics, device_type, velocity_state=(common, *finals))
    json.dumps(diagnostics, allow_nan=False)
    return diagnostics[0]


def test_cuda_velocity_change_accepts_only_one_local_model_ulp():
    benchmark = load_benchmark()
    common, finals, ulp = velocity_probe()
    assert float(ulp[0]) == 0.000244140625
    accepted = compare_probe(benchmark, 'physical_changes.velocity', common, finals)
    assert accepted['passed'] and accepted['old_failed'] == 1
    assert accepted['rescued_by_ulp'] == 1 and accepted['still_failed'] == 0
    assert accepted['floor_min'] == accepted['floor_max'] == float(ulp[0])
    assert accepted['samples'][0]['common'] == 2100.
    assert accepted['samples'][0]['final_original'] == float(finals[0][0])
    assert accepted['samples'][0]['final_optimized'] == float(finals[1][0])

    two_ulp_final = finals[0] - 2*ulp
    rejected = compare_probe(benchmark, 'physical_changes.velocity', common,
                             [finals[0], two_ulp_final])
    assert not rejected['passed'] and rejected['still_failed'] == 1


def test_velocity_floor_tracks_model_magnitude_and_exponent_boundary():
    benchmark = load_benchmark()
    common = torch.tensor([2100., 4096., 2048.], dtype=torch.float32)
    upper = torch.nextafter(common, torch.full_like(common, float('inf')))
    lower = torch.nextafter(common, torch.full_like(common, -float('inf')))
    result = compare_probe(benchmark, 'physical_changes.velocity', common, [lower, upper])
    assert result['floor_min'] == 2**-12
    assert result['floor_max'] == 2**-11
    assert result['still_failed'] >= 1  # The two stored finals span more than one local ULP.
    boundary = compare_probe(benchmark, 'physical_changes.velocity', common[-1:],
                             [lower[-1:], common[-1:]])
    assert boundary['floor_min'] == 2**-12
    assert boundary['rescued_by_ulp'] == 1


@pytest.mark.parametrize('path,device_type', [
    ('physical_changes.velocity', 'cpu'),
    ('physical_changes.source', 'cuda'),
    ('gradients.velocity', 'cuda'),
    ('physical_optimizer.state.0.exp_avg', 'cuda'),
    ('critic.layer', 'cuda'),
])
def test_velocity_floor_does_not_change_other_comparisons(path, device_type):
    benchmark = load_benchmark()
    common, finals, _ = velocity_probe()
    result = compare_probe(benchmark, path, common, finals, device_type)
    assert not result['passed'] and result['old_failed'] == 1
    assert result.get('rescued_by_ulp', 0) == 0


@pytest.mark.parametrize('problem', ['dtype', 'shape', 'nonfinite', 'floor_nonfinite'])
def test_velocity_floor_rejects_malformed_or_nonfinite_tensors(problem):
    benchmark = load_benchmark()
    common, finals, _ = velocity_probe()
    if problem == 'dtype': finals[1] = finals[1].double()
    if problem == 'shape': finals[1] = finals[1].reshape(1, 1)
    if problem == 'nonfinite': finals[1][0] = float('nan')
    if problem == 'floor_nonfinite':
        common = torch.tensor([torch.finfo(torch.float32).max])
        finals = [common.clone(), common.clone()]
    diagnostics = []
    benchmark.compare_value('physical_changes.velocity', finals[0]-common, finals[1]-common,
                            2e-4, 2e-5, diagnostics, 'cuda', velocity_state=(common, *finals))
    assert not diagnostics[0]['passed']
    json.dumps(diagnostics, allow_nan=False)


def test_numerical_failure_json_retains_field_and_completed_work(tmp_path):
    benchmark = load_benchmark()
    common, finals, _ = velocity_probe()
    diagnostics = [compare_probe(benchmark, path, common, finals)
                   for path in ('physical_changes.source', 'critic.layer')]
    output = tmp_path / 'partial.json'
    report = {'workloads': {'tiny-clean': {'status': 'running',
               'calibration': {'precheck_attempts': [1]},
               'windows': {'original': [{'actual_updates': {'physical': 1, 'critic': 6}, 'seconds': .1}],
                           'optimized': [{'actual_updates': {'physical': 1, 'critic': 6}, 'seconds': .09}]}}}}
    with pytest.raises(AssertionError, match='physical_changes.source'):
        benchmark.record_comparison(report, 'tiny-clean', 0, diagnostics, output)
    saved = json.loads(output.read_text())['workloads']['tiny-clean']
    assert saved['status'] == 'numerical_failed'
    assert saved['calibration']['precheck_attempts'] == [1]
    assert saved['windows']['original'][0]['actual_updates']['critic'] == 6
    assert saved['comparison_diagnostics'][0]['fields'][0]['path'] == 'physical_changes.source'
    assert 'critic.layer' in saved['error']


def test_diagnostic_replays_compare_both_kinds_without_timing_claims():
    benchmark = load_benchmark()
    benchmark.seed(12)
    config = SimpleNamespace(batches=1, critic_cycles=6)
    starts = []
    make = fake_factory(benchmark, {'physical': 2048, 'critic': 32}, starts)
    common, _ = benchmark.calibrate_common(make, config, torch.device('cpu'), 'fake',
                                           steps=2, max_warmup=24,
                                           sync=lambda: None, release=lambda: None)
    starts.clear()

    result = benchmark.diagnostic_replays(make, common, config, torch.device('cpu'),
                                          replays=3, sync=lambda: None, release=lambda: None)

    assert result['status'] == 'diagnostic_only'
    assert [run['kind'] for run in result['runs']] == ['original', 'optimized'] * 3
    assert all(run['start_valid'] and run['common_unchanged'] for run in result['runs'])
    assert all(len(run['epochs']) == 2 for run in result['runs'])
    assert len(result['comparisons']) == 2 * (3 + 2 + 2)
    assert all(not entry['failed_fields'] for entry in result['comparisons'])
    assert 'median_seconds_per_batch' not in result
    assert 'seconds' not in json.dumps(result)
    json.dumps(result, allow_nan=False)


def test_diagnostic_replays_keep_all_material_comparison_failures():
    benchmark = load_benchmark()
    benchmark.seed(12)
    config = SimpleNamespace(batches=1, critic_cycles=6)
    base_make = fake_factory(benchmark, {'physical': 65536, 'critic': 65536}, [],
                             original_thresholds={'physical': 65536, 'critic': 65536})
    common, _ = benchmark.calibrate_common(base_make, config, torch.device('cpu'), 'fake',
                                           steps=2, max_warmup=2,
                                           sync=lambda: None, release=lambda: None)

    def make(kind, state=None):
        branch = base_make(kind, state)
        if kind == 'optimized' and state is not None:
            original_run = branch.run
            def run():
                original_run()
                branch.value += 1
            branch.run = run
        return branch

    result = benchmark.diagnostic_replays(make, common, config, torch.device('cpu'),
                                          replays=3, sync=lambda: None, release=lambda: None)
    assert result['status'] == 'diagnostic_only'
    assert len(result['comparisons']) == 14
    assert result['comparison_failure_count'] == 6
    assert all('physical.velocity' in entry['failed_fields']
               for entry in result['comparisons'][:6])
    assert all(not entry['failed_fields'] for entry in result['comparisons'][6:])
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('fault', ['loaded_state', 'common_mutation'])
def test_diagnostic_replays_reject_bad_start_or_common_mutation(fault):
    benchmark = load_benchmark()
    benchmark.seed(12)
    config = SimpleNamespace(batches=1, critic_cycles=6)
    base_make = fake_factory(benchmark, {'physical': 65536, 'critic': 65536}, [],
                             original_thresholds={'physical': 65536, 'critic': 65536})
    common, _ = benchmark.calibrate_common(base_make, config, torch.device('cpu'), 'fake',
                                           steps=2, max_warmup=2,
                                           sync=lambda: None, release=lambda: None)

    def make(kind, state=None):
        branch = base_make(kind, state)
        if fault == 'loaded_state' and state is not None:
            branch.value += 1
        if fault == 'common_mutation' and state is not None:
            original_run = branch.run
            def run():
                original_run()
                state['physical']['velocity'] += 1
            branch.run = run
        return branch

    with pytest.raises(benchmark.DiagnosticError, match='loaded state|common snapshot') as failure:
        benchmark.diagnostic_replays(make, common, config, torch.device('cpu'),
                                     replays=2, sync=lambda: None, release=lambda: None)
    assert failure.value.report['status'] == 'diagnostic_invalid'
    json.dumps(failure.value.report, allow_nan=False)


def test_cuda_benchmark_memory_metadata_queries_current_device(tmp_path, monkeypatch):
    """The real pinned index validator rejects cuda without an explicit index."""
    path=Path(__file__).resolve().parents[1]/'examples'/'benchmark.py'
    spec=importlib.util.spec_from_file_location('benchmark_device_regression',path)
    benchmark=importlib.util.module_from_spec(spec);spec.loader.exec_module(benchmark)
    monkeypatch.setattr(sys,'argv',[str(path),'--device','cuda','--suite','all',
        '--output',str(tmp_path/'benchmark.json')])
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'current_device',lambda:3)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda device:'validation GPU')
    monkeypatch.setattr(torch,'set_num_threads',lambda count:None)
    monkeypatch.setattr(torch,'set_num_interop_threads',lambda count:None)
    monkeypatch.setattr(torch.backends.cudnn,'deterministic',torch.backends.cudnn.deterministic)
    monkeypatch.setattr(torch.backends.cudnn,'benchmark',torch.backends.cudnn.benchmark)
    class Runtime:
        def cudaMemGetInfo(self,index):
            # Keep real Torch1.13 mem_get_info and its strict index validator.
            # Only the hardware runtime is replaced; cuda:0 must also fail
            # when the selected current device is 3.
            assert index==3
            raise MetadataQueryReached
    monkeypatch.setattr(torch.cuda,'cudart',lambda:Runtime())
    with pytest.raises(MetadataQueryReached):benchmark.main()
    assert not (tmp_path/'benchmark.json').exists()
