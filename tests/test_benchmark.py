"""Pinned CUDA startup contracts checked without allocating a GPU."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


class MetadataQueryReached(Exception):
    pass


def load_benchmark():
    path = Path(__file__).resolve().parents[1] / 'examples' / 'benchmark.py'
    spec = importlib.util.spec_from_file_location('benchmark_calibration_regression', path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    return benchmark


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
