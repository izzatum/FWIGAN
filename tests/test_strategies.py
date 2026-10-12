import copy
import importlib
import math
import warnings

import pytest
import torch
from reference import Critic
from reference.strategy import make_reference
from fwigan.config import ExperimentConfig
from fwigan.data import make_tiny_data, legacy_normalize, to_critic_layout
from fwigan.parameters import InversionParameters
from fwigan.physics import LegacyDeepwave


def inversion():
    assert importlib.util.find_spec('fwigan.inversion') is not None, 'inversion strategies are missing'
    return importlib.import_module('fwigan.inversion')


def parameters(data, config):
    source = data.source if config.method == 'fwi' or data.initial_source is None else data.initial_source
    return InversionParameters(data.initial_velocity, source, torch.tensor(config.sigma_init),
                               config.train_source, config.train_sigma)


def adam_equal(actual, expected):
    a, b = actual.state_dict()['state'], expected.state_dict()['state']
    assert a.keys() == b.keys()
    for k in a:
        for field in a[k]:
            left, right = a[k][field], b[k][field]
            if torch.is_tensor(left):
                # Shared source uses the canonical ordering in production.
                if left.shape != right.shape: right = right.permute(1,2,0)
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            else: assert left == right


@pytest.mark.parametrize('noise', [False, True])
def test_original_loop_losses_gradients_updates_rng_and_actual_adjoints(noise, monkeypatch):
    f = inversion()
    data = make_tiny_data(nt=64)
    config = ExperimentConfig(noise=noise, sigma_init=.08, batches=1, critic_cycles=6)
    observed = to_critic_layout(legacy_normalize(data.traces))
    if noise:
        torch.manual_seed(71)
        observed = observed + .05 * torch.randn_like(observed)
    torch.manual_seed(43)
    reference = make_reference(data, config)
    torch.manual_seed(43)
    actual = f.FWIGANStrategy(parameters(data, config), LegacyDeepwave(), data, config,
                             observed=observed.squeeze(0).permute(0,2,1))
    for a,b in zip(actual.critic.parameters(), reference[1].parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    from deepwave.scalar import scalar
    counts = {'forward':0, 'backward':0}
    original_f, original_b = scalar.PropagatorFunction.forward, scalar.PropagatorFunction.backward
    def forward(*args, **kwargs):
        counts['forward'] += 1
        return original_f(*args, **kwargs)
    def backward(*args, **kwargs):
        counts['backward'] += 1
        return original_b(*args, **kwargs)
    monkeypatch.setattr(scalar.PropagatorFunction, 'forward', staticmethod(forward))
    monkeypatch.setattr(scalar.PropagatorFunction, 'backward', staticmethod(backward))
    torch.manual_seed(17)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='User provided device_type.*CUDA is not available.*')
        gl,cl = reference[-1](observed)
    assert counts == {'forward':1, 'backward':7}
    expected_rng = torch.get_rng_state()
    counts.update(forward=0, backward=0)
    torch.manual_seed(17)
    physical_grad_seen=[]
    handle=actual.critic.register_forward_pre_hook(lambda *_: physical_grad_seen.append(actual.parameters.velocity.grad is not None))
    metrics = actual.step([0,1]); handle.remove()
    assert counts == {'forward':1, 'backward':1}
    assert not any(physical_grad_seen)
    assert all(p.grad is None and p.requires_grad for p in actual.critic.parameters())
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert metrics['generator_loss'] == gl
    assert metrics['critic_loss'] == pytest.approx(cl, abs=2e-6)
    assert metrics['physical_updates'] == 1 and metrics['critic_updates'] == 6
    assert metrics['physical_skipped'] == metrics['critic_skipped'] == 0
    for a,b in [(actual.parameters.velocity, reference[0].model),
                (actual.parameters.source, reference[0].source.permute(1,2,0)),
                (actual.parameters.sigma, reference[0].sigma)]:
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.testing.assert_close(actual.parameters.velocity.grad, reference[0].model.grad,rtol=0,atol=0)
    torch.testing.assert_close(actual.parameters.source.grad, reference[0].source.grad.permute(1,2,0),rtol=0,atol=0)
    if noise: torch.testing.assert_close(actual.parameters.sigma.grad, reference[0].sigma.grad,rtol=0,atol=0)
    for a,b in zip(actual.critic.parameters(), reference[1].parameters()): torch.testing.assert_close(a,b,rtol=0,atol=0)
    adam_equal(actual.physical_optimizer, reference[2]); adam_equal(actual.critic_optimizer, reference[3])
    assert all(math.isfinite(v) for v in metrics.values())


@pytest.mark.parametrize('geometry', [(0,64,60),(2,31,60),(2,64,31),(2,64,64)])
def test_critic_rejects_invalid_pooling_and_dense_geometry(geometry):
    with pytest.raises(ValueError): inversion().LegacyCritic(*geometry)


def test_critic_layout_and_frozen_initialization():
    f=inversion()
    torch.manual_seed(3); expected=Critic(2,65,63)
    torch.manual_seed(3); actual=f.LegacyCritic(2,65,63)
    x=torch.randn(2,63,65)
    torch.testing.assert_close(actual.from_canonical(x),expected(to_critic_layout(x)),rtol=0,atol=0)
    with pytest.raises(ValueError): actual.from_canonical(x[:1])


def test_invalid_group_fails_before_forward_and_observed_is_used_unchanged():
    f=inversion();data=make_tiny_data(nt=64);config=ExperimentConfig(batches=1,critic_cycles=1)
    target=legacy_normalize(data.traces)*.7
    strategy=f.FWIGANStrategy(parameters(data,config),LegacyDeepwave(),data,config,observed=target)
    torch.testing.assert_close(strategy.observed,target,rtol=0,atol=0)
    with pytest.raises(ValueError): strategy.step([0])
    with pytest.raises(ValueError): strategy.step([0,0])
    with pytest.raises(ValueError): strategy.step([0,2])
    with pytest.raises(ValueError): f.FWIGANStrategy(parameters(data,config),LegacyDeepwave(),data,config,observed=torch.ones(1))
    torch.manual_seed(19)
    # Only critic initialization consumes RNG, even when fake noise is enabled.
    noisy=ExperimentConfig(noise=True,batches=1,critic_cycles=1)
    a=f.FWIGANStrategy(parameters(data,noisy),LegacyDeepwave(),data,noisy)
    actual_rng=torch.get_rng_state()
    torch.manual_seed(19); f.LegacyCritic(2,64,60)
    assert torch.equal(actual_rng,torch.get_rng_state())
    torch.testing.assert_close(a.observed,legacy_normalize(data.traces),rtol=0,atol=0)


def test_fwi_partial_batches_match_full_gradient_fixed_source_and_projection():
    f=inversion();data=make_tiny_data(nt=64,shots=3,receivers=8)
    strategies=[]
    for batch in (3,2):
        config=ExperimentConfig(method='fwi',batch_size=batch,velocity_lr=.5,regularization=1e-5)
        strategy=f.FWIStrategy(parameters(data,config),LegacyDeepwave(),data,config)
        with torch.no_grad(): strategy.parameters.velocity[5,5]+=10
        strategies.append(strategy)
    metrics=[s.step_epoch() for s in strategies]
    torch.testing.assert_close(strategies[0].parameters.velocity.grad, strategies[1].parameters.velocity.grad,rtol=2e-5,atol=1e-8)
    torch.testing.assert_close(strategies[0].parameters.velocity,strategies[1].parameters.velocity,rtol=1e-6,atol=1e-4)
    for strategy in strategies:
        torch.testing.assert_close(strategy.parameters.source,data.source,rtol=0,atol=0)
        assert not strategy.parameters.source.requires_grad
    assert metrics[0]['data_loss'] == pytest.approx(metrics[1]['data_loss'],rel=2e-6)
    config=ExperimentConfig(method='fwi',velocity_max=2000.01,velocity_lr=5.)
    strategy=f.FWIStrategy(parameters(data,config),LegacyDeepwave(),data,config)
    strategy.step_epoch()
    assert strategy.parameters.velocity.min()>=config.velocity_min
    assert strategy.parameters.velocity.max()<=config.velocity_max


def test_fwi_directional_gradient_and_data_fit_improvement():
    f=inversion();data=make_tiny_data(nt=64,shots=2,receivers=8)
    config=ExperimentConfig(method='fwi',velocity_lr=2.)
    strategy=f.FWIStrategy(parameters(data,config),LegacyDeepwave(),data,config)
    def loss(v): return (LegacyDeepwave().predict(v,data.source,data.acquisition,data.grid,data.sampling)-data.traces).square().mean()
    v=data.initial_velocity.clone().requires_grad_();base=loss(v);base.backward()
    direction=v.grad / v.grad.norm()
    with torch.no_grad(): numeric=(loss(v+direction*.5)-loss(v-direction*.5))/1.
    analytic=(v.grad*direction).sum()
    assert analytic.abs()>0
    torch.testing.assert_close(numeric,analytic,rtol=.025,atol=1e-7)
    for _ in range(6): strategy.step_epoch()
    assert loss(strategy.parameters.velocity).item()<base.item()*.9


@pytest.mark.parametrize('variant', ['fwi','fwigan-clean','fwigan-noisy'])
def test_strategy_state_roundtrip_named_lrs_and_scheduler_counters(variant):
    method='fwi' if variant=='fwi' else 'fwigan'
    f=inversion();data=make_tiny_data(nt=64);config=ExperimentConfig(method=method,noise=variant=='fwigan-noisy',batches=1,critic_cycles=1,physical_scheduler_step_size=1,critic_scheduler_step_size=1)
    a=f.build_strategy(parameters(data,config),LegacyDeepwave(),data,config)
    a.finish_epoch()
    assert a.physical_optimizer.param_groups[0]['lr']==config.velocity_lr
    if method=='fwi': a.step_epoch()
    else: a.step([0,1])
    a.finish_epoch()
    b=f.build_strategy(parameters(data,config),LegacyDeepwave(),data,config)
    b.parameters.load_state_dict(a.parameters.state_dict());b.load_state_dict(copy.deepcopy(a.state_dict()))
    assert [(g['name'],g['lr']) for g in b.physical_optimizer.param_groups]==[(g['name'],g['lr']) for g in a.physical_optimizer.param_groups]
    torch.manual_seed(33)
    ma=a.step_epoch() if method=='fwi' else a.step([0,1])
    torch.manual_seed(33)
    mb=b.step_epoch() if method=='fwi' else b.step([0,1])
    assert ma==mb
    for key in a.parameters.state_dict(): torch.testing.assert_close(a.parameters.state_dict()[key],b.parameters.state_dict()[key],rtol=0,atol=0)
    bad=copy.deepcopy(a.state_dict());bad['schema']=999
    with pytest.raises(ValueError): b.load_state_dict(bad)


def test_fwi_requires_known_source_when_initial_source_differs():
    f=inversion();data=make_tiny_data(nt=64,receivers=8)
    data.initial_source=data.source*.8
    config=ExperimentConfig(method='fwi')
    strategy=f.FWIStrategy(parameters(data,config),LegacyDeepwave(),data,config)
    torch.testing.assert_close(strategy.parameters.source,data.source,rtol=0,atol=0)
    guessed=InversionParameters(data.initial_velocity,data.initial_source,torch.tensor(1.),False,False)
    with pytest.raises(ValueError,match='known'): f.FWIStrategy(guessed,LegacyDeepwave(),data,config)


def test_fwi_regularization_is_counted_once_and_has_expected_gradient():
    f=inversion();data=make_tiny_data(nt=64,shots=3,receivers=8)
    config=ExperimentConfig(method='fwi',batch_size=2,regularization=.1)
    # Make the exact current physical model the target, leaving only the penalty gradient.
    v=data.initial_velocity.clone();v[3,3]+=10
    data.initial_velocity=v
    data.traces=LegacyDeepwave().predict(v,data.source,data.acquisition,data.grid,data.sampling)
    strategy=f.FWIStrategy(parameters(data,config),LegacyDeepwave(),data,config)
    metrics=strategy.step_epoch()
    expected=torch.zeros_like(v)
    expected[3,3]=.1*(40/((v.shape[0]-1)*v.shape[1])+40/(v.shape[0]*(v.shape[1]-1)))
    expected[2,3]=expected[4,3]=-.1*20/((v.shape[0]-1)*v.shape[1])
    expected[3,2]=expected[3,4]=-.1*20/(v.shape[0]*(v.shape[1]-1))
    torch.testing.assert_close(strategy.parameters.velocity.grad,expected,rtol=2e-6,atol=1e-9)
    assert metrics['data_loss']==0
    assert metrics['regularization_loss']==pytest.approx(.1*(200/2730+200/2760),rel=1e-6)


def test_skipped_steps_have_host_counters_and_do_not_start_scheduler():
    f=inversion();data=make_tiny_data(nt=64);config=ExperimentConfig(batches=1,critic_cycles=2,physical_scheduler_step_size=1,critic_scheduler_step_size=1)
    strategy=f.FWIGANStrategy(parameters(data,config),LegacyDeepwave(),data,config)
    class SkippingScaler:
        def scale(self, loss): return loss
        def unscale_(self, optimizer): pass
        def step(self, optimizer): pass
        def update(self): pass
    strategy.physical_scaler=SkippingScaler();strategy.critic_scaler=SkippingScaler()
    before=copy.deepcopy(strategy.parameters.state_dict())
    critic_before=copy.deepcopy(strategy.critic.state_dict())
    metrics=strategy.step([0,1]);strategy.finish_epoch()
    assert metrics['physical_updates']==metrics['critic_updates']==0
    assert metrics['physical_skipped']==1 and metrics['critic_skipped']==2
    assert strategy.physical_optimizer.param_groups[0]['lr']==config.velocity_lr
    assert strategy.critic_optimizer.param_groups[0]['lr']==config.critic_lr
    for key,value in before.items(): torch.testing.assert_close(strategy.parameters.state_dict()[key],value,rtol=0,atol=0)
    for key,value in critic_before.items(): torch.testing.assert_close(strategy.critic.state_dict()[key],value,rtol=0,atol=0)


def test_nonphysical_state_includes_observed_and_rejects_changed_optimizer_names():
    f=inversion();data=make_tiny_data(nt=64);config=ExperimentConfig(batches=1,critic_cycles=1)
    a=f.FWIGANStrategy(parameters(data,config),LegacyDeepwave(),data,config,observed=legacy_normalize(data.traces)*.6)
    b=f.FWIGANStrategy(parameters(data,config),LegacyDeepwave(),data,config)
    state=a.state_dict();b.load_state_dict(state)
    torch.testing.assert_close(b.observed,a.observed,rtol=0,atol=0)
    bad=copy.deepcopy(state);bad['physical_optimizer']['param_groups'][0]['name']='source'
    with pytest.raises(ValueError,match='named'): b.load_state_dict(bad)
    def tensors(value):
        if torch.is_tensor(value): yield value
        elif isinstance(value,dict):
            for v in value.values(): yield from tensors(v)
        elif isinstance(value,(list,tuple)):
            for v in value: yield from tensors(v)
    assert all(t.device.type=='cpu' for t in tensors(state))


def test_legacy_groups_derive_from_shots_and_strided_batches_before_solving():
    from fwigan.domain import ObservationData, Acquisition
    f=inversion();tiny=make_tiny_data(nt=32)
    data=ObservationData(tiny.traces.repeat(15,1,1),
        Acquisition(tiny.acquisition.sources.repeat(15,1,1),tiny.acquisition.receivers.repeat(15,1,1)),
        tiny.grid,tiny.sampling,tiny.source,tiny.initial_velocity)
    config=ExperimentConfig(critic_cycles=1)  # 30 shots / 6 groups, batch_size=2 for FWI only.
    class ReachedForward(Exception): pass
    class ProbeForward:
        def predict(self,velocity,source,acquisition,grid,sampling):
            assert acquisition.sources.shape[0]==5
            raise ReachedForward
    strategy=f.FWIGANStrategy(parameters(data,config),ProbeForward(),data,config)
    assert strategy.shots_per_batch==5
    assert strategy.critic.geometry==(5,32,60)
    with pytest.raises(ValueError): strategy.step([0,1])
    with pytest.raises(ReachedForward): strategy.step(slice(0,None,6))
    config.batches=7
    with pytest.raises(ValueError,match='divisible'): f.FWIGANStrategy(parameters(data,config),ProbeForward(),data,config)


@pytest.mark.parametrize('value', [999.,6001.])
def test_fwi_rejects_initial_velocity_outside_bounds(value):
    f=inversion();data=make_tiny_data(nt=64,receivers=8);config=ExperimentConfig(method='fwi')
    p=parameters(data,config)
    with torch.no_grad(): p.velocity[0,0]=value
    with pytest.raises(ValueError,match='bounds'): f.FWIStrategy(p,LegacyDeepwave(),data,config)


def test_fwi_rejects_nonfinite_objective_before_optimizer_corrupts_model():
    f=inversion();data=make_tiny_data(nt=64,receivers=8);data.traces.fill_(1e30)
    config=ExperimentConfig(method='fwi',batch_size=1)
    p=parameters(data,config);strategy=f.FWIStrategy(p,LegacyDeepwave(),data,config)
    before=p.velocity.detach().clone()
    with pytest.raises(ValueError,match='nonfinite'): strategy.step_epoch()
    torch.testing.assert_close(p.velocity,before,rtol=0,atol=0)
    assert not strategy.physical_optimizer.state
    assert strategy.counters['physical_updates']==0


def test_raw_source_directional_finite_difference():
    inversion();data=make_tiny_data(nt=64,shots=2,receivers=8)
    source=(data.source*.8).clone().requires_grad_()
    def loss(s):
        return (LegacyDeepwave().predict(data.initial_velocity,s,data.acquisition,data.grid,data.sampling)-data.traces).square().mean()
    value=loss(source);value.backward()
    direction=source.grad/source.grad.norm()
    with torch.no_grad(): numeric=(loss(source+direction*.001)-loss(source-direction*.001))/.002
    analytic=(source.grad*direction).sum()
    assert analytic.abs()>0
    torch.testing.assert_close(numeric,analytic,rtol=.005,atol=1e-7)


def test_noisy_strided_groups_preserve_full_notebook_epoch_updates():
    f=inversion();data=make_tiny_data(nt=64,shots=4)
    config=ExperimentConfig(noise=True,sigma_init=.08,batches=2,critic_cycles=2)
    observed=to_critic_layout(legacy_normalize(data.traces))
    torch.manual_seed(48);reference=make_reference(data,config)
    torch.manual_seed(48);actual=f.FWIGANStrategy(parameters(data,config),LegacyDeepwave(),data,config)
    torch.manual_seed(27)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='User provided device_type.*CUDA is not available.*')
        gl,cl=reference[-1](observed,all_batches=True)
    ref_rng=torch.get_rng_state()
    torch.manual_seed(27)
    metrics=[actual.step(slice(b,None,config.batches)) for b in range(config.batches)]
    actual.finish_epoch()
    for m,g,c in zip(metrics,gl,cl):
        assert m['generator_loss']==g
        assert m['critic_loss']==pytest.approx(c,abs=2e-6)
    assert torch.equal(torch.get_rng_state(),ref_rng)
    torch.testing.assert_close(actual.parameters.velocity,reference[0].model,rtol=0,atol=0)
    torch.testing.assert_close(actual.parameters.source,reference[0].source.permute(1,2,0),rtol=0,atol=0)
    torch.testing.assert_close(actual.parameters.sigma,reference[0].sigma,rtol=0,atol=0)
    for a,b in zip(actual.critic.parameters(),reference[1].parameters()): torch.testing.assert_close(a,b,rtol=0,atol=0)
    adam_equal(actual.physical_optimizer,reference[2]);adam_equal(actual.critic_optimizer,reference[3])


def test_shot_specific_known_source_partial_batches_match_full_fwi():
    f=inversion();data=make_tiny_data(nt=64,shots=3,receivers=8)
    data.source=data.source.repeat(3,1,1)*torch.tensor([.9,1.,1.1]).reshape(3,1,1)
    data.traces=LegacyDeepwave().predict(data.reference_velocity,data.source,data.acquisition,data.grid,data.sampling)
    strategies=[]
    for batch in (3,2):
        config=ExperimentConfig(method='fwi',batch_size=batch)
        s=f.FWIStrategy(parameters(data,config),LegacyDeepwave(),data,config);s.step_epoch();strategies.append(s)
    torch.testing.assert_close(strategies[0].parameters.velocity.grad,strategies[1].parameters.velocity.grad,rtol=2e-5,atol=1e-8)
    for s in strategies: torch.testing.assert_close(s.parameters.source,data.source,rtol=0,atol=0)


def test_critic_flags_restore_after_generator_exception():
    f=inversion();data=make_tiny_data(nt=64);config=ExperimentConfig(batches=1,critic_cycles=1)
    strategy=f.FWIGANStrategy(parameters(data,config),LegacyDeepwave(),data,config)
    params=list(strategy.critic.parameters());params[0].requires_grad_(False)
    original_flags=[p.requires_grad for p in params]
    calls=[]
    class GeneratorFailure(Exception): pass
    def hook(*_):
        calls.append(1)
        if len(calls)==4:
            assert not any(p.requires_grad for p in params)
            assert all(p.grad is None for p in params)
            raise GeneratorFailure
    handle=strategy.critic.register_forward_pre_hook(hook)
    try:
        with pytest.raises(GeneratorFailure): strategy.step([0,1])
    finally: handle.remove()
    assert [p.requires_grad for p in params]==original_flags


def test_invalid_observed_tensor_type_raises_clear_value_error():
    f=inversion();data=make_tiny_data(nt=64);config=ExperimentConfig(batches=1)
    with pytest.raises(ValueError,match='float32 tensor'):
        f.FWIGANStrategy(parameters(data,config),LegacyDeepwave(),data,config,observed=[1,2])


def test_controlled_initial_velocity_prior_improves_datafit_and_model_error():
    f=inversion();data=make_tiny_data(nt=64,shots=2,receivers=8)
    # The prior uses the supplied initial model only, not reference truth.
    config=ExperimentConfig(method='fwi',velocity_lr=2.,velocity_min=data.initial_velocity.min().item())
    strategy=f.FWIStrategy(parameters(data,config),LegacyDeepwave(),data,config)
    def data_loss(v):
        return (LegacyDeepwave().predict(v,data.source,data.acquisition,data.grid,data.sampling)-data.traces).square().mean()
    def model_error(v): return (v-data.reference_velocity).square().mean().sqrt()
    initial_loss=data_loss(data.initial_velocity).item()
    initial_error=model_error(data.initial_velocity).item()
    for _ in range(6): strategy.step_epoch()
    assert data_loss(strategy.parameters.velocity).item()<initial_loss*.9
    assert model_error(strategy.parameters.velocity).item()<initial_error
