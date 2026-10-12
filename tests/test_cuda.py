"""Optional real CUDA checks. CPU skips do not establish GPU correctness.

Float32 propagation may differ by reduction order: traces use 3e-4 relative /
3e-3 absolute pressure tolerance; gradients use 2e-3 relative / 2e-4 absolute.
Restart uses the same GPU and RNG, with 1e-5 float32 tensor tolerance. Default
GradScaler(65536) is allowed to back off naturally; tests never lower it first.
"""
import copy
import pytest
import torch
from fwigan import ExperimentConfig
from fwigan.data import make_tiny_data
from fwigan.experiment import Experiment
from fwigan.physics import LegacyDeepwave
from fwigan.checkpoints import restore_rng, rng_snapshot

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable; optional integration')


def test_raw_cpu_cuda_traces_and_velocity_source_gradients():
    data=make_tiny_data(nt=64,shots=2,receivers=8)
    outputs=[]
    for device in ('cpu','cuda'):
        value=data.to(device)
        velocity=value.initial_velocity.clone().requires_grad_()
        source=value.source.clone().requires_grad_()
        prediction=LegacyDeepwave().predict(velocity,source,value.acquisition,value.grid,value.sampling)
        prediction.square().mean().backward()
        assert torch.isfinite(velocity.grad).all() and velocity.grad.abs().sum()>0
        assert torch.isfinite(source.grad).all() and source.grad.abs().sum()>0
        outputs.append(tuple(t.detach().cpu() for t in (prediction,velocity.grad,source.grad)))
    torch.testing.assert_close(outputs[0][0],outputs[1][0],rtol=3e-4,atol=3e-3)
    for cpu,cuda in zip(outputs[0][1:],outputs[1][1:]):
        torch.testing.assert_close(cpu,cuda,rtol=2e-3,atol=2e-4)


def assert_cpu(value):
    if torch.is_tensor(value):assert value.device.type=='cpu'
    elif isinstance(value,dict):
        for item in value.values():assert_cpu(item)
    elif isinstance(value,(list,tuple)):
        for item in value:assert_cpu(item)


@pytest.mark.parametrize('noise',[False,True])
def test_default_amp_backoff_actual_steps_portable_checkpoint_and_interrupted_resume(tmp_path,noise):
    data=make_tiny_data(nt=64,shots=4)
    cfg=ExperimentConfig(device='cuda',amp=True,noise=noise,sigma_init=.08,
        epochs=32,batches=2,critic_cycles=2,evaluation_interval=32,checkpoint_interval=32)
    run=Experiment.create(cfg,data,tmp_path)
    assert run.strategy.physical_scaler.get_scale()==65536.
    assert run.strategy.critic_scaler.get_scale()==65536.
    stable=0
    initial_physical={key:value.detach().cpu().clone() for key,value in run.parameters.state_dict().items()}
    initial_critic={key:value.detach().cpu().clone() for key,value in run.strategy.critic.state_dict().items()}
    for _ in range(24):
        before=run.strategy.counters.copy()
        scales=(run.strategy.physical_scaler.get_scale(),run.strategy.critic_scaler.get_scale())
        run.run(max_batches=1)
        after=run.strategy.counters
        unchanged=scales==(run.strategy.physical_scaler.get_scale(),run.strategy.critic_scaler.get_scale())
        successful=after['physical_updates']-before['physical_updates']==1 and after['critic_updates']-before['critic_updates']==2
        stable=stable+1 if successful and unchanged else 0
        if stable>=2:break
    assert stable>=2, 'default AMP did not reach stable real optimizer updates in 24 steps'
    assert run.strategy.counters['physical_skipped']+run.strategy.counters['critic_skipped']>0
    assert min(run.strategy.physical_scaler.get_scale(),run.strategy.critic_scaler.get_scale())<65536.
    enabled=('velocity','source','sigma') if noise else ('velocity','source')
    for name in enabled:
        parameter=getattr(run.parameters,name)
        assert torch.isfinite(parameter).all() and not torch.equal(parameter.cpu(),initial_physical[name])
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum()>0
    assert any(not torch.equal(value.cpu(),initial_critic[key]) for key,value in run.strategy.critic.state_dict().items())
    # Force the next step across the epoch boundary so scheduler restoration is exercised.
    if run.position.batch==0:run.run(max_batches=1)
    assert run.position.batch==1
    saved=run.save_checkpoint(tmp_path/'interrupted.pt')
    payload=torch.load(saved,map_location='cpu');assert_cpu(payload)
    cpu=Experiment.resume(saved,device='cpu')
    assert cpu.parameters.velocity.device.type=='cpu' and not cpu.strategy.amp_enabled
    assert cpu.strategy.physical_scaler.state_dict()==payload['strategy']['physical_scaler']
    # Restore snapshot RNG before the baseline next step after the CPU portability probe.
    restore_rng(payload['rng'])
    run.run(max_batches=1)
    expected=copy.deepcopy({key:value.detach().cpu() for key,value in run.parameters.state_dict().items()})
    expected_critic=copy.deepcopy({key:value.detach().cpu() for key,value in run.strategy.critic.state_dict().items()})
    expected_counters=run.strategy.counters.copy()
    expected_strategy=run.strategy.state_dict()
    expected_rng=rng_snapshot()
    restored=Experiment.resume(saved,device='cuda'); restored.run(max_batches=1)
    for key,value in restored.parameters.state_dict().items():
        torch.testing.assert_close(value.cpu(),expected[key],rtol=1e-5,atol=1e-5)
    for key,value in restored.strategy.critic.state_dict().items():
        torch.testing.assert_close(value.cpu(),expected_critic[key],rtol=1e-5,atol=1e-5)
    assert restored.strategy.counters==expected_counters
    assert restored.position==run.position
    restored_strategy=restored.strategy.state_dict()
    def same(left,right):
        if torch.is_tensor(left):torch.testing.assert_close(left,right,rtol=1e-5,atol=1e-5)
        elif isinstance(left,dict):
            assert left.keys()==right.keys()
            for key in left:same(left[key],right[key])
        elif isinstance(left,(list,tuple)):
            assert len(left)==len(right)
            for x,y in zip(left,right):same(x,y)
        else:assert left==right
    for name in ('physical_optimizer','critic_optimizer','physical_scheduler','critic_scheduler','physical_scaler','critic_scaler'):
        same(restored_strategy[name],expected_strategy[name])
    actual_rng=rng_snapshot()
    assert torch.equal(actual_rng['torch'],expected_rng['torch'])
    assert all(torch.equal(a,b) for a,b in zip(actual_rng['cuda'],expected_rng['cuda']))
    assert actual_rng['python']==expected_rng['python']
    import numpy as np
    for a,b in zip(actual_rng['numpy'],expected_rng['numpy']):np.testing.assert_array_equal(a,b)


def test_cuda_fwi_improves_raw_data_fit_and_keeps_fixed_source(tmp_path):
    data=make_tiny_data(nt=128,receivers=8)
    cfg=ExperimentConfig(method='fwi',device='cuda',amp=False,epochs=8,batch_size=2,
        velocity_min=2000.,velocity_max=2300.,evaluation_interval=8)
    run=Experiment.create(cfg,data,tmp_path)
    initial=run.evaluate()['raw_residual_mse'];run.run()
    assert run.evaluations[-1]['raw_residual_mse']<initial
    assert run.strategy.counters['physical_updates']==8
    torch.testing.assert_close(run.parameters.source.cpu(),data.source,rtol=0,atol=0)
    assert_cpu(torch.load(run.run_dir/'final.pt',map_location='cpu'))
