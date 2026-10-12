"""Real-solver lifecycle checks: a restart must preserve the next update."""
import copy
import importlib.util
import json
import random
from pathlib import Path
import numpy as np
import pytest
import torch
from fwigan import ExperimentConfig
from fwigan.data import make_tiny_data, legacy_normalize, to_critic_layout, save_observations, load_observations


def experiment():
    assert importlib.util.find_spec('fwigan.experiment') is not None, 'experiment lifecycle is missing'
    from fwigan.experiment import Experiment
    return Experiment


def same(a,b):
    if torch.is_tensor(a): torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,np.ndarray): np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: same(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b): same(x,y)
    else: assert a==b


def config(**kwargs):
    return ExperimentConfig(device='cpu',epochs=2,batches=2,critic_cycles=1,
        physical_scheduler_step_size=1,critic_scheduler_step_size=1,**kwargs)


@pytest.mark.parametrize('noise',[False,True])
def test_partial_epoch_restart_preserves_exact_next_update_and_rng(tmp_path,noise):
    E=experiment(); data=make_tiny_data(nt=32,shots=4)
    cfg=config(noise=noise,sigma_init=.08)
    full=E.create(cfg,data,tmp_path);full.run()
    expected=(copy.deepcopy(full.parameters.state_dict()),full.strategy.state_dict(),torch.get_rng_state(),
              random.getstate(),np.random.get_state())
    split=E.create(cfg,data,tmp_path); split.run(max_batches=1)
    assert (split.position.epoch,split.position.batch,split.position.global_step)==(0,1,1)
    saved=split.save_checkpoint()
    restored=E.resume(saved,device='cpu')
    same(restored.strategy.observed,split.strategy.observed)
    restored.run()
    same((restored.parameters.state_dict(),restored.strategy.state_dict(),torch.get_rng_state(),
          random.getstate(),np.random.get_state()),expected)
    assert restored.run_dir==split.run_dir and len(restored.metrics)==4
    assert restored.position.epoch==2 and restored.position.batch==0


def test_resume_named_lrs_signed_sigma_scalers_and_exact_target(tmp_path):
    E=experiment(); cfg=config(noise=True);data=make_tiny_data(nt=32,shots=4)
    run=E.create(cfg,data,tmp_path);run.run(max_batches=1)
    for i,g in enumerate(run.strategy.physical_optimizer.param_groups):g['lr']=.0123*(i+1)
    with torch.no_grad():run.parameters.sigma.fill_(-.04)
    run.strategy.physical_scaler.load_state_dict(dict(scale=512.,growth_factor=2.,backoff_factor=.5,growth_interval=100,_growth_tracker=7))
    checkpoint=run.save_checkpoint()
    loaded=torch.load(checkpoint,map_location='cpu')
    assert set(loaded['physical'])=={'velocity','source','sigma'}
    assert all(v.device.type=='cpu' for v in loaded['physical'].values())
    restored=E.resume(checkpoint)
    same(restored.parameters.state_dict(),run.parameters.state_dict())
    same(restored.strategy.state_dict(),run.strategy.state_dict())


def test_fwi_restart_without_truth_keeps_known_source(tmp_path):
    E=experiment();data=make_tiny_data(nt=64,shots=3,receivers=8)
    data.reference_velocity=None;data.initial_source=data.source*.3
    path=tmp_path/'external.pt';save_observations(path,data);data=load_observations(path)
    cfg=config(method='fwi',batch_size=2)
    full=E.create(cfg,data,tmp_path);full.run()
    expected=copy.deepcopy(full.parameters.state_dict());state=full.strategy.state_dict()
    split=E.create(cfg,data,tmp_path);split.run(epochs=1,max_batches=1)
    assert split.position.epoch==1 and split.position.batch==0
    resumed=E.resume(split.save_checkpoint());resumed.run()
    same(resumed.parameters.state_dict(),expected);same(resumed.strategy.state_dict(),state)
    same(resumed.parameters.source,data.source)
    assert 'model_rmse' not in resumed.evaluate()
    exported=torch.load(resumed.run_dir/'final.pt',map_location='cpu')
    assert exported['predicted'].shape==data.traces.shape
    same(exported['source'],data.source)


def test_noise_is_seeded_once_before_critic_and_clean_draw_is_preserved(tmp_path):
    E=experiment();data=make_tiny_data(nt=32,shots=4);data.initial_source=data.source*.7
    cfg=config(noise=True,observation_noise_level=.12)
    torch.manual_seed(cfg.seed)
    clean=to_critic_layout(legacy_normalize(data.traces))
    target=(clean+.12*torch.randn_like(clean))[0].permute(0,2,1)
    from fwigan.inversion import LegacyCritic
    expected=LegacyCritic(2,32,60).state_dict()
    noisy=E.create(cfg,data,tmp_path)
    same(noisy.strategy.observed,target);same(noisy.strategy.critic.state_dict(),expected)
    same(noisy.parameters.source,data.initial_source)
    same(noisy.data.traces,data.traces)
    clean_run=E.create(config(noise=False,observation_noise_level=.12),data,tmp_path)
    same(clean_run.strategy.observed,legacy_normalize(data.traces))
    same(clean_run.strategy.critic.state_dict(),expected)
    zero=E.create(config(noise=True,observation_noise_level=0.),data,tmp_path)
    same(zero.strategy.observed,legacy_normalize(data.traces))


def test_warm_start_uses_physical_values_with_fresh_history_and_new_run(tmp_path):
    E=experiment();data=make_tiny_data(nt=32,shots=4)
    original=E.create(config(noise=True),data,tmp_path);original.run(max_batches=1)
    warm=E.warm_start(original.save_checkpoint(),config(noise=True),data,tmp_path)
    assert warm.run_dir!=original.run_dir
    same(warm.parameters.state_dict(),original.parameters.state_dict())
    assert not warm.strategy.physical_optimizer.state and not warm.strategy.critic_optimizer.state
    assert warm.position.global_step==0 and warm.metrics==[]
    fwi_data=make_tiny_data(nt=64,receivers=8)
    with pytest.raises(ValueError,match='shape|geometry'):E.warm_start(original.save_checkpoint(),config(method='fwi'),fwi_data,tmp_path)


@pytest.mark.parametrize('field,value',[('observation_noise_level',float('nan')),('observation_noise_level',-.1),('checkpoint_interval',0),('evaluation_interval',1.5)])
def test_new_numerical_config_rejects_invalid_fields(field,value):
    with pytest.raises(ValueError,match=field): ExperimentConfig(**{field:value})


@pytest.mark.parametrize('corruption',['schema','data','config','position','physical','observed','metrics','rng'])
def test_corrupt_snapshots_are_rejected(tmp_path,corruption):
    E=experiment();run=E.create(config(noise=True),make_tiny_data(nt=32,shots=4),tmp_path)
    run.run(max_batches=1);payload=torch.load(run.save_checkpoint(),map_location='cpu')
    if corruption=='schema':payload['schema']=999
    elif corruption=='data':payload['dataset']['traces'][0,0,0]+=1
    elif corruption=='config':payload['config']['velocity_lr']+=1
    elif corruption=='position':payload['position']['batch']=2
    elif corruption=='physical':payload['physical']['velocity']=torch.ones(2)
    elif corruption=='observed':payload['strategy']['observed'][0,0,0]=float('nan')
    elif corruption=='metrics':payload['metrics'][0]['generator_loss']=float('nan')
    else:payload['rng']['torch']=torch.ones(4)
    path=tmp_path/'corrupt.pt';torch.save(payload,path)
    with pytest.raises(ValueError):E.resume(path)


@pytest.mark.parametrize('artifact',['observations.pt','config.json'])
def test_changed_immutable_run_artifacts_are_rejected(tmp_path,artifact):
    E=experiment();run=E.create(config(),make_tiny_data(nt=32,shots=4),tmp_path)
    path=run.save_checkpoint()
    target=run.run_dir/artifact
    if artifact.endswith('.pt'):
        payload=torch.load(target,map_location='cpu');payload['traces'][0,0,0]+=1;torch.save(payload,target)
    else:
        payload=json.loads(target.read_text());payload['seed']+=1;target.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='fingerprint|config'):E.resume(path)


def test_atomic_failure_preserves_previous_checkpoint_and_rejects_other_run(tmp_path,monkeypatch):
    E=experiment();data=make_tiny_data(nt=32,shots=4)
    run=E.create(config(),data,tmp_path);path=run.save_checkpoint();before=path.read_bytes()
    other=E.create(config(),data,tmp_path)
    with pytest.raises(FileExistsError):other.save_checkpoint(path)
    import fwigan.checkpoints as c
    def fail(*args,**kwargs):raise OSError('simulated full disk')
    monkeypatch.setattr(c.os,'replace',fail)
    with pytest.raises(OSError,match='full disk'):run.save_checkpoint(path)
    assert path.read_bytes()==before and not list(run.run_dir.glob('.*.tmp'))


def test_cadences_count_equivalent_solver_work_without_per_batch_evaluation(tmp_path):
    E=experiment();data=make_tiny_data(nt=32,shots=4)
    run=E.create(config(evaluation_interval=2,checkpoint_interval=2),data,tmp_path)
    run.run(max_batches=1)
    assert run.solver_counts.forward==1 and run.solver_counts.adjoint==1 and run.evaluations==[]
    state=torch.get_rng_state().clone();python=random.getstate();numpy=np.random.get_state()
    metrics=run.evaluate();same(torch.get_rng_state(),state);same(random.getstate(),python);same(np.random.get_state(),numpy)
    assert metrics['raw_residual_mse']>=0 and run.solver_counts.forward==2
    run.run()
    assert run.solver_counts.forward==6 and run.solver_counts.adjoint==4
    assert len(run.evaluations)==2
    assert (run.run_dir/'checkpoint.pt').exists() and (run.run_dir/'final.pt').exists()


def test_resume_prefers_current_validated_directory_even_if_original_exists(tmp_path):
    import shutil
    E=experiment();data=make_tiny_data(nt=32,shots=4)
    run=E.create(config(noise=True),data,tmp_path);run.run(max_batches=1)
    checkpoint=run.save_checkpoint();copied=tmp_path/'copied-run';shutil.copytree(run.run_dir,copied)
    run.run();expected=copy.deepcopy(run.parameters.state_dict());state=run.strategy.state_dict()
    resumed=E.resume(copied/checkpoint.name)
    assert resumed.run_dir==copied
    resumed.run();same(resumed.parameters.state_dict(),expected);same(resumed.strategy.state_dict(),state)
    assert torch.load(copied/'checkpoint.pt',map_location='cpu')['run_dir']==str(copied)
    shutil.rmtree(run.run_dir)
    moved=tmp_path/'moved-run';shutil.move(str(copied),str(moved))
    assert E.resume(moved/'checkpoint.pt').run_dir==moved


@pytest.mark.parametrize('corruption',['last_epoch','period','gamma','step_count','solver_history','evaluation_position'])
def test_inconsistent_epoch_scheduler_and_history_rejected(tmp_path,corruption):
    E=experiment();run=E.create(config(),make_tiny_data(nt=32,shots=4),tmp_path)
    run.run(epochs=1);run.evaluate();payload=torch.load(run.save_checkpoint(),map_location='cpu')
    state=payload['strategy']['physical_scheduler']
    if corruption=='last_epoch':state['last_epoch']+=1
    elif corruption=='period':state['step_size']+=1
    elif corruption=='gamma':state['gamma']=.1
    elif corruption=='step_count':state['_step_count']+=1
    elif corruption=='solver_history':payload['metrics'][0]['solver_forward']=-1
    else:payload['evaluations'][0]['epoch']=-1
    path=tmp_path/'corrupt-epoch.pt';torch.save(payload,path)
    with pytest.raises(ValueError):E.resume(path)


def test_warm_start_fwi_keeps_new_known_source_and_resets_schedule(tmp_path):
    E=experiment();data=make_tiny_data(nt=64,shots=3,receivers=8)
    old=E.create(config(method='fwi',batch_size=2),data,tmp_path);old.run(epochs=1)
    new_data=copy.deepcopy(data);new_data.source*=.8
    warm=E.warm_start(old.save_checkpoint(),config(method='fwi'),new_data,tmp_path)
    same(warm.parameters.velocity,old.parameters.velocity);same(warm.parameters.source,new_data.source)
    assert warm.strategy.physical_scheduler.last_epoch==0 and warm.position.epoch==0


def test_owned_destination_cannot_publish_new_name_in_another_run(tmp_path):
    E=experiment();data=make_tiny_data(nt=32,shots=4)
    a=E.create(config(),data,tmp_path);b=E.create(config(),data,tmp_path)
    with pytest.raises(FileExistsError):a.save_checkpoint(b.run_dir/'new-checkpoint.pt')
    assert not (b.run_dir/'new-checkpoint.pt').exists()


def test_invalid_resume_preserves_all_process_rng(tmp_path):
    E=experiment();run=E.create(config(),make_tiny_data(nt=32,shots=4),tmp_path)
    payload=torch.load(run.save_checkpoint(),map_location='cpu');payload['physical']['velocity'][0,0]=float('nan')
    path=tmp_path/'bad-physical.pt';torch.save(payload,path)
    before=(torch.get_rng_state(),random.getstate(),np.random.get_state())
    with pytest.raises(ValueError):E.resume(path)
    same((torch.get_rng_state(),random.getstate(),np.random.get_state()),before)


def test_invalid_portable_cuda_rng_history_rejects_before_process_rng_change(tmp_path):
    E=experiment();run=E.create(config(),make_tiny_data(nt=32,shots=4),tmp_path)
    path=run.save_checkpoint();payload=torch.load(path,map_location='cpu')
    payload['rng']['cuda']=[torch.ones(3,dtype=torch.uint8)]
    bad=tmp_path/'bad-cuda-rng.pt';torch.save(payload,bad)
    before=(torch.get_rng_state(),random.getstate(),np.random.get_state())
    with pytest.raises(ValueError,match='CUDA RNG'):E.resume(bad)
    same((torch.get_rng_state(),random.getstate(),np.random.get_state()),before)


def test_unknown_snapshot_and_overwrite_are_clear_errors(tmp_path):
    E=experiment();run=E.create(config(),make_tiny_data(nt=32,shots=4),tmp_path)
    path=tmp_path/'unknown.pt';path.write_bytes(b'not a torch pickle')
    with pytest.raises(ValueError,match='snapshot'):E.resume(path)
    with pytest.raises(FileExistsError,match='unknown'):run.save_checkpoint(path)
    assert path.read_bytes()==b'not a torch pickle'


def test_nonfinite_physical_state_cannot_replace_valid_checkpoint(tmp_path):
    E=experiment();run=E.create(config(),make_tiny_data(nt=32,shots=4),tmp_path)
    path=run.save_checkpoint();before=path.read_bytes()
    with torch.no_grad():run.parameters.source[0,0,0]=float('nan')
    with pytest.raises(ValueError,match='source'):run.save_checkpoint()
    assert path.read_bytes()==before


def test_result_contains_global_counters_and_raw_physical_prediction(tmp_path):
    from fwigan.physics import LegacyDeepwave
    E=experiment();data=make_tiny_data(nt=32,shots=4)
    run=E.create(config(noise=True),data,tmp_path);run.run()
    result=torch.load(run.run_dir/'final.pt',map_location='cpu')
    assert result['counters']==run.strategy.counters
    assert result['solver_counts']=={'forward':run.solver_counts.forward,'adjoint':run.solver_counts.adjoint}
    assert result['elapsed_seconds']==run.elapsed_seconds
    with torch.no_grad():expected=LegacyDeepwave().predict(run.parameters.velocity,run.parameters.source,data.acquisition,data.grid,data.sampling)
    same(result['predicted'],expected)


@pytest.mark.parametrize('size',[808,816])
def test_cuda_rng_history_is_portable_on_cpu_with_pinned_state_sizes(tmp_path,size):
    E=experiment();run=E.create(config(),make_tiny_data(nt=32,shots=4),tmp_path)
    payload=torch.load(run.save_checkpoint(),map_location='cpu')
    payload['rng']['cuda']=[torch.zeros(size,dtype=torch.uint8)]
    path=tmp_path/'portable-cuda-history.pt';torch.save(payload,path)
    loaded=E.resume(path,device='cpu')
    same(loaded.parameters.state_dict(),run.parameters.state_dict())


def test_invalid_philox_offset_rejects_on_cpu(tmp_path):
    E=experiment();run=E.create(config(),make_tiny_data(nt=32,shots=4),tmp_path)
    payload=torch.load(run.save_checkpoint(),map_location='cpu')
    rng=torch.zeros(816,dtype=torch.uint8);rng[808]=1;payload['rng']['cuda']=[rng]
    path=tmp_path/'bad-offset.pt';torch.save(payload,path)
    with pytest.raises(ValueError,match='offset'):E.resume(path,device='cpu')


def test_fwi_epoch_failure_does_not_save_partial_accumulation(tmp_path):
    from fwigan.physics import LegacyDeepwave
    E=experiment();data=make_tiny_data(nt=64,shots=3,receivers=8)
    class InterruptedSolve:
        def __init__(self):self.calls=0
        def predict(self,*args):
            self.calls+=1
            if self.calls==2:raise RuntimeError('interrupted shot solve')
            return LegacyDeepwave().predict(*args)
    run=E.create(config(method='fwi',batch_size=2),data,tmp_path,forward=InterruptedSolve())
    checkpoint=run.save_checkpoint();before=copy.deepcopy(run.parameters.state_dict())
    with pytest.raises(RuntimeError,match='interrupted'):run.run(max_batches=1)
    same(run.parameters.state_dict(),before)
    assert not run.strategy.physical_optimizer.state
    assert run.position.epoch==0 and run.position.global_step==0 and run.metrics==[]
    restored=E.resume(checkpoint);restored.run(max_batches=1)
    assert restored.position.epoch==1 and restored.position.batch==0


def test_cpu_resave_preserves_absent_cuda_rng_history(tmp_path,monkeypatch):
    # Opaque CUDA history is retained for devices absent from this runtime.
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    E=experiment();run=E.create(config(),make_tiny_data(nt=32,shots=4),tmp_path)
    payload=torch.load(run.save_checkpoint(),map_location='cpu')
    history=torch.zeros(816,dtype=torch.uint8);history[800]=27;history[808]=4
    payload['rng']['cuda']=[history]
    checkpoint=tmp_path/'from-cuda.pt';torch.save(payload,checkpoint)
    loaded=E.resume(checkpoint,device='cpu')
    roundtrip=torch.load(loaded.save_checkpoint(),map_location='cpu')
    same(roundtrip['rng']['cuda'],[history])


def test_fwi_fixed_sigma_checkpoint_uses_float32_initializer_value(tmp_path):
    E=experiment();data=make_tiny_data(nt=64,receivers=8)
    run=E.create(config(method='fwi',sigma_init=.08),data,tmp_path)
    restored=E.resume(run.save_checkpoint())
    same(restored.parameters.sigma,torch.tensor(.08,dtype=torch.float32))


def test_warm_start_validates_cuda_snapshot_on_cpu_without_old_runtime(tmp_path):
    from fwigan.checkpoints import fingerprint_config
    E=experiment();data=make_tiny_data(nt=32,shots=4)
    old=E.create(config(noise=True),data,tmp_path);old.run(max_batches=1)
    payload=torch.load(old.save_checkpoint(),map_location='cpu')
    payload['config']['device']='cuda';payload['config_fingerprint']=fingerprint_config(payload['config'])
    snapshot=tmp_path/'cuda-origin.pt';torch.save(payload,snapshot)
    new=E.warm_start(snapshot,config(noise=True),data,tmp_path)
    same(new.parameters.state_dict(),old.parameters.state_dict())
    assert new.parameters.velocity.device.type=='cpu' and new.metrics==[]


@pytest.mark.parametrize('method',['fwigan','fwi'])
@pytest.mark.parametrize('corruption',['zero','descending','evaluation_descending','missing_evaluation_work','under_counted_training'])
def test_solver_accounting_rejects_nonnegative_impossible_histories(tmp_path,method,corruption):
    E=experiment();data=make_tiny_data(nt=32,shots=4,receivers=8 if method=='fwi' else 60)
    run=E.create(config(method=method,batch_size=3),data,tmp_path)
    run.evaluate();run.run();run.evaluate()
    payload=torch.load(run.save_checkpoint(),map_location='cpu')
    if corruption=='zero':
        payload['solver_counts']={'forward':0,'adjoint':0}
        for row in payload['metrics']+payload['evaluations']:row['solver_forward']=row['solver_adjoint']=0
    elif corruption=='descending':
        payload['metrics'][1]['solver_forward']=payload['metrics'][1]['solver_adjoint']=0
    elif corruption=='evaluation_descending':
        payload['evaluations'][-1]['solver_forward']=payload['evaluations'][-1]['solver_adjoint']=0
    elif corruption=='missing_evaluation_work':
        for row in payload['metrics']+payload['evaluations']:row['solver_forward']-=1
        payload['solver_counts']['forward']-=1
    else:
        payload['metrics'][0]['solver_adjoint']=0 if method=='fwigan' else 1
    path=tmp_path/'invalid-accounting.pt';torch.save(payload,path)
    with pytest.raises(ValueError,match='solver'):E.resume(path)


def test_failed_fwi_solves_allow_larger_cumulative_accounting(tmp_path):
    from fwigan.physics import LegacyDeepwave
    E=experiment();data=make_tiny_data(nt=64,shots=3,receivers=8)
    class FailOnce:
        def __init__(self):self.calls=0
        def predict(self,*args):
            self.calls+=1
            if self.calls==2:raise RuntimeError('one failed solve')
            return LegacyDeepwave().predict(*args)
    run=E.create(config(method='fwi',batch_size=2),data,tmp_path,forward=FailOnce())
    with pytest.raises(RuntimeError,match='failed solve'):run.run(max_batches=1)
    run.evaluate();run.run(max_batches=1)
    assert run.solver_counts.forward==5 and run.solver_counts.adjoint==3
    restored=E.resume(run.save_checkpoint())
    assert restored.solver_counts==run.solver_counts
    same(restored.parameters.state_dict(),run.parameters.state_dict())


@pytest.mark.parametrize('field,value',[('observation_noise_level',.06),('epochs',3),('device','auto')])
def test_strategy_config_divergence_cannot_replace_restart_point(tmp_path,field,value):
    E=experiment();run=E.create(config(),make_tiny_data(nt=32,shots=4),tmp_path)
    run.run(max_batches=1);checkpoint=run.save_checkpoint();before=checkpoint.read_bytes()
    setattr(run.strategy.config,field,value)
    with pytest.raises(ValueError,match='strategy.*config'):run.save_checkpoint()
    assert checkpoint.read_bytes()==before
    assert E.resume(checkpoint).position.global_step==1


def test_corrupt_training_signature_rejected_by_snapshot_boundary(tmp_path):
    E=experiment();run=E.create(config(),make_tiny_data(nt=32,shots=4),tmp_path)
    payload=torch.load(run.save_checkpoint(),map_location='cpu')
    payload['strategy']['signature']['config']['observation_noise_level']+=.01
    with pytest.raises(ValueError,match='strategy.*config'):E._validate_structure(payload)


def test_resumed_runtime_device_override_keeps_expected_strategy_config(tmp_path):
    from fwigan.checkpoints import fingerprint_config
    E=experiment();run=E.create(config(),make_tiny_data(nt=32,shots=4),tmp_path)
    checkpoint=run.save_checkpoint();payload=torch.load(checkpoint,map_location='cpu')
    payload['config']['device']='cuda';payload['config_fingerprint']=fingerprint_config(payload['config'])
    (run.run_dir/'config.json').write_text(json.dumps(payload['config']))
    manifest=json.loads((run.run_dir/'run.json').read_text())
    manifest['config_fingerprint']=payload['config_fingerprint'];(run.run_dir/'run.json').write_text(json.dumps(manifest))
    torch.save(payload,checkpoint)
    restored=E.resume(checkpoint,device='cpu');saved=torch.load(restored.save_checkpoint(),map_location='cpu')
    assert saved['config']['device']=='cuda'
    assert restored.config.device==restored.strategy.config.device=='cpu'


def test_elapsed_aggregate_cannot_undercount_successful_step_times(tmp_path):
    E=experiment();run=E.create(config(),make_tiny_data(nt=32,shots=4),tmp_path)
    run.run();payload=torch.load(run.save_checkpoint(),map_location='cpu')
    assert sum(row['elapsed_seconds'] for row in payload['metrics'])>0
    payload['elapsed_seconds']=0.
    with pytest.raises(ValueError,match='elapsed'):E._validate_structure(payload)


def test_elapsed_aggregate_allows_roundoff_and_additional_failed_work(tmp_path):
    import math
    E=experiment();run=E.create(config(),make_tiny_data(nt=32,shots=4),tmp_path)
    run.run();payload=torch.load(run.save_checkpoint(),map_location='cpu')
    total=math.fsum(row['elapsed_seconds'] for row in payload['metrics'])
    for elapsed in (total-1e-13,total+1.):
        payload['elapsed_seconds']=elapsed;E._validate_structure(payload)
