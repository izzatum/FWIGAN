"""Bounded real-solver command line workflows and public notebook clients."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

from fwigan.data import load_observations, save_observations, make_tiny_data


def cli(*args):
    return subprocess.run([sys.executable, '-m', 'fwigan', *map(str, args)],
                          text=True, capture_output=True, env=os.environ.copy())


def write_config(path, method='fwi', noise=False, dataset=None):
    value = {'experiment': {'method': method, 'device': 'cpu', 'epochs': 2,
             'batches': 2, 'batch_size': 2, 'critic_cycles': 1, 'noise': noise,
             'sigma_init': .08, 'evaluation_interval': 1},
             'dataset': dataset or {'kind': 'tiny', 'nt': 32, 'shots': 4, 'receivers': 60}}
    path.write_text(yaml.safe_dump(value))
    return path


@pytest.mark.parametrize('method,noise', [('fwi',False), ('fwigan',False), ('fwigan',True)])
def test_simulation_inversion_interruption_resume_evaluate(tmp_path, method, noise):
    cfg = write_config(tmp_path/'config.yaml', method, noise)
    observations = tmp_path/'observations.pt'
    result = cli('simulate', '--config', cfg, '--output', observations)
    assert result.returncode == 0, result.stderr
    data = load_observations(observations)
    assert data.traces.device.type == 'cpu' and data.traces.shape == (4,60,32)
    cfg = write_config(cfg, method, noise, {'kind':'observations','path':'observations.pt'})
    result = cli('invert', '--config', cfg, '--output-root', tmp_path/'runs', '--max-batches', 1)
    assert result.returncode == 0, result.stderr
    run_dir = Path(json.loads(result.stdout)['run_dir'])
    checkpoint = torch.load(run_dir/'checkpoint.pt', map_location='cpu')
    assert checkpoint['position']['global_step'] == 1
    assert all(v.device.type == 'cpu' for v in checkpoint['physical'].values())
    result = cli('resume', run_dir/'checkpoint.pt', '--device','cpu')
    assert result.returncode == 0, result.stderr
    final = torch.load(run_dir/'final.pt', map_location='cpu')
    assert final['position'] == {'epoch':2,'batch':0,'global_step':2 if method=='fwi' else 4}
    assert final['predicted'].shape == data.traces.shape
    assert final['counters']['physical_updates'] == final['position']['global_step']
    result = cli('evaluate', run_dir)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['metrics']['raw_residual_mse'] >= 0


def test_external_observations_invert_without_truth(tmp_path):
    data = make_tiny_data(nt=32, receivers=8); data.reference_velocity = None
    save_observations(tmp_path/'external.pt',data)
    cfg = write_config(tmp_path/'fwi.yaml', dataset={'kind':'observations','path':'external.pt'})
    result = cli('invert','--config',cfg,'--output-root',tmp_path/'runs')
    assert result.returncode == 0, result.stderr
    final = torch.load(Path(json.loads(result.stdout)['run_dir'])/'final.pt')
    assert 'model_rmse' not in final['metrics']
    torch.testing.assert_close(final['source'],data.source,rtol=0,atol=0)


@pytest.mark.parametrize('change,message', [('unknown','unknown'),('missing','does not exist'),('bad_dataset','unknown'),('duplicate','duplicate')])
def test_bad_yaml_rejected_before_run_creation(tmp_path,change,message):
    cfg = write_config(tmp_path/'bad.yaml')
    value = yaml.safe_load(cfg.read_text())
    if change=='unknown': value['experiment']['velocity_lrr'] = 1
    elif change=='missing': value['dataset'] = {'kind':'observations','path':'missing.pt'}
    elif change=='bad_dataset': value['dataset']['ntt'] = 64
    else:
        cfg.write_text('experiment: {method: fwi}\ndataset: {kind: tiny, kind: observations}\n')
    if change!='duplicate':cfg.write_text(yaml.safe_dump(value))
    result=cli('invert','--config',cfg,'--output-root',tmp_path/'runs')
    assert result.returncode == 2 and message in result.stderr.lower(), result.stderr
    assert not (tmp_path/'runs').exists()


def test_simulate_refuses_overwrite_and_cuda_unavailable_is_clear(tmp_path):
    cfg=write_config(tmp_path/'config.yaml'); destination=tmp_path/'protected.pt'
    destination.write_bytes(b'preserved')
    result=cli('simulate','--config',cfg,'--output',destination)
    assert result.returncode==2 and 'overwrite' in result.stderr
    assert destination.read_bytes()==b'preserved'
    if not torch.cuda.is_available():
        result=cli('invert','--config',cfg,'--output-root',tmp_path/'runs','--device','cuda')
        assert result.returncode==2 and 'CUDA requested but unavailable' in result.stderr
        assert not (tmp_path/'runs').exists()


def test_notebooks_are_thin_bounded_public_clients():
    for path in Path('notebooks').glob('*.ipynb'):
        notebook=json.loads(path.read_text())
        source='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
        for forbidden in ('drive.mount','!pip','/content/','class Generator','class Critic','def train_fn'):
            assert forbidden not in source
        assert 'load_config' in source and 'prepare_data' in source and 'Experiment.create' in source
        assert 'tiny-' in source and 'plot_results' in source
        assert all(not cell.get('outputs') for cell in notebook['cells'])


def test_marmousi_preparation_preserves_grids_geometry_wavelets_without_full_solve(tmp_path):
    assert importlib.util.find_spec('fwigan.cli'), 'workflow loader is missing'
    from fwigan.cli import load_config, prepare_data
    value={'experiment':{'method':'fwigan','device':'cpu','epochs':1},
           'dataset':{'kind':'marmousi','truth_path':'fine.pt','initial_path':'coarse.pt','nt':32}}
    torch.save(torch.full((1000,3100),2100.),tmp_path/'fine.pt')
    torch.save(torch.full((100,310),2000.),tmp_path/'coarse.pt')
    cfg=tmp_path/'marmousi.yaml'; cfg.write_text(yaml.safe_dump(value))
    class InspectForward:
        def predict(self,velocity,source,acquisition,grid,sampling):
            assert grid.shape==(1000,3100) and grid.spacing==3.
            assert source.shape==(1,1,32) and sampling.dt==.003
            assert acquisition.sources[-1,0,1]==8700 and acquisition.receivers[0,-1,1]==8850
            return source.expand(30,60,32).clone()
    workflow=load_config(cfg); data=prepare_data(workflow,forward=InspectForward())
    assert data.grid.shape==(100,310) and data.grid.spacing==30.
    assert data.reference_velocity.shape==(100,310)
    assert data.initial_source.shape==(1,1,32)
    assert not torch.equal(data.initial_source,data.source)


def test_plotting_public_results_writes_physical_and_trace_figures(tmp_path):
    assert importlib.util.find_spec('fwigan.plotting'), 'public plotting client is missing'
    from fwigan.experiment import Experiment
    from fwigan import ExperimentConfig
    from fwigan.plotting import plot_results
    data=make_tiny_data(nt=32,receivers=8)
    run=Experiment.create(ExperimentConfig(method='fwi',device='cpu',epochs=1),data,tmp_path)
    run.run()
    figures=plot_results(data,torch.load(run.run_dir/'final.pt'),output_dir=tmp_path/'plots')
    assert len(figures)==2
    assert (tmp_path/'plots'/'velocity.png').stat().st_size>1000
    assert (tmp_path/'plots'/'traces.png').stat().st_size>1000
    import matplotlib.pyplot as plt
    for figure in figures:plt.close(figure)
