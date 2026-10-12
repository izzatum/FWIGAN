"""Shared YAML preparation and bounded command-line experiment clients.

Dataset paths are resolved relative to the YAML file. Torch artifacts must be
trusted local files: the pinned Torch release uses pickle for deserialization.
"""
from dataclasses import dataclass
import argparse
import json
from pathlib import Path
import sys

import deepwave
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import yaml

from .config import ExperimentConfig
from .domain import Acquisition, GridSpec, SamplingSpec, ObservationData, finite_tensor
from .data import make_tiny_data, load_observations, save_observations
from .experiment import Experiment
from .physics import LegacyDeepwave


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ValueError('YAML keys must be strings')
        if key in mapping:
            raise ValueError(f'duplicate YAML key: {key}')
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


@dataclass
class WorkflowConfig:
    experiment: ExperimentConfig
    dataset: dict
    path: Path


def _keys(value, allowed, label):
    if not isinstance(value, dict):
        raise ValueError(f'{label} must be a mapping')
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f'unknown {label} keys: {sorted(unknown)}')


def load_config(path, device=None):
    """Validate a complete workflow before preparing data or creating a run."""
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f'config does not exist: {path}')
    value = yaml.load(path.read_text(), Loader=_UniqueLoader)
    _keys(value, ('experiment', 'dataset'), 'workflow')
    if set(value) != {'experiment', 'dataset'}:
        raise ValueError('workflow requires experiment and dataset sections')
    numerical = value['experiment']
    if not isinstance(numerical, dict):
        raise ValueError('experiment must be a mapping')
    numerical = dict(numerical)
    if device is not None:
        numerical['device'] = device
    experiment = ExperimentConfig.from_dict(numerical)
    if experiment.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    dataset = value['dataset']
    if not isinstance(dataset, dict):
        raise ValueError('dataset must be a mapping')
    dataset = dict(dataset)
    kind = dataset.get('kind')
    allowed = {'tiny': ('kind','seed','nt','shots','receivers','noise'),
               'observations': ('kind','path'),
               'marmousi': ('kind','truth_path','initial_path','nt')}
    if kind not in allowed:
        raise ValueError('dataset kind must be tiny, observations or marmousi')
    _keys(dataset, allowed[kind], 'dataset')
    required = {'tiny': (), 'observations': ('path',), 'marmousi': ('truth_path','initial_path')}[kind]
    for name in required:
        if not isinstance(dataset.get(name), str) or not dataset[name]:
            raise ValueError(f'dataset requires {name}')
        resolved = (path.parent / dataset[name]).resolve()
        if not resolved.is_file():
            raise ValueError(f'dataset {name} does not exist: {resolved}')
        dataset[name] = str(resolved)
    return WorkflowConfig(experiment, dataset, path)


def prepare_data(workflow, forward=None):
    """Return raw, CPU-portable data; preserve legacy Marmousi physical grids."""
    settings = workflow.dataset
    kind = settings['kind']
    if kind == 'observations':
        return load_observations(settings['path'])
    if kind == 'tiny':
        options = {key: value for key, value in settings.items() if key != 'kind'}
        options.setdefault('seed', workflow.experiment.seed)
        return make_tiny_data(**options)
    truth = torch.load(settings['truth_path'], map_location='cpu')
    coarse = torch.load(settings['initial_path'], map_location='cpu')
    finite_tensor(truth, 'Marmousi fine truth', 2)
    finite_tensor(coarse, 'Marmousi coarse model', 2)
    if tuple(truth.shape) != (1000,3100) or tuple(coarse.shape) != (100,310):
        raise ValueError('Marmousi requires original fine [1000,3100] and coarse [100,310] grids')
    if (truth <= 0).any() or (coarse <= 0).any():
        raise ValueError('Marmousi velocities must be positive')
    sampling = SamplingSpec(settings.get('nt',2000), .003)
    sources = torch.zeros(30,1,2); sources[:,0,1] = torch.arange(30)*300.
    receivers = torch.zeros(30,60,2); receivers[:,:,1] = torch.arange(60)*150.
    acquisition = Acquisition(sources,receivers)
    source = deepwave.wavelets.ricker(7.,sampling.nt,sampling.dt,1/7.).reshape(1,1,-1)
    guess = deepwave.wavelets.ricker(10.,sampling.nt,sampling.dt,1/7.).reshape(1,1,-1)
    initial = torch.from_numpy(gaussian_filter(coarse.numpy(),sigma=10).astype(np.float32))
    device = workflow.experiment.device
    if device == 'auto': device = 'cuda' if torch.cuda.is_available() else 'cpu'
    solver = LegacyDeepwave() if forward is None else forward
    fine_grid = GridSpec(tuple(truth.shape),3.)
    inversion_grid = GridSpec(tuple(coarse.shape),30.)
    with torch.no_grad():
        traces = solver.predict(truth.to(device),source.to(device),acquisition.to(device),
                                fine_grid,sampling).detach().cpu()
    # DeepWave 0.0.9 injects a coefficient into one cell without area division.
    # Preserve fine observations, but express their known forcing on the inverse
    # grid. This exact geometric factor is not an empirical trace-amplitude fit.
    known_source = source * (fine_grid.spacing / inversion_grid.spacing)**2
    return ObservationData(traces,acquisition,inversion_grid,sampling,
                           known_source,initial,truth[::10,::10].clone(),guess)


def _parser():
    parser = argparse.ArgumentParser(prog='fwigan',description='Bounded FWI and legacy FWIGAN experiments')
    commands = parser.add_subparsers(dest='command',required=True)
    simulate = commands.add_parser('simulate',help='create raw observation artifact')
    invert = commands.add_parser('invert',help='create and run a new experiment')
    for command in (simulate,invert):
        command.add_argument('--config',required=True,type=Path)
        command.add_argument('--device',choices=('cpu','cuda','auto'))
    simulate.add_argument('--output',required=True,type=Path)
    invert.add_argument('--output-root',required=True,type=Path)
    invert.add_argument('--max-batches',type=int)
    resume = commands.add_parser('resume',help='continue the same run from a trusted checkpoint')
    resume.add_argument('checkpoint',type=Path)
    resume.add_argument('--device',choices=('cpu','cuda','auto'))
    resume.add_argument('--max-batches',type=int)
    evaluate = commands.add_parser('evaluate',help='evaluate current checkpoint with raw amplitudes')
    evaluate.add_argument('run_dir',type=Path)
    evaluate.add_argument('--device',choices=('cpu','cuda','auto'),default='cpu')
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command in ('simulate','invert'):
            workflow = load_config(arguments.config,arguments.device)
            if arguments.command == 'simulate' and arguments.output.exists():
                raise ValueError(f'refusing to overwrite observations: {arguments.output}')
            if arguments.command == 'invert' and arguments.max_batches is not None and arguments.max_batches < 1:
                raise ValueError('max_batches must be a positive integer')
            data = prepare_data(workflow)
            if arguments.command == 'simulate':
                save_observations(arguments.output,data)
                result = {'observations':str(arguments.output.resolve()),'data_fingerprint':data.fingerprint()}
            else:
                run = Experiment.create(workflow.experiment,data,arguments.output_root)
                run.run(max_batches=arguments.max_batches)
                result = {'run_dir':str(run.run_dir),'position':vars(run.position)}
        elif arguments.command == 'resume':
            if arguments.max_batches is not None and arguments.max_batches < 1:
                raise ValueError('max_batches must be a positive integer')
            run = Experiment.resume(arguments.checkpoint,device=arguments.device)
            run.run(max_batches=arguments.max_batches)
            result = {'run_dir':str(run.run_dir),'position':vars(run.position)}
        else:
            run = Experiment.resume(arguments.run_dir/'checkpoint.pt',device=arguments.device)
            result = {'run_dir':str(run.run_dir),'position':vars(run.position),'metrics':run.evaluate()}
        print(json.dumps(result,allow_nan=False))
        return 0
    except (ValueError,TypeError,KeyError,OSError,RuntimeError,yaml.YAMLError) as error:
        print(f'fwigan: {error}',file=sys.stderr)
        return 2
