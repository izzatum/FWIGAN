"""Portable trusted-file snapshots and atomic publication with run ownership."""
from dataclasses import asdict
import hashlib
import json
import os
import pickle
import struct
from pathlib import Path
import random
import tempfile
import numpy as np
import torch
from .domain import Acquisition, GridSpec, SamplingSpec, ObservationData
from .inversion.adversarial import _portable

CHECKPOINT_SCHEMA = 1
DATA_FIELDS = {'schema','grid','sampling','traces','sources','receivers','source',
               'initial_velocity','reference_velocity','initial_source','fingerprint'}


def fingerprint_config(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()


def fingerprint_tensor(tensor):
    return hashlib.sha256(str(tuple(tensor.shape)).encode()+tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def dataset_snapshot(data):
    return _portable(dict(schema=1,grid=asdict(data.grid),sampling=asdict(data.sampling),traces=data.traces,
        sources=data.acquisition.sources,receivers=data.acquisition.receivers,source=data.source,
        initial_velocity=data.initial_velocity,reference_velocity=data.reference_velocity,
        initial_source=data.initial_source,fingerprint=data.fingerprint()))


def restore_dataset(payload):
    if not isinstance(payload,dict) or set(payload)!=DATA_FIELDS or type(payload['schema']) is not int or payload['schema']!=1:
        raise ValueError('invalid dataset snapshot schema or fields')
    try:
        data=ObservationData(payload['traces'],Acquisition(payload['sources'],payload['receivers']),
            GridSpec(**payload['grid']),SamplingSpec(**payload['sampling']),payload['source'],
            payload['initial_velocity'],payload['reference_velocity'],payload['initial_source'])
        if data.fingerprint()!=payload['fingerprint']:raise ValueError('dataset fingerprint mismatch')
        return data
    except (KeyError,TypeError,RuntimeError,AttributeError) as error:
        raise ValueError('invalid dataset snapshot') from error


def load_snapshot(path):
    """torch.save is pickle based: only open files from trusted sources."""
    try:
        value=torch.load(path,map_location='cpu')
    except (OSError,RuntimeError,EOFError,ValueError,pickle.UnpicklingError) as error:
        raise ValueError(f'cannot read snapshot: {path}') from error
    if not isinstance(value,dict):raise ValueError('invalid snapshot structure')
    return value


def atomic_save(path,payload):
    """An existing artifact may only be replaced by its owning run and kind."""
    path=Path(path)
    if path.is_symlink():raise FileExistsError(f'refusing to overwrite symlink: {path}')
    for parent in path.resolve().parents:
        manifest=parent/'run.json'
        if manifest.exists():
            try:owner=json.loads(manifest.read_text())['run_id']
            except (KeyError,ValueError):raise FileExistsError('invalid destination run ownership')
            if owner!=payload['run_id']:raise FileExistsError('destination belongs to another run')
            break
    existed=path.exists()
    if existed:
        try:previous=load_snapshot(path)
        except ValueError as error:raise FileExistsError(f'refusing to overwrite unknown artifact: {path}') from error
        if previous.get('run_id')!=payload['run_id'] or previous.get('kind')!=payload['kind']:
            raise FileExistsError(f'refusing to overwrite another run or artifact: {path}')
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(dir=path.parent,prefix=f'.{path.name}.',suffix='.tmp')
    try:
        with os.fdopen(fd,'wb') as stream:
            torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
        if existed:os.replace(temporary,path)
        else:os.link(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)
    return path


def seed_all(seed):
    random.seed(seed);np.random.seed(seed % (2**32));torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False


def rng_snapshot():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state().clone(),
        cuda=[v.cpu().clone() for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
        cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark)


def validate_rng(state):
    if not isinstance(state,dict) or set(state)!={'python','numpy','torch','cuda','cudnn_deterministic','cudnn_benchmark'}:
        raise ValueError('invalid RNG state fields')
    try:
        random.Random().setstate(state['python']);np.random.RandomState().set_state(state['numpy'])
        value=state['torch']
        if not torch.is_tensor(value) or value.ndim!=1 or value.dtype!=torch.uint8:raise ValueError('invalid Torch RNG state')
        torch.Generator().set_state(value)
        if not isinstance(state['cuda'],list):raise ValueError('invalid CUDA RNG states')
        for value in state['cuda']:
            if not torch.is_tensor(value) or value.dtype!=torch.uint8 or value.ndim!=1 or value.numel() not in (808,816) or not value.is_contiguous():
                raise ValueError('invalid CUDA RNG state')
            # Torch 1.13.1 stores 800 BC bytes, seed, and optional Philox offset.
            if value.numel()==816 and struct.unpack('<q',bytes(value[808:].tolist()))[0] % 4:
                raise ValueError('invalid CUDA RNG Philox offset')
        for key in ('cudnn_deterministic','cudnn_benchmark'):
            if type(state[key]) is not bool:raise ValueError('invalid cuDNN RNG settings')
    except (TypeError,RuntimeError,IndexError) as error:raise ValueError('invalid RNG state') from error


def restore_rng(state):
    validate_rng(state)
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if torch.cuda.is_available():
        for index,value in enumerate(state['cuda'][:torch.cuda.device_count()]):torch.cuda.set_rng_state(value,index)
    torch.backends.cudnn.deterministic=state['cudnn_deterministic'];torch.backends.cudnn.benchmark=state['cudnn_benchmark']
