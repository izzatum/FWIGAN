"""Self-contained CPU tensor artifacts. Load only trusted torch.save files."""
from dataclasses import asdict
import os
from pathlib import Path
import tempfile
import torch
from ..domain import Acquisition, GridSpec, SamplingSpec, ObservationData

OBSERVATION_SCHEMA = 1


def save_observations(path, data, overwrite=False):
    data.validate()
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f'refusing to overwrite observations: {path}')
    data = data.to('cpu')
    artifact = {'schema': OBSERVATION_SCHEMA, 'grid': asdict(data.grid), 'sampling': asdict(data.sampling),
        'traces': data.traces.detach(), 'sources': data.acquisition.sources.detach(),
        'receivers': data.acquisition.receivers.detach(), 'source': data.source.detach(),
        'initial_velocity': data.initial_velocity.detach(),
        'reference_velocity': None if data.reference_velocity is None else data.reference_velocity.detach(),
        'initial_source': None if data.initial_source is None else data.initial_source.detach(),
        'fingerprint': data.fingerprint()}
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(artifact, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # Atomic exclusive publication also rejects racing writers.
            os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_observations(path):
    artifact = torch.load(path, map_location='cpu')
    if not isinstance(artifact, dict) or artifact.get('schema') != OBSERVATION_SCHEMA:
        raise ValueError('unsupported observation schema')
    try:
        data = ObservationData(artifact['traces'], Acquisition(artifact['sources'], artifact['receivers']),
            GridSpec(**artifact['grid']), SamplingSpec(**artifact['sampling']), artifact['source'],
            artifact['initial_velocity'], artifact['reference_velocity'], artifact['initial_source'])
        if artifact['fingerprint'] != data.fingerprint():
            raise ValueError('observation fingerprint mismatch')
        return data
    except (KeyError, TypeError) as error:
        raise ValueError('invalid observation artifact fields') from error
