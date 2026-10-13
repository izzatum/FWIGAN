"""Physical contracts: metres in [z,x], seconds, and float32 tensors."""
from dataclasses import dataclass
import hashlib
import json
import math
import torch
from torch import Tensor


def finite_tensor(value: Tensor, name: str, ndim: int | None = None) -> None:
    if not isinstance(value, Tensor) or value.dtype != torch.float32:
        raise ValueError(f'{name} must be a float32 tensor')
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f'{name} must have {ndim} dimensions')
    if value.numel() == 0 or not torch.isfinite(value).all().item():
        raise ValueError(f'{name} must contain finite, nonempty values')


def positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')


@dataclass(frozen=True)
class GridSpec:
    shape: tuple[int, int]
    spacing: float
    origin: tuple[float, float] = (0., 0.)
    pml_width: int = 10
    accuracy: int = 4

    def __post_init__(self):
        if len(self.shape) != 2:
            raise ValueError('grid shape must be [nz,nx]')
        for size in self.shape:
            positive_int(size, 'grid size')
        if not math.isfinite(self.spacing) or self.spacing <= 0:
            raise ValueError('spacing must be finite and positive')
        if len(self.origin) != 2 or not all(math.isfinite(v) for v in self.origin):
            raise ValueError('origin must contain finite [z,x] coordinates')
        if type(self.pml_width) is not int or self.pml_width < 0:
            raise ValueError('pml_width must be a nonnegative integer')
        if self.accuracy != 4:
            raise ValueError('DeepWave 0.0.9 supports only accuracy=4')
        object.__setattr__(self, 'shape', tuple(self.shape))
        object.__setattr__(self, 'origin', tuple(self.origin))


@dataclass(frozen=True)
class SamplingSpec:
    nt: int
    dt: float

    def __post_init__(self):
        positive_int(self.nt, 'nt')
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError('dt must be finite and positive seconds')


@dataclass
class Acquisition:
    sources: Tensor
    receivers: Tensor

    def __post_init__(self):
        self.validate()

    def validate(self):
        for name in ('sources', 'receivers'):
            value = getattr(self, name)
            finite_tensor(value, name, 3)
            if value.shape[-1] != 2:
                raise ValueError(f'{name} must have shape [shots,points,2] in [z,x]')
        if self.sources.shape[0] != self.receivers.shape[0]:
            raise ValueError('source and receiver shot counts differ')
        if self.sources.device != self.receivers.device:
            raise ValueError('acquisition tensors must share a device')
        return self

    def select(self, indices):
        if isinstance(indices, int):
            indices = [indices]
        return Acquisition(self.sources[indices], self.receivers[indices])

    def to(self, device):
        return Acquisition(self.sources.to(device), self.receivers.to(device))

    def validate_grid(self, grid: GridSpec):
        self.validate()
        for name in ('sources', 'receivers'):
            points = getattr(self, name)
            relative = points - points.new_tensor(grid.origin)
            spacing = points.new_tensor(grid.spacing)
            cells = relative / spacing
            maximum = (points.new_tensor(grid.shape) - 1) * spacing
            if (relative < 0).any() or (relative > maximum).any():
                raise ValueError(f'{name} must be inside the grid')
            if not torch.allclose(cells, cells.round(), rtol=0., atol=1e-5):
                raise ValueError(f'{name} must be grid-aligned; off-grid points are unsupported')
            # Native 0.0.9 truncates before adding padding. Near-grid float32
            # values below an integer can select the previous cell, including
            # a last-grid-point value that cannot be nudged without exiting it.
            if not torch.equal(cells.long(), cells.round().long()):
                raise ValueError(f'{name} has unsupported coordinates: DeepWave would truncate to another grid cell')


def validate_physics(velocity, source, acquisition, grid, sampling):
    acquisition.validate_grid(grid)
    finite_tensor(velocity, 'velocity', 2)
    if tuple(velocity.shape) != grid.shape or (velocity <= 0).any():
        raise ValueError('velocity must be positive with shape [nz,nx] matching grid')
    finite_tensor(source, 'source', 3)
    shots, sources, nt = source.shape
    if shots not in (1, acquisition.sources.shape[0]) or sources != acquisition.sources.shape[1] or nt != sampling.nt:
        raise ValueError('source must have shape [shots or 1,sources,time] matching acquisition and sampling')
    if velocity.device != source.device or velocity.device != acquisition.sources.device:
        raise ValueError('velocity, source and acquisition must share a device')


@dataclass
class ObservationData:
    traces: Tensor
    acquisition: Acquisition
    grid: GridSpec
    sampling: SamplingSpec
    source: Tensor
    initial_velocity: Tensor
    reference_velocity: Tensor | None = None
    initial_source: Tensor | None = None

    def __post_init__(self):
        self.validate()

    def validate(self):
        validate_physics(self.initial_velocity, self.source, self.acquisition, self.grid, self.sampling)
        finite_tensor(self.traces, 'traces', 3)
        expected = (self.acquisition.receivers.shape[0], self.acquisition.receivers.shape[1], self.sampling.nt)
        if tuple(self.traces.shape) != expected:
            raise ValueError('traces must have shape [shots,receivers,time] matching acquisition and sampling')
        if self.traces.device != self.initial_velocity.device:
            raise ValueError('observation tensors must share a device')
        if self.reference_velocity is not None:
            validate_physics(self.reference_velocity, self.source, self.acquisition, self.grid, self.sampling)
        if self.initial_source is not None:
            validate_physics(self.initial_velocity, self.initial_source, self.acquisition, self.grid, self.sampling)
        return self

    def to(self, device):
        return ObservationData(self.traces.to(device), self.acquisition.to(device), self.grid,
            self.sampling, self.source.to(device), self.initial_velocity.to(device),
            None if self.reference_velocity is None else self.reference_velocity.to(device),
            None if self.initial_source is None else self.initial_source.to(device))

    def fingerprint(self) -> str:
        """Hash canonical CPU content, independent of strides and device."""
        from dataclasses import asdict
        self.validate()
        digest = hashlib.sha256(json.dumps({'grid': asdict(self.grid), 'sampling': asdict(self.sampling)},
                                          sort_keys=True, separators=(',', ':')).encode())
        for name, tensor in (('traces', self.traces), ('sources', self.acquisition.sources),
            ('receivers', self.acquisition.receivers), ('source', self.source),
            ('initial_velocity', self.initial_velocity), ('reference_velocity', self.reference_velocity),
            ('initial_source', self.initial_source)):
            digest.update(name.encode())
            if tensor is None:
                digest.update(b'null')
            else:
                digest.update(str(tuple(tensor.shape)).encode())
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
