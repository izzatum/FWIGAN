"""Numerical settings; filesystem and data preparation live outside this config."""
from dataclasses import asdict, dataclass, fields
import math
from .domain import positive_int


@dataclass
class ExperimentConfig:
    method: str = 'fwigan'
    seed: int = 42
    device: str = 'auto'
    epochs: int = 401
    batches: int = 6
    batch_size: int = 2
    critic_cycles: int = 6
    amp: bool = False
    noise: bool = False
    observation_noise_level: float = .05
    checkpoint_interval: int = 50
    evaluation_interval: int = 10
    sigma_init: float = 1.
    velocity_lr: float = 5.
    source_lr: float = .001
    sigma_lr: float = .01
    critic_lr: float = .001
    velocity_min: float = 1000.
    velocity_max: float = 6000.
    regularization: float = 0.
    physical_scheduler_step_size: int = 150
    critic_scheduler_step_size: int = 100
    scheduler_gamma: float = .5
    adam_betas: tuple[float, float] = (.5, .9)
    gradient_penalty_weight: float = 10.
    critic_drift: float = .001
    velocity_clip: float = 1000.
    critic_clip: float = 1e6
    train_source: bool | None = None
    train_sigma: bool | None = None
    observation_mode: str = 'raw'

    def __post_init__(self):
        if self.method not in ('fwigan', 'fwi'):
            raise ValueError('method must be fwigan or fwi')
        if self.device not in ('cpu', 'cuda', 'auto'):
            raise ValueError('device must be cpu, cuda or auto')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        for name in ('epochs', 'batches', 'batch_size', 'critic_cycles', 'physical_scheduler_step_size', 'critic_scheduler_step_size', 'checkpoint_interval', 'evaluation_interval'):
            positive_int(getattr(self, name), name)
        for name in ('velocity_lr', 'source_lr', 'sigma_lr', 'critic_lr', 'velocity_min', 'velocity_max',
                     'scheduler_gamma', 'velocity_clip', 'critic_clip'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('observation_noise_level', 'sigma_init', 'regularization', 'gradient_penalty_weight', 'critic_drift'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.velocity_min >= self.velocity_max:
            raise ValueError('velocity_min must be smaller than velocity_max')
        if self.scheduler_gamma > 1:
            raise ValueError('scheduler_gamma must be at most 1')
        self.adam_betas = tuple(self.adam_betas)
        if len(self.adam_betas) != 2 or any(not math.isfinite(b) or not 0 <= b < 1 for b in self.adam_betas):
            raise ValueError('adam_betas must contain two values in [0,1)')
        for name in ('amp', 'noise'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be boolean')
        if self.train_source is None:
            self.train_source = self.method == 'fwigan'
        if self.train_sigma is None:
            self.train_sigma = self.method == 'fwigan' and self.noise
        if type(self.train_source) is not bool or type(self.train_sigma) is not bool:
            raise ValueError('train_source and train_sigma must be boolean')
        if self.observation_mode != 'raw':
            raise ValueError('observation_mode must be raw; legacy preprocessing is explicit during training')
        if self.train_sigma and not self.noise:
            raise ValueError('train_sigma requires noise=True')
        if self.method == 'fwi' and (self.noise or self.train_source or self.train_sigma):
            raise ValueError('fwi requires a fixed source, fixed sigma and noise=False')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        unknown = set(value) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f'unknown config keys: {sorted(unknown)}')
        return cls(**value)
