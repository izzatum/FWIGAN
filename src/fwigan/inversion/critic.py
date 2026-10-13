"""Original shot-stack critic, including its historical dense-layer geometry."""
from torch import nn
from ..domain import positive_int
from ..data import to_critic_layout


class LegacyCritic(nn.Module):
    @staticmethod
    def validate_geometry(shots, nt, receivers):
        for name, value in (('shots', shots), ('nt', nt), ('receivers', receivers)):
            positive_int(value, name)
        if nt < 32 or not 32 <= receivers < 64:
            raise ValueError('legacy critic requires nt >= 32 and 32 <= receivers < 64 after five pooling layers')

    def __init__(self, shots, nt, receivers):
        super().__init__()
        self.validate_geometry(shots, nt, receivers)
        self.geometry = (shots, nt, receivers)
        layers = []
        channels = shots
        for _ in range(5):
            layers.extend([nn.Conv2d(channels, channels * 2, 3, stride=1, padding=1),
                           nn.MaxPool2d(2, stride=2), nn.LeakyReLU(.1)])
            channels *= 2
        self.conv_block = nn.Sequential(*layers)
        dense = shots * 32 * (nt // 32)
        self.dense_block = nn.Sequential(nn.Linear(dense, dense // 2), nn.LeakyReLU(.1),
            nn.Linear(dense // 2, 2000), nn.LeakyReLU(.1), nn.Linear(2000, 1))

    def forward(self, x):
        if tuple(x.shape) != (1, *self.geometry):
            raise ValueError(f'critic input must have shape {(1, *self.geometry)}')
        return self.dense_block(self.conv_block(x).reshape(1, -1))

    def from_canonical(self, traces):
        return self(to_critic_layout(traces))
