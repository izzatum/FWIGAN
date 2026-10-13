"""Explicit legacy transforms; propagation and observation artifacts remain raw."""
from ..domain import finite_tensor


def legacy_normalize(traces):
    finite_tensor(traces, 'traces', 3)
    return traces / (traces.max(dim=-1, keepdim=True).values.abs() + 1e-10)


def to_critic_layout(traces):
    finite_tensor(traces, 'traces', 3)
    return traces.permute(0, 2, 1).unsqueeze(0)


def from_critic_layout(data):
    finite_tensor(data, 'critic data', 4)
    if data.shape[0] != 1:
        raise ValueError('legacy critic data must have shape [1,shot_group,time,receivers]')
    return data.squeeze(0).permute(0, 2, 1)
