"""Small real-solver dataset for numerical integration and development."""
import math
import torch
import deepwave
from ..domain import GridSpec, SamplingSpec, Acquisition, ObservationData, positive_int
from ..physics import LegacyDeepwave


def make_tiny_data(seed=42, noise=0., nt=128, shots=2, receivers=60):
    positive_int(shots, 'shots')
    positive_int(receivers, 'receivers')
    if shots > 70 or receivers > 70:
        raise ValueError('tiny acquisition cannot exceed 70 grid-aligned surface points')
    if type(seed) is not int or seed < 0 or not math.isfinite(noise) or noise < 0:
        raise ValueError('seed must be nonnegative and noise finite and nonnegative')
    grid = GridSpec((40, 70), 30.)
    sampling = SamplingSpec(nt, .003)
    sources = torch.zeros(shots, 1, 2)
    sources[:, 0, 1] = torch.linspace(10, 59, shots).round() * grid.spacing
    receiver_locations = torch.zeros(shots, receivers, 2)
    receiver_locations[..., 1] = torch.arange(receivers).float() * grid.spacing
    acquisition = Acquisition(sources, receiver_locations)
    source = deepwave.wavelets.ricker(25., nt, sampling.dt, 1/25.).reshape(1, 1, nt)
    initial = torch.full(grid.shape, 2000.)
    truth = torch.full(grid.shape, 2100.)
    with torch.no_grad():
        traces = LegacyDeepwave().predict(truth, source, acquisition, grid, sampling)
        if noise:
            generator = torch.Generator(device='cpu').manual_seed(seed)
            traces = traces + noise * torch.empty_like(traces).normal_(generator=generator)
    return ObservationData(traces, acquisition, grid, sampling, source, initial, truth)
