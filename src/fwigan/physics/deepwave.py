"""Adapter for DeepWave 0.0.9's fourth-order scalar Propagator."""
from typing import Protocol
import deepwave
from torch import Tensor
from ..domain import Acquisition, GridSpec, SamplingSpec, validate_physics


class ForwardModel(Protocol):
    def predict(self, velocity: Tensor, source: Tensor, acquisition: Acquisition,
                grid: GridSpec, sampling: SamplingSpec) -> Tensor: ...


class LegacyDeepwave:
    def predict(self, velocity, source, acquisition, grid, sampling):
        validate_physics(velocity, source, acquisition, grid, sampling)
        # 0.0.9 hardcodes fd_width=2 (accuracy=4); it has no accuracy keyword.
        prop = deepwave.scalar.Propagator({'vp': velocity}, float(grid.spacing),
                                          pml_width=grid.pml_width, survey_pad=None)
        amplitudes = source.permute(2, 0, 1)
        if source.shape[0] == 1:
            amplitudes = amplitudes.repeat(1, acquisition.sources.shape[0], 1)
        origin = acquisition.sources.new_tensor(grid.origin)
        native = prop(amplitudes.contiguous(), acquisition.sources - origin,
                      acquisition.receivers - origin, float(sampling.dt))
        # Preserve native storage order: legacy randn_like noise draws follow it.
        return native.permute(1, 2, 0)
