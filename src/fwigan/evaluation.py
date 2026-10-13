"""Deterministic raw-amplitude evaluation, independent of GAN generator noise."""
from dataclasses import dataclass
import torch
from .domain import finite_tensor
from .checkpoints import rng_snapshot, restore_rng


@dataclass
class SolverCounts:
    forward: int = 0
    adjoint: int = 0


class CountedForward:
    """Count predict calls and reverse passes; LegacyDeepwave solves once each."""
    def __init__(self,forward,counts):self.forward,self.counts=forward,counts

    def predict(self,*args,**kwargs):
        self.counts.forward+=1
        result=self.forward.predict(*args,**kwargs)
        if result.requires_grad:
            def reverse(gradient):
                self.counts.adjoint+=1
                return gradient
            result.register_hook(reverse)
        return result


def evaluate_raw(parameters,forward,data):
    """Return CPU traces plus metrics; preserve all training random streams."""
    rng=rng_snapshot()
    try:
        with torch.no_grad():
            predicted=forward.predict(parameters.velocity,parameters.source,data.acquisition,data.grid,data.sampling)
            finite_tensor(predicted,'predicted',3)
            if predicted.shape!=data.traces.shape:raise ValueError('predicted traces must match observations')
            residual=(predicted-data.traces).square().mean()
            if not torch.isfinite(residual):raise ValueError('nonfinite raw residual')
            metrics={'raw_residual_mse':residual.item()}
            if data.reference_velocity is not None and data.reference_velocity.shape==parameters.velocity.shape:
                metrics['model_rmse']=(parameters.velocity-data.reference_velocity).square().mean().sqrt().item()
            return predicted.detach().cpu().clone(),metrics
    finally:restore_rng(rng)
