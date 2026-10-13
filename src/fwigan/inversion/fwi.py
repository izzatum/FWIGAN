"""Raw-amplitude FWI: one projected Adam update over the full shot dataset."""
import math
import torch
from .adversarial import _PhysicalStrategy, _PortableGradScaler


class FWIStrategy(_PhysicalStrategy):
    def __init__(self, parameters, forward, data, config):
        if config.method != 'fwi': raise ValueError('FWIStrategy requires method=fwi')
        if ((parameters.velocity < config.velocity_min) | (parameters.velocity > config.velocity_max)).any():
            raise ValueError('initial fwi velocity must lie within the configured bounds')
        super().__init__(parameters, forward, data, config)
        if not torch.equal(parameters.source, self.data.source):
            raise ValueError('fwi requires the fixed known observation source')
        # No autocast or scaling is needed for the float32 least-squares objective.
        self.physical_scaler = _PortableGradScaler(enabled=False)

    def step_epoch(self):
        self.physical_optimizer.zero_grad(set_to_none=True)
        data_loss = self.parameters.velocity.new_zeros(())
        with torch.cuda.amp.autocast(enabled=False):
            for start in range(0, self.data.traces.shape[0], self.config.batch_size):
                indices = slice(start, start+self.config.batch_size)
                residual = self._predict(indices)-self.data.traces[indices]
                loss = residual.square().sum()/self.data.traces.numel()
                loss.backward()
                data_loss += loss.detach()
            regularization = self.parameters.velocity.new_zeros(())
            if self.config.regularization:
                v = self.parameters.velocity
                differences = []
                if v.shape[0]>1: differences.append((v[1:]-v[:-1]).square().mean())
                if v.shape[1]>1: differences.append((v[:,1:]-v[:,:-1]).square().mean())
                if differences:
                    regularization = self.config.regularization * sum(differences)
                    regularization.backward()
            finite = torch.isfinite(data_loss + regularization.detach())
            finite = finite & torch.isfinite(self.parameters.velocity.grad).all()
            if not finite.item():
                self.physical_optimizer.zero_grad(set_to_none=True)
                raise ValueError('nonfinite fwi objective or accumulated gradients')
            self.physical_optimizer.step()
            self.counters['physical_updates'] += 1
            with torch.no_grad(): self.parameters.velocity.clamp_(self.config.velocity_min, self.config.velocity_max)
        values=torch.stack([data_loss, regularization.detach(), data_loss+regularization.detach()]).cpu().tolist()
        if not all(math.isfinite(value) for value in values): raise ValueError('nonfinite fwi metrics')
        return dict(zip(('data_loss','regularization_loss','loss'),values), physical_updates=1,physical_skipped=0)
