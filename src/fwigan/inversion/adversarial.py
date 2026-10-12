"""Legacy adversarial objective with one physical adjoint per shot group."""
import copy
import math
from contextlib import contextmanager
import torch
from torch import nn
from ..domain import finite_tensor, validate_physics
from ..data import legacy_normalize, to_critic_layout
from .critic import LegacyCritic


class _CountedAdam(torch.optim.Adam):
    """Host counter increments only after a completed optimizer.step call."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.completed_steps = 0

    def step(self, closure=None):
        result = super().step(closure)
        self.completed_steps += 1
        return result


def _portable(value):
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: _portable(v) for k,v in value.items()}
    if isinstance(value, list): return [_portable(v) for v in value]
    if isinstance(value, tuple): return tuple(_portable(v) for v in value)
    return copy.deepcopy(value)


class _PhysicalStrategy:
    def __init__(self, parameters, forward, data, config):
        data.validate()
        self.parameters, self.forward, self.config = parameters, forward, copy.deepcopy(config)
        self.device = parameters.velocity.device
        self.data = data if data.traces.device == self.device else data.to(self.device)
        validate_physics(parameters.velocity, parameters.source, self.data.acquisition, data.grid, data.sampling)
        finite_tensor(parameters.sigma, 'sigma')
        self.physical_optimizer = _CountedAdam(parameters.optimizer_groups(config))
        self.physical_scheduler = torch.optim.lr_scheduler.StepLR(self.physical_optimizer,
            config.physical_scheduler_step_size, gamma=config.scheduler_gamma)
        self.physical_scaler = torch.cuda.amp.GradScaler(enabled=config.amp and self.device.type == 'cuda')
        self.counters = dict(physical_updates=0, physical_skipped=0)

    def _predict(self, indices):
        source = self.parameters.source
        if source.shape[0] != 1: source = source[indices]
        return self.forward.predict(self.parameters.velocity, source, self.data.acquisition.select(indices),
                                    self.data.grid, self.data.sampling)

    def _update(self, optimizer, scaler, name):
        before = optimizer.completed_steps
        scaler.step(optimizer)
        scaler.update()
        actual = optimizer.completed_steps - before
        self.counters[f'{name}_updates'] += actual
        self.counters[f'{name}_skipped'] += 1 - actual
        return actual

    def finish_epoch(self):
        if self.counters['physical_updates']: self.physical_scheduler.step()

    def _signature(self):
        config = self.config.to_dict()
        for key in ('epochs', 'device'): config.pop(key)
        return dict(method=self.config.method, config=config,
                    shape=tuple(self.data.traces.shape), grid=self.data.grid, sampling=self.data.sampling)

    def state_dict(self):
        return _portable(dict(schema=1, signature=self._signature(), counters=self.counters,
            physical_optimizer=self.physical_optimizer.state_dict(),
            physical_scheduler=self.physical_scheduler.state_dict(), physical_scaler=self.physical_scaler.state_dict()))

    def _validate_state(self, state):
        if not isinstance(state, dict) or state.get('schema') != 1 or state.get('signature') != self._signature():
            raise ValueError('incompatible strategy state schema or configuration')
        for name in ('physical', 'critic'):
            optimizer = getattr(self, f'{name}_optimizer', None)
            if optimizer is None: continue
            try:
                groups = state[f'{name}_optimizer']['param_groups']
                if [g['name'] for g in groups] != [g['name'] for g in optimizer.param_groups]:
                    raise ValueError('incompatible named optimizer groups')
            except (KeyError, TypeError) as error:
                raise ValueError('invalid strategy optimizer state') from error
        counters=state.get('counters', {})
        if set(counters) != set(self.counters) or any(type(v) is not int or v<0 for v in counters.values()):
            raise ValueError('invalid strategy update counters')

    def load_state_dict(self, state):
        self._validate_state(state)
        try:
            self.physical_optimizer.load_state_dict(state['physical_optimizer'])
            self.physical_scheduler.load_state_dict(state['physical_scheduler'])
            self.physical_scaler.load_state_dict(state['physical_scaler'])
            self.counters = copy.deepcopy(state['counters'])
        except (KeyError, TypeError, RuntimeError) as error:
            raise ValueError('invalid physical strategy state') from error


@contextmanager
def _frozen(module):
    flags = [p.requires_grad for p in module.parameters()]
    try:
        for p in module.parameters(): p.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(module.parameters(), flags): p.requires_grad_(flag)


def gradient_penalty(critic, observed, predicted):
    # Preserve the notebook's CPU random stream and single beta per shot stack.
    beta = torch.rand((1,1,1,1)).to(observed.device)
    mixed = (observed * beta + predicted.detach() * (1-beta)).requires_grad_(True)
    scores = critic(mixed)
    gradient = torch.autograd.grad(scores, mixed, torch.ones_like(scores), create_graph=True, retain_graph=True)[0]
    return (gradient.reshape(1,-1).norm(2,dim=1)-1).square().mean()


class FWIGANStrategy(_PhysicalStrategy):
    def __init__(self, parameters, forward, data, config, observed=None):
        if config.method != 'fwigan': raise ValueError('FWIGANStrategy requires method=fwigan')
        shots = data.traces.shape[0]
        if shots % config.batches:
            raise ValueError('legacy shot count must be divisible by the number of strided batches')
        self.shots_per_batch = shots // config.batches
        LegacyCritic.validate_geometry(self.shots_per_batch, data.sampling.nt, data.traces.shape[1])
        super().__init__(parameters, forward, data, config)
        if observed is not None:
            finite_tensor(observed, 'observed', 3)
        target = legacy_normalize(self.data.traces) if observed is None else observed.to(self.device)
        finite_tensor(target, 'observed', 3)
        if target.shape != self.data.traces.shape: raise ValueError('observed must match canonical data shape')
        self.observed = target.detach().clone()
        self.critic = LegacyCritic(self.shots_per_batch, data.sampling.nt, data.traces.shape[1]).to(self.device)
        self.critic_optimizer = _CountedAdam([{'name':'critic', 'params':list(self.critic.parameters()),
                                             'lr':config.critic_lr, 'betas':config.adam_betas}])
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer,
            config.critic_scheduler_step_size, gamma=config.scheduler_gamma)
        self.critic_scaler = torch.cuda.amp.GradScaler(enabled=config.amp and self.device.type == 'cuda')
        self.counters.update(critic_updates=0, critic_skipped=0)
        self.amp_enabled = config.amp and self.device.type == 'cuda'

    def _indices(self, batch_indices):
        total = self.data.traces.shape[0]
        if isinstance(batch_indices, slice): indices = list(range(total))[batch_indices]
        elif isinstance(batch_indices, torch.Tensor):
            if batch_indices.ndim != 1 or batch_indices.dtype != torch.long:
                raise ValueError('batch indices must be one-dimensional integer indices')
            indices = batch_indices.tolist()
        else:
            try: indices = list(batch_indices)
            except TypeError as error: raise ValueError('batch indices must be a sequence') from error
        if (len(indices) != self.shots_per_batch or any(type(i) is not int or not 0<=i<total for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError('batch must contain the fixed critic shot-group size of distinct valid indices')
        return indices

    def step(self, batch_indices):
        indices = self._indices(batch_indices)
        before = self.counters.copy()
        self.physical_optimizer.zero_grad(set_to_none=True)
        # Physics is float32 even when the critic uses CUDA autocast.
        with torch.cuda.amp.autocast(enabled=False):
            predicted = to_critic_layout(legacy_normalize(self._predict(indices)))
            if self.config.noise: predicted = predicted + self.parameters.sigma * torch.randn_like(predicted)
        observed = to_critic_layout(self.observed[indices])
        losses=[]
        for _ in range(self.config.critic_cycles):
            self.critic_optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                fake, real = self.critic(predicted.detach()), self.critic(observed)
                penalty = gradient_penalty(self.critic, observed, predicted)
                loss = (fake.sum()-real.sum() + self.config.gradient_penalty_weight*penalty
                        + self.config.critic_drift*real.square().mean())
            losses.append(loss.detach())
            self.critic_scaler.scale(loss).backward()
            self.critic_scaler.unscale_(self.critic_optimizer)
            nn.utils.clip_grad_value_(self.critic.parameters(), self.config.critic_clip)
            self._update(self.critic_optimizer, self.critic_scaler, 'critic')
        self.critic_optimizer.zero_grad(set_to_none=True)
        with _frozen(self.critic):
            with torch.cuda.amp.autocast(enabled=self.amp_enabled): generator_loss = -self.critic(predicted).sum()
            self.physical_scaler.scale(generator_loss).backward()
        self.physical_scaler.unscale_(self.physical_optimizer)
        nn.utils.clip_grad_value_([self.parameters.velocity], self.config.velocity_clip)
        self._update(self.physical_optimizer, self.physical_scaler, 'physical')
        values = torch.stack([generator_loss.detach(), torch.stack(losses).mean(), self.parameters.sigma.detach().reshape(())]).cpu().tolist()
        if not all(math.isfinite(value) for value in values): raise ValueError('nonfinite adversarial metrics')
        metrics=dict(zip(('generator_loss','critic_loss','sigma'), values))
        metrics.update({k:self.counters[k]-before[k] for k in before})
        return metrics

    def finish_epoch(self):
        super().finish_epoch()
        if self.counters['critic_updates']: self.critic_scheduler.step()

    def state_dict(self):
        state=super().state_dict()
        state.update(_portable(dict(observed=self.observed, critic=self.critic.state_dict(), critic_optimizer=self.critic_optimizer.state_dict(),
            critic_scheduler=self.critic_scheduler.state_dict(), critic_scaler=self.critic_scaler.state_dict())))
        return state

    def load_state_dict(self, state):
        self._validate_state(state)
        try:
            observed = state['observed']
            finite_tensor(observed, 'state observed', 3)
            if observed.shape != self.observed.shape: raise ValueError('incompatible observed state shape')
            self.critic.load_state_dict(state['critic'])
            self.critic_optimizer.load_state_dict(state['critic_optimizer'])
            self.critic_scheduler.load_state_dict(state['critic_scheduler'])
            self.critic_scaler.load_state_dict(state['critic_scaler'])
        except (KeyError, TypeError, RuntimeError) as error:
            raise ValueError('invalid critic strategy state') from error
        super().load_state_dict(state)
        self.observed = observed.to(self.device).detach().clone()
