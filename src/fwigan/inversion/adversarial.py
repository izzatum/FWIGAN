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


def _number(value, name, minimum=0., strict=False, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (value <= minimum if strict else value < minimum)
            or (maximum is not None and value > maximum)):
        raise ValueError(f'invalid restored {name}')


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'invalid restored {name}')


def _validate_scaler(state):
    if not isinstance(state, dict): raise ValueError('invalid restored scaler state')
    if not state: return
    keys = {'scale', 'growth_factor', 'backoff_factor', 'growth_interval', '_growth_tracker'}
    if set(state) != keys: raise ValueError('invalid restored scaler fields')
    _number(state['scale'], 'scaler scale', strict=True, maximum=torch.finfo(torch.float32).max)
    if torch.tensor(state['scale'], dtype=torch.float32).item() == 0:
        raise ValueError('restored scaler scale underflows float32')
    _number(state['growth_factor'], 'scaler growth factor', minimum=1., strict=True)
    _number(state['backoff_factor'], 'scaler backoff factor', strict=True)
    if state['backoff_factor'] >= 1.: raise ValueError('invalid restored scaler backoff factor')
    _integer(state['growth_interval'], 'scaler growth interval', minimum=1)
    _integer(state['_growth_tracker'], 'scaler growth tracker')
    if state['_growth_tracker'] >= state['growth_interval']:
        raise ValueError('invalid restored scaler growth tracker')


class _PortableGradScaler(torch.cuda.amp.GradScaler):
    """Keep enabled scaler history while runtime AMP is disabled on CPU."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_history = {}
        self._fresh_history = super().state_dict()

    def state_dict(self):
        if self.is_enabled(): return super().state_dict()
        return copy.deepcopy(self._saved_history)

    def load_state_dict(self, state):
        _validate_scaler(state)
        self._saved_history = copy.deepcopy(state)
        if self.is_enabled():
            # Empty CPU history begins fresh calibration on a CUDA runtime.
            super().load_state_dict(state if state else self._fresh_history)


def _validate_adam(state, optimizer):
    if not isinstance(state, dict) or set(state) != {'state', 'param_groups'}:
        raise ValueError('invalid restored Adam state')
    groups, slots = state['param_groups'], state['state']
    if not isinstance(groups, list) or not isinstance(slots, dict) or len(groups) != len(optimizer.param_groups):
        raise ValueError('incompatible restored Adam groups')
    parameters = {}
    for saved, live in zip(groups, optimizer.param_groups):
        if not isinstance(saved, dict) or set(saved) != set(live) or saved['name'] != live['name']:
            raise ValueError('incompatible named optimizer groups')
        ids = saved['params']
        if not isinstance(ids, list) or len(ids) != len(live['params']):
            raise ValueError('incompatible restored Adam parameters')
        for key, parameter in zip(ids, live['params']):
            _integer(key, 'Adam parameter identifier')
            if key in parameters: raise ValueError('duplicate restored Adam parameter identifier')
            parameters[key] = parameter
        for name in ('lr', 'initial_lr'):
            _number(saved[name], f'Adam {name}', strict=True)
        _number(saved['eps'], 'Adam epsilon', strict=True)
        _number(saved['weight_decay'], 'Adam weight decay')
        betas = saved['betas']
        if not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise ValueError('invalid restored Adam betas')
        for beta in betas:
            _number(beta, 'Adam beta')
            if beta >= 1.: raise ValueError('invalid restored Adam beta')
        # These runtime options are not configurable by this toolkit.
        for name in ('amsgrad', 'maximize', 'foreach', 'capturable', 'differentiable', 'fused'):
            if type(saved[name]) is not type(live[name]) or saved[name] != live[name]:
                raise ValueError(f'incompatible restored Adam option {name}')
    if any(type(key) is not int or key not in parameters for key in slots):
        raise ValueError('invalid restored Adam state identifier')
    for key, slot in slots.items():
        if not isinstance(slot, dict) or set(slot) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise ValueError('invalid restored Adam tensor fields')
        step = slot['step']
        if (not torch.is_tensor(step) or step.shape != torch.Size([]) or step.dtype != torch.float32
                or not torch.isfinite(step).item() or step.item() < 0 or step.item() != int(step.item())):
            raise ValueError('invalid restored Adam step')
        for name in ('exp_avg', 'exp_avg_sq'):
            value, parameter = slot[name], parameters[key]
            if (not torch.is_tensor(value) or value.shape != parameter.shape or value.dtype != parameter.dtype
                    or not torch.isfinite(value).all().item()):
                raise ValueError(f'invalid restored Adam {name}')
            if name == 'exp_avg_sq' and (value < 0).any().item():
                raise ValueError('negative restored Adam second moment')


def _validate_scheduler(state, scheduler):
    if not isinstance(state, dict) or set(state) != set(scheduler.state_dict()):
        raise ValueError('invalid restored scheduler fields')
    _integer(state['step_size'], 'scheduler period', minimum=1)
    _number(state['gamma'], 'scheduler gamma', strict=True, maximum=1.)
    _integer(state['last_epoch'], 'scheduler epoch')
    _integer(state['_step_count'], 'scheduler step count', minimum=1)
    for name in ('verbose', '_get_lr_called_within_step'):
        if type(state[name]) is not bool: raise ValueError(f'invalid restored scheduler {name}')
    for name in ('base_lrs', '_last_lr'):
        values = state[name]
        if not isinstance(values, list) or len(values) != len(scheduler.optimizer.param_groups):
            raise ValueError('invalid restored scheduler learning rates')
        for value in values: _number(value, 'scheduler learning rate', strict=True)


def _validate_critic(state, critic):
    expected = critic.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError('invalid restored critic fields')
    for key, value in state.items():
        if (not torch.is_tensor(value) or value.shape != expected[key].shape or value.dtype != expected[key].dtype
                or not torch.isfinite(value).all().item()):
            raise ValueError('invalid restored critic tensor')


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
        self.physical_scaler = _PortableGradScaler(enabled=config.amp and self.device.type == 'cuda')
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

    def _check_objective(self, loss, optimizer):
        if not torch.isfinite(loss.detach()).all().item():
            optimizer.zero_grad(set_to_none=True)
            raise ValueError('nonfinite adversarial objective')

    def _check_unscaled_gradients(self, optimizer):
        if self.amp_enabled:
            # GradScaler already records overflow during unscale_ and handles skips.
            return
        finite = [torch.isfinite(p.grad).all() for group in optimizer.param_groups
                  for p in group['params'] if p.grad is not None]
        if finite and not torch.stack(finite).all().item():
            optimizer.zero_grad(set_to_none=True)
            raise ValueError('nonfinite adversarial unscaled gradients')

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
        try:
            for name in ('physical', 'critic'):
                optimizer = getattr(self, f'{name}_optimizer', None)
                if optimizer is None: continue
                _validate_adam(state[f'{name}_optimizer'], optimizer)
                _validate_scheduler(state[f'{name}_scheduler'], getattr(self, f'{name}_scheduler'))
                _validate_scaler(state[f'{name}_scaler'])
            if hasattr(self, 'critic'):
                _validate_critic(state['critic'], self.critic)
                finite_tensor(state['observed'], 'state observed', 3)
                if state['observed'].shape != self.observed.shape:
                    raise ValueError('incompatible observed state shape')
        except (KeyError, TypeError, RuntimeError, IndexError) as error:
            raise ValueError('invalid strategy numerical state') from error
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
        self.critic_scaler = _PortableGradScaler(enabled=config.amp and self.device.type == 'cuda')
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
            self._check_objective(loss, self.critic_optimizer)
            losses.append(loss.detach())
            self.critic_scaler.scale(loss).backward()
            self.critic_scaler.unscale_(self.critic_optimizer)
            self._check_unscaled_gradients(self.critic_optimizer)
            nn.utils.clip_grad_value_(self.critic.parameters(), self.config.critic_clip)
            self._update(self.critic_optimizer, self.critic_scaler, 'critic')
        self.critic_optimizer.zero_grad(set_to_none=True)
        with _frozen(self.critic):
            with torch.cuda.amp.autocast(enabled=self.amp_enabled): generator_loss = -self.critic(predicted).sum()
            self._check_objective(generator_loss, self.physical_optimizer)
            self.physical_scaler.scale(generator_loss).backward()
        self.physical_scaler.unscale_(self.physical_optimizer)
        self._check_unscaled_gradients(self.physical_optimizer)
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
