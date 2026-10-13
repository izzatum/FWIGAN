"""Physical quantities are module state, including frozen source and sigma."""
import torch
from torch import nn
from .domain import finite_tensor


class InversionParameters(nn.Module):
    def __init__(self, velocity, source, sigma, train_source=True, train_sigma=False):
        super().__init__()
        finite_tensor(velocity, 'velocity', 2)
        finite_tensor(source, 'source', 3)
        finite_tensor(sigma, 'sigma')
        if (velocity <= 0).any() or sigma.numel() != 1 or sigma.item() < 0:
            raise ValueError('velocity must be positive and sigma a nonnegative scalar')
        self.velocity = nn.Parameter(velocity.detach().clone())
        for name, value, enabled in (('source', source, train_source), ('sigma', sigma, train_sigma)):
            if enabled:
                self.register_parameter(name, nn.Parameter(value.detach().clone()))
            else:
                self.register_buffer(name, value.detach().clone())

    def optimizer_groups(self, config):
        """Call after moving this module to its training device."""
        groups = []
        for name in ('velocity', 'source', 'sigma'):
            value = getattr(self, name)
            enabled = isinstance(value, nn.Parameter) and value.requires_grad
            if name != 'velocity' and enabled != getattr(config, f'train_{name}'):
                raise ValueError(f'{name} training state differs between module and configuration')
            if enabled:
                groups.append({'name': name, 'params': [value], 'lr': getattr(config, f'{name}_lr'),
                               'betas': config.adam_betas})
        return groups
