"""Inversion strategies share canonical physical and observation contracts."""
from .critic import LegacyCritic
from .adversarial import FWIGANStrategy
from .fwi import FWIStrategy


def build_strategy(parameters, forward, data, config):
    if config.method == 'fwigan': return FWIGANStrategy(parameters, forward, data, config)
    if config.method == 'fwi': return FWIStrategy(parameters, forward, data, config)
    raise ValueError('unsupported inversion method')


__all__ = ['LegacyCritic', 'FWIGANStrategy', 'FWIStrategy', 'build_strategy']
