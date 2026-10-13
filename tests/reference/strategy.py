"""Execute the frozen notebook training cell without importing production loops."""
import torch
from torch import nn
import numpy as np
import deepwave as dw
from . import FROZEN


class QuietProgress:
    def __init__(self, iterable): self.iterable = iterable
    def __iter__(self): return iter(self.iterable)
    def set_postfix(self, **kwargs): pass


def make_reference(data, config):
    ns = dict(torch=torch, nn=nn, np=np, dw=dw, tqdm=QuietProgress,
              BATCHES=config.batches, CRITIC_CYCLE=config.critic_cycles, DEVICE='cpu', epoch=0)
    for index in ('8', '10', '12'):
        exec(FROZEN['notebooks']['notebooks/FWIGAN.ipynb'][index]['source'], ns)
    source = data.source if data.initial_source is None else data.initial_source
    generator = ns['Generator'](data.initial_velocity.clone(), torch.tensor(config.sigma_init),
        source.permute(2,0,1).contiguous().clone(), config.batch_size,
        data.grid.spacing, data.sampling.dt, config.noise)
    critic = ns['Critic'](config.batch_size, data.sampling.nt, data.traces.shape[1])
    groups = [{'params': [generator.model], 'lr': config.velocity_lr},
              {'params': [generator.source], 'lr': config.source_lr}]
    if config.noise: groups.append({'params': [generator.sigma], 'lr': config.sigma_lr})
    go = torch.optim.Adam(groups, betas=config.adam_betas)
    co = torch.optim.Adam(critic.parameters(), lr=config.critic_lr, betas=config.adam_betas)
    gs = torch.optim.lr_scheduler.StepLR(go, config.physical_scheduler_step_size, gamma=config.scheduler_gamma)
    cs = torch.optim.lr_scheduler.StepLR(co, config.critic_scheduler_step_size, gamma=config.scheduler_gamma)
    sg, sc = torch.cuda.amp.GradScaler(enabled=False), torch.cuda.amp.GradScaler(enabled=False)
    def run(observed, all_batches=False):
        gl, cl = [], []
        ns['train_fn'](0, observed, generator, critic, go, co, gs, cs, gl, cl, sg, sc,
                      generator.source, data.acquisition.sources, data.acquisition.receivers)
        return (gl, cl) if all_batches else (gl[0], cl[0])
    return generator, critic, go, co, run
