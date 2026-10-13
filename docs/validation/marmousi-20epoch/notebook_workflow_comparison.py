"""Bounded, external execution of current and frozen Marmousi notebooks.

Only notebook copies and run artifacts are written. This is not a benchmark
acceptance test: original and current training may follow different AMP paths.
"""
import argparse
import copy
import gc
import hashlib
from importlib.metadata import version as package_version
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


BASELINE = 'c7ec47ed89ba1380ad7a02bbc4526b93675e76f3'
MILESTONES = {0, 2, 5, 10, 15, 20}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def cell_source(cell):
    return ''.join(cell['source'])


def code_cell(source):
    return dict(cell_type='code', execution_count=None, metadata={}, outputs=[], source=source)


def copied_cell(cell, source=None):
    result = copy.deepcopy(cell)
    if source is not None:
        result['source'] = source
    result['execution_count'] = None
    result['outputs'] = []
    return result


def prelude(repo, output, name, deadline):
    return f'''import sys, time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})
import notebook_workflow_comparison as wf
WF_OUTPUT = Path({str(output)!r})
WF_NAME = {name!r}
WF_DEADLINE = {int(deadline)!r}
WF_REPO = Path({str(repo)!r})
'''


def build_new_notebook(notebook, name, repo, output, epochs, deadline):
    """Keep the public create/evaluate/plot cells and extend their short budget."""
    clean = name == 'FWIGAN'
    original = copy.deepcopy(notebook)
    first = cell_source(original['cells'][1])
    config = Path(output) / ('new-clean.yaml' if clean else 'new-noisy.yaml')
    first = first.replace('CONFIG = ROOT / "configs" / "tiny-fwigan' + ('' if clean else '-noisy') + '.yaml"',
                          f'CONFIG = Path({str(config)!r})')
    first = first.replace('data = prepare_data(workflow)',
                          f'from fwigan.data import load_observations\ndata = load_observations({str(Path(output) / "shared_observations.pt")!r})')
    first = first.replace('ROOT / "runs" / "notebooks"', f'WF_OUTPUT / "{name}-new-runs"')
    if 'WF_OUTPUT' not in first or 'shared_observations.pt' not in first:
        raise ValueError('current notebook first cell changed unexpectedly')
    first += '\nwf.new_epoch(run, 0, WF_OUTPUT, WF_NAME, WF_DEADLINE)\n'
    train = '''for epoch in range(1, min(2, workflow.experiment.epochs) + 1):
    wf.check_deadline(WF_DEADLINE)
    run.run(epochs=epoch)
    wf.new_epoch(run, epoch, WF_OUTPUT, WF_NAME, WF_DEADLINE)
'''
    resume = '''split_id, split_fingerprint = run.run_id, run.data_fingerprint
del run
run = Experiment.resume(next((WF_OUTPUT / (WF_NAME + "-new-runs")).glob("fwigan-*/checkpoint.pt")), device=workflow.experiment.device)
assert run.run_id == split_id and run.data_fingerprint == split_fingerprint
for epoch in range(3, workflow.experiment.epochs + 1):
    wf.check_deadline(WF_DEADLINE)
    run.run(epochs=epoch)
    wf.new_epoch(run, epoch, WF_OUTPUT, WF_NAME, WF_DEADLINE)
result = torch.load(run.export_final(), map_location="cpu")
figures = plot_results(data, result, output_dir=WF_OUTPUT / (WF_NAME + "-new-figures"))
wf.close_figures()
'''
    original['cells'] = [code_cell(prelude(repo, output, name, deadline)),
                         copied_cell(original['cells'][1], first),
                         copied_cell(original['cells'][3], train),
                         copied_cell(original['cells'][5], resume)]
    original['metadata']['workflow_comparison'] = dict(kind='new', source=name,
        substitutions=['absolute observations YAML', 'shared raw observations',
                       '2-epoch checkpoint/resume', f'{epochs} complete epochs', 'progress and output sink'])
    return original


def build_old_notebook(notebook, name, repo, output, epochs, deadline, smoke=False):
    """Retain frozen numeric definitions and train_fn; replace Colab I/O only."""
    original = copy.deepcopy(notebook)
    selected = [5, 8, 10, 12, 14, 16, 22, 23, 24]
    cells = {index: cell_source(original['cells'][index]) for index in selected}
    cells[14] = cells[14].replace("'/models/marmousi_3.bin'", repr(str(Path(repo) / 'models/marmousi_3.bin')))
    cells[14] = cells[14].replace("'/models/marmousi_30.bin'", repr(str(Path(repo) / 'models/marmousi_30.bin')))
    cells[14] = cells[14].replace('EPOCHS = 401', f'EPOCHS = {int(epochs)}')
    cells[14] = cells[14].replace('DEVICE = torch.device(0 if torch.cuda.is_available() else \'cpu\')',
                                  "DEVICE = torch.device('cuda')")
    suffix = 'clean' if name == 'FWIGAN' else 'noisy'
    for key, stem in [('CHECKPOINT_GEN', 'generator'), ('CHECKPOINT_CRITIC', 'critic')]:
        lines = cells[14].splitlines()
        cells[14] = '\n'.join(f'{key} = {str(Path(output) / ("old-" + suffix + "-" + stem + ".pt"))!r}'
                               if line.startswith(key + ' = ') else line for line in lines) + '\n'
    raw_line = 'clean_data = generate_data(TRUTH, source, xs, xr, DX_TRUTH, DT, DEVICE)'
    replacement = (f'shared = wf.load_shared({str(Path(output) / "shared_observations.pt")!r})\n'
                   'clean_data = data_normalization(shared.traces.permute(2, 0, 1).to(DEVICE)).permute(1, 0, 2).unsqueeze(0)')
    if raw_line not in cells[16]:
        raise ValueError('frozen notebook data-generation cell changed unexpectedly')
    cells[16] = cells[16].replace(raw_line, replacement)
    if smoke:
        cells[14] = cells[14].replace(
            'TRUTH = load_true_model(' + repr(str(Path(repo) / 'models/marmousi_3.bin')) + ')',
            f'TRUTH = wf.load_shared({str(Path(output) / "shared_observations.pt")!r}).reference_velocity')
        cells[14] = cells[14].replace(
            'MODEL_INIT = load_true_model(' + repr(str(Path(repo) / 'models/marmousi_30.bin')) + ')',
            f'MODEL_INIT = wf.load_shared({str(Path(output) / "shared_observations.pt")!r}).initial_velocity')
        settings = {'FREQ = 7': 'FREQ = 25', 'DX_TRUTH = 3.0': 'DX_TRUTH = 30.0',
                    'NT = int(6 / DT)': 'NT = 128', 'SHOTS = 30': 'SHOTS = 2',
                    'BATCHES = 6': 'BATCHES = 1',
                    'CRITIC_CYCLE = SHOTS_PER_BATCH + 1': 'CRITIC_CYCLE = 1',
                    "DEVICE = torch.device('cuda')": "DEVICE = torch.device('cpu')"}
        for before, after in settings.items():
            cells[14] = cells[14].replace(before, after)
        first = cells[16].index('# Create arrays containing the source and receiver locations')
        last = cells[16].index('# Create true source amplitudes')
        cells[16] = (cells[16][:first]
            + f'shared = wf.load_shared({str(Path(output) / "shared_observations.pt")!r})\n'
              'xs = shared.acquisition.sources.to(DEVICE)\n'
              'xr = shared.acquisition.receivers.to(DEVICE)\n\n'
            + cells[16][last:])
        # On CUDA the constructor's .to(device) copies this CPU tensor. Mirror
        # that alias boundary for the CPU smoke so the frozen plot can use numpy.
        cells[16] = cells[16].replace('Generator(model, SIGMA, source_init,',
                                      'Generator(model, SIGMA, source_init.clone(),')
    cells[16] += '\nwf.install_old_step_counters(generator_opt, critic_opt)\n'
    cells[16] += 'wf.old_epoch(locals(), 0, WF_OUTPUT, WF_NAME, WF_DEADLINE)\n'
    anchor = '    if epoch % 10 == 0:'
    if anchor not in cells[22] or 'for epoch in range(EPOCHS):' not in cells[22]:
        raise ValueError('frozen notebook training cell changed unexpectedly')
    cells[22] = cells[22].replace(anchor,
        '    wf.old_epoch(locals(), epoch + 1, WF_OUTPUT, WF_NAME, WF_DEADLINE)\n\n' + anchor, 1)
    cells[22] = cells[22].replace('for epoch in range(EPOCHS):',
        'for epoch in range(EPOCHS):\n    wf.check_deadline(WF_DEADLINE)', 1)
    cells[22] = cells[22].replace('crit_loss = []', 'crit_loss = []\n_wf_total_train_seconds = 0.0', 1)
    cells[22] = cells[22].replace('    train_fn(', '    _wf_epoch_started = wf.sync_time()\n    train_fn(', 1)
    cells[22] = cells[22].replace('    wf.old_epoch(locals(), epoch + 1, WF_OUTPUT, WF_NAME, WF_DEADLINE)',
        '    _wf_total_train_seconds += wf.sync_time() - _wf_epoch_started\n'
        '    wf.old_epoch(locals(), epoch + 1, WF_OUTPUT, WF_NAME, WF_DEADLINE)', 1)
    cells[22] += '\nwf.finish_old(locals(), WF_OUTPUT, WF_NAME)\n'
    cells[24] += '\nwf.save_open_figures(WF_OUTPUT / (WF_NAME + "-old-figures"))\n'
    original['cells'] = [code_cell(prelude(repo, output, name, deadline))] + [copied_cell(original['cells'][i], cells[i]) for i in selected]
    original['metadata']['workflow_comparison'] = dict(kind='frozen', source=name, baseline=BASELINE,
        preserved_numeric_cells=[8, 10, 12], substitutions=['Colab/install cells omitted',
        'absolute model/checkpoint paths', 'shared raw observations before mandatory noise draw',
        f'{epochs} complete epochs', 'progress and output sink'], tiny_cpu_smoke=smoke)
    return original


class QuietProgress:
    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, **_):
        pass


def check_deadline(deadline):
    if time.time() >= float(deadline):
        raise TimeoutError('workflow wall deadline reached before next epoch')


def sync_time():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _finite(value):
    if isinstance(value, dict):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def tensor_sha(tensor):
    h = hashlib.sha256()
    value = tensor.detach().cpu().contiguous()
    h.update(str(value.dtype).encode())
    h.update(str(tuple(value.shape)).encode())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def module_sha(module):
    h = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        h.update(name.encode())
        h.update(tensor_sha(tensor).encode())
    return h.hexdigest()


def rng_hashes():
    import numpy as np
    from fwigan.checkpoints import rng_snapshot
    state = rng_snapshot()
    return dict(python=hashlib.sha256(repr(state['python']).encode()).hexdigest(),
                numpy=hashlib.sha256(repr(state['numpy'][0:1] + state['numpy'][2:]).encode()
                                     + state['numpy'][1].tobytes()).hexdigest(),
                torch=tensor_sha(state['torch']),
                cuda=[tensor_sha(value) for value in state['cuda']],
                cudnn_deterministic=state['cudnn_deterministic'],
                cudnn_benchmark=state['cudnn_benchmark'])


def load_shared(path):
    from fwigan.data import load_observations
    return load_observations(path)


def _metrics(predicted, data, velocity, source, sigma):
    import torch
    from fwigan.data import legacy_normalize
    observed = data.traces
    if predicted.shape != observed.shape:
        raise ValueError('evaluation prediction shape changed')
    if not torch.isfinite(predicted).all():
        raise ValueError('nonfinite evaluation prediction')
    raw = (predicted - observed).square().mean().item()
    normalized = (legacy_normalize(predicted) - legacy_normalize(observed)).square().mean().item()
    reference = data.reference_velocity
    model_rmse = None if reference is None else (velocity.detach().cpu() - reference).square().mean().sqrt().item()
    learned = source.detach().cpu().reshape(-1)
    # data.initial_source is the unscaled 10 Hz initializer; its true 7 Hz
    # reference is data.source / 0.01 for this exact 3 m -> 30 m fixture.
    true_coarse = data.source.detach().cpu().reshape(-1)
    true_unscaled = true_coarse * 100.
    corr = torch.nn.functional.cosine_similarity(learned, true_unscaled, dim=0).item()
    return dict(raw_residual_mse=raw, normalized_clean_mse=normalized,
                model_rmse=model_rmse, source_correlation=corr,
                source_norm_ratio_unscaled=(learned.norm() / true_unscaled.norm()).item(),
                source_norm_ratio_coarse=(learned.norm() / true_coarse.norm()).item(),
                sigma_signed=sigma.detach().cpu().item(), sigma_magnitude=abs(sigma.detach().cpu().item()),
                normalized_noise_std_reference=0.05)


def _case_path(output, name, kind):
    return Path(output) / f'{name}-{kind}-metrics.json'


def _record(output, name, kind, row):
    path = _case_path(output, name, kind)
    existing = json.loads(path.read_text()) if path.exists() else dict(name=name, kind=kind, rows=[])
    existing['rows'].append(_finite(row))
    atomic_json(path, existing)
    status_file = Path(os.environ.get('WF_STATUS_FILE', str(Path(output) / 'progress.json')))
    atomic_json(status_file, dict(status='running', case=name + '-' + kind,
                                  epoch=row['epoch'], updated_utc=time.time(),
                                  deadline_utc=os.environ.get('WF_DEADLINE')))


def _gpu_memory():
    import torch
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    return dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())


def new_epoch(run, epoch, output, name, deadline):
    import torch
    from fwigan.data import legacy_normalize
    check_deadline(deadline)
    row = dict(epoch=epoch, position=vars(run.position).copy(), run_id=run.run_id,
               data_fingerprint=run.data_fingerprint, config_fingerprint=run.config_fingerprint,
               elapsed_train_seconds=run.elapsed_seconds, solver_counts=vars(run.solver_counts).copy(),
               counters=copy.deepcopy(run.strategy.counters),
               recent_group_metrics=copy.deepcopy(run.metrics[-run.config.batches:]),
               memory=_gpu_memory())
    if epoch == 0:
        from fwigan.data import to_critic_layout
        clean_target = to_critic_layout(legacy_normalize(run.strategy.data.traces))[0].permute(0, 2, 1) if run.config.method == 'fwigan' else None
        row['initial'] = dict(rng=rng_hashes(), physical={key: tensor_sha(value)
                              for key, value in run.parameters.state_dict().items()},
                              critic=module_sha(run.strategy.critic) if run.config.method == 'fwigan' else None,
                              observed=tensor_sha(run.strategy.observed) if run.config.method == 'fwigan' else None,
                              noise_std=float((run.strategy.observed - clean_target).std()) if run.config.method == 'fwigan' else None)
    if epoch in MILESTONES or epoch == run.config.epochs:
        evaluated = run.evaluate()
        predicted = run._last_prediction
        row['evaluation'] = dict(evaluated, **_metrics(predicted, run.data,
            run.parameters.velocity, run.parameters.source, run.parameters.sigma))
    _record(output, name, 'new', row)
    check_deadline(deadline)


def install_old_step_counters(physical_optimizer, critic_optimizer):
    for optimizer in (physical_optimizer, critic_optimizer):
        optimizer._wf_steps = 0
        original_step = optimizer.step

        def counted_step(*args, _optimizer=optimizer, _original=original_step, **kwargs):
            result = _original(*args, **kwargs)
            _optimizer._wf_steps += 1
            return result

        optimizer.step = counted_step


def _old_evaluate(context, data):
    import torch
    from fwigan.domain import Acquisition
    from fwigan.physics import LegacyDeepwave
    generator = context['generator']
    source = generator.source.permute(1, 2, 0).contiguous()
    device = generator.model.device
    solver = LegacyDeepwave()
    chunks = []
    with torch.no_grad():
        for first in range(0, data.traces.shape[0], 5):
            last = min(first + 5, data.traces.shape[0])
            subset = Acquisition(data.acquisition.sources[first:last],
                                 data.acquisition.receivers[first:last]).to(device)
            chunks.append(solver.predict(generator.model, source, subset,
                                         data.grid, data.sampling).detach().cpu())
    return torch.cat(chunks, dim=0)


def old_epoch(context, epoch, output, name, deadline):
    import torch
    check_deadline(deadline)
    generator = context['generator']
    physical = context['generator_opt']
    critic = context['critic_opt']
    data = context['shared']
    row = dict(epoch=epoch, actual_updates=dict(physical=physical._wf_steps,
                                              critic=critic._wf_steps),
               elapsed_train_seconds=float(context.get('_wf_total_train_seconds', 0.0)),
               skipped_updates=dict(physical=epoch * context['BATCHES'] - physical._wf_steps,
                                    critic=epoch * context['BATCHES'] * context['CRITIC_CYCLE'] - critic._wf_steps),
               losses=dict(generator=list(context.get('gen_loss', [])),
                           critic=list(context.get('crit_loss', []))),
               scales=dict(physical=float(context['scaler_generator'].get_scale()),
                           critic=float(context['scaler_critic'].get_scale())),
               learning_rates=dict(physical=[group['lr'] for group in physical.param_groups],
                                   critic=[group['lr'] for group in critic.param_groups]),
               schedulers=dict(physical=context['generator_scheduler'].last_epoch,
                               critic=context['critic_scheduler'].last_epoch),
               memory=_gpu_memory())
    if epoch == 0:
        target = context['noisy_data'] if context['NOISE'] else context['clean_data']
        row['initial'] = dict(rng=rng_hashes(), observed=tensor_sha(target[0].permute(0, 2, 1)),
                              critic=module_sha(context['critic']),
                              noise_std=float((target - context['clean_data']).std()),
                              physical=dict(velocity=tensor_sha(generator.model),
                                            source=tensor_sha(generator.source.reshape(1, 1, -1)),
                                            sigma=tensor_sha(generator.sigma)),
                              frozen_generator_state_keys=list(generator.state_dict()))
    if epoch in MILESTONES or epoch == context['EPOCHS']:
        predicted = _old_evaluate(context, data)
        row['evaluation'] = _metrics(predicted, data, generator.model,
                                     generator.source, generator.sigma)
        if epoch == context['EPOCHS']:
            torch.save(dict(epoch=epoch, velocity=generator.model.detach().cpu(),
                            source=generator.source.detach().cpu(), sigma=generator.sigma.detach().cpu(),
                            predicted=predicted), Path(output) / (name + '-old-final.pt'))
    # Old generator.state_dict() is empty: these supplemental tensors make
    # interrupted old runs inspectable without claiming notebook resumability.
    torch.save(dict(epoch=epoch, velocity=generator.model.detach().cpu(),
                    source=generator.source.detach().cpu(), sigma=generator.sigma.detach().cpu()),
               Path(output) / (name + '-old-physical-latest.pt'))
    _record(output, name, 'old', row)
    check_deadline(deadline)


def finish_old(context, output, name):
    # The frozen cell still saves its original optimizer-only checkpoints at
    # epoch zero. The supplemental physical export is written by old_epoch.
    save_open_figures(Path(output) / (name + '-old-figures'))


def save_open_figures(destination):
    import matplotlib.pyplot as plt
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for number in plt.get_fignums():
        figure = plt.figure(number)
        figure.savefig(destination / f'original-figure-{number:02d}.png', dpi=120)
    plt.close('all')


def close_figures():
    import matplotlib.pyplot as plt
    plt.close('all')


def _prepare_shared(repo, output, device, deadline):
    import torch
    from fwigan.cli import load_config, prepare_data
    from fwigan.data import save_observations
    from fwigan.domain import Acquisition
    from fwigan.physics import LegacyDeepwave

    class BoundedForward:
        def __init__(self):
            self.mode = 'all_shots'
            self.probe = None

        def predict(self, velocity, source, acquisition, grid, sampling):
            solver = LegacyDeepwave()
            try:
                check_deadline(deadline)
                return solver.predict(velocity, source, acquisition, grid, sampling)
            except torch.cuda.OutOfMemoryError:
                if device != 'cuda':
                    raise
                torch.cuda.empty_cache()
                self.mode = 'shot_blocks_after_all_shot_oom'
            for size in (5, 1):
                try:
                    blocks = []
                    for first in range(0, acquisition.sources.shape[0], size):
                        check_deadline(deadline)
                        last = min(first + size, acquisition.sources.shape[0])
                        subset = Acquisition(acquisition.sources[first:last], acquisition.receivers[first:last])
                        blocks.append(solver.predict(velocity, source, subset, grid, sampling).detach().cpu())
                    result = torch.cat(blocks, dim=0)
                    # Recreate the native [time, shot, receiver] backing stride;
                    # legacy randn_like noise follows this storage order.
                    result = result.permute(2, 0, 1).contiguous().permute(1, 2, 0)
                    self.mode += f'_size_{size}'
                    if size == 5:
                        first_five = result[:5]
                        singles = []
                        for shot in range(5):
                            one = Acquisition(acquisition.sources[shot:shot+1],
                                              acquisition.receivers[shot:shot+1])
                            singles.append(solver.predict(velocity, source, one, grid, sampling).detach().cpu())
                        diff = (first_five - torch.cat(singles, 0)).abs()
                        self.probe = dict(max_absolute=float(diff.max()),
                                          mean_absolute=float(diff.mean()), shots=5)
                    return result
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if size == 1:
                        raise

    check_deadline(deadline)
    config = load_config(Path(repo) / 'configs/marmousi-fwigan.yaml', device=device)
    forward = BoundedForward()
    started = time.perf_counter()
    data = prepare_data(config, forward=forward)
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    path = Path(output) / 'shared_observations.pt'
    save_observations(path, data)
    if not torch.isfinite(data.traces).all() or data.traces.shape != (30, 60, 2000):
        raise ValueError('shared raw observations have nonfinite values or unexpected geometry')
    return data, dict(path=str(path), sha256=digest(path), data_fingerprint=data.fingerprint(),
                      shape=list(data.traces.shape), fine_shape=[1000, 3100],
                      coarse_shape=list(data.grid.shape), seconds=elapsed,
                      simulation_mode=forward.mode, shot_block_probe=forward.probe,
                      raw_sha256=tensor_sha(data.traces), raw_strides=list(data.traces.stride()),
                      peak_memory=_gpu_memory())


def _write_workflow_config(repo, output, name, epochs, device):
    import yaml
    source = {'FWIGAN': 'marmousi-fwigan.yaml',
              'FWIGAN_Noisy': 'marmousi-fwigan-noisy.yaml',
              'FWI': 'marmousi-fwi.yaml'}[name]
    config = yaml.safe_load((Path(repo) / 'configs' / source).read_text())
    config['experiment']['epochs'] = int(epochs)
    config['experiment']['device'] = device
    config['dataset'] = dict(kind='observations', path=str(Path(output) / 'shared_observations.pt'))
    path = Path(output) / ('new-clean.yaml' if name == 'FWIGAN' else
                           'new-noisy.yaml' if name == 'FWIGAN_Noisy' else 'new-fwi.yaml')
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def build_fwi_notebook(repo, output, epochs, deadline):
    import nbformat
    name = 'FWI'
    first = f'''from pathlib import Path
import torch
from fwigan.cli import load_config
from fwigan.data import load_observations
from fwigan.experiment import Experiment
from fwigan.plotting import plot_results
workflow = load_config({str(Path(output) / 'new-fwi.yaml')!r})
data = load_observations({str(Path(output) / 'shared_observations.pt')!r})
run = Experiment.create(workflow.experiment, data, WF_OUTPUT / "FWI-new-runs")
initial_metrics = run.evaluate()
wf.new_epoch(run, 0, WF_OUTPUT, WF_NAME, WF_DEADLINE)
'''
    train = '''for epoch in range(1, min(2, workflow.experiment.epochs) + 1):
    wf.check_deadline(WF_DEADLINE)
    run.run(epochs=epoch)
    wf.new_epoch(run, epoch, WF_OUTPUT, WF_NAME, WF_DEADLINE)
split_id, split_fingerprint = run.run_id, run.data_fingerprint
del run
run = Experiment.resume(next((WF_OUTPUT / "FWI-new-runs").glob("fwi-*/checkpoint.pt")), device=workflow.experiment.device)
assert run.run_id == split_id and run.data_fingerprint == split_fingerprint
for epoch in range(3, workflow.experiment.epochs + 1):
    wf.check_deadline(WF_DEADLINE)
    run.run(epochs=epoch)
    wf.new_epoch(run, epoch, WF_OUTPUT, WF_NAME, WF_DEADLINE)
result = torch.load(run.export_final(), map_location="cpu")
figures = plot_results(data, result, output_dir=WF_OUTPUT / "FWI-new-figures")
wf.close_figures()
'''
    return nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(prelude(repo, output, name, deadline)),
        nbformat.v4.new_code_cell(first), nbformat.v4.new_code_cell(train)],
        metadata=dict(workflow_comparison=dict(kind='additional_fwi', total_epochs=epochs,
            note='No frozen FWI notebook counterpart')))


def _execute_notebook(notebook, repo, output, file_name, deadline):
    import nbformat
    from nbclient import NotebookClient
    nb = nbformat.from_dict(notebook)
    path = Path(output) / file_name
    nbformat.write(nb, path)
    started = time.perf_counter()
    client = NotebookClient(nb, timeout=None, kernel_name='python3',
                            resources={'metadata': {'path': str(repo)}}, allow_errors=False)
    try:
        check_deadline(deadline)
        client.execute()
        return dict(status='completed', seconds=time.perf_counter() - started, notebook=str(path))
    finally:
        # NotebookClient keeps successful outputs and the first failing cell's
        # traceback on the mutable notebook object. Persist either outcome.
        nbformat.write(nb, path)


def _initial_pair(output, name):
    new_path = _case_path(output, name, 'new')
    old_path = _case_path(output, name, 'old')
    if not new_path.exists() or not old_path.exists():
        return dict(status='incomplete')
    new = json.loads(new_path.read_text())['rows'][0]['initial']
    old = json.loads(old_path.read_text())['rows'][0]['initial']
    keys = ['observed', 'critic', 'physical', 'rng']
    return dict(status='matched' if all(new[k] == old[k] for k in keys) else 'mismatch',
                fields={key: new[key] == old[key] for key in keys},
                new=new, old=old)


def _preflight_pair(new, old, repo, output, name, deadline, smoke=False):
    """Check the common seeded start before spending on complete training."""
    import yaml
    preflight = Path(output) / 'preflight' / name
    preflight.mkdir(parents=True, exist_ok=True)
    shared = preflight / 'shared_observations.pt'
    shared.symlink_to(Path(output) / 'shared_observations.pt')
    config_name = 'new-clean.yaml' if name == 'FWIGAN' else 'new-noisy.yaml'
    config = yaml.safe_load((Path(output) / config_name).read_text())
    config['dataset']['path'] = str(shared)
    (preflight / config_name).write_text(yaml.safe_dump(config, sort_keys=False))
    current = build_new_notebook(new, name, repo, preflight, 2, deadline)
    frozen = build_old_notebook(old, name, repo, preflight, 2, deadline, smoke=smoke)
    current['cells'] = current['cells'][:2]
    frozen['cells'] = frozen['cells'][:7]
    _execute_notebook(current, repo, preflight, 'new-initial.ipynb', deadline)
    _execute_notebook(frozen, repo, preflight, 'old-initial.ipynb', deadline)
    comparison = _initial_pair(preflight, name)
    if comparison['status'] != 'matched':
        raise RuntimeError(f'{name} initialized target/critic/physical/RNG mismatch: {comparison["fields"]}')
    return comparison


def _manifest(output, max_bytes=2 * 1024**3):
    output = Path(output)
    files, omitted = [], []
    used = 0
    for path in sorted(output.rglob('*')):
        relative = str(path.relative_to(output))
        if any(part.startswith('.') or part in ('matplotlib-cache', '__pycache__')
               for part in path.relative_to(output).parts):
            continue
        if path.is_symlink():
            omitted.append(dict(path=relative, bytes=0, reason='symlink excluded'))
            continue
        if not path.is_file() or path.name in ('artifact_manifest.json', 'progress.json'):
            continue
        size = path.stat().st_size
        if path.name == 'checkpoint.pt' or path.name == 'observations.pt' or relative.endswith('-critic.pt'):
            omitted.append(dict(path=relative, bytes=size, reason='uncompressed or duplicate large checkpoint'))
            continue
        if size > 1024**3 or used + size > max_bytes:
            omitted.append(dict(path=relative, bytes=size, reason='artifact byte cap'))
            continue
        files.append(dict(path=relative, bytes=size, sha256=digest(path)))
        used += size
    return dict(schema=1, files=files, omitted=omitted, selected_bytes=used,
                max_selected_bytes=max_bytes, baseline_commit=BASELINE)


def _select_checkpoint(output):
    import gzip
    source = next((Path(output) / 'FWIGAN-new-runs').glob('fwigan-*/checkpoint.pt'), None)
    if source is None:
        return dict(status='unavailable')
    target = Path(output) / 'selected-new-clean-checkpoint.pt.gz'
    with open(source, 'rb') as original, gzip.open(target, 'wb', compresslevel=6) as compressed:
        for block in iter(lambda: original.read(1024 * 1024), b''):
            compressed.write(block)
    if target.stat().st_size > 400 * 1024**2:
        size = target.stat().st_size
        target.unlink()
        return dict(status='omitted', reason='compressed checkpoint exceeds conservative 400 MiB bundle allowance',
                    compressed_bytes=size, source_bytes=source.stat().st_size)
    return dict(status='selected', path=target.name, bytes=target.stat().st_size,
                sha256=digest(target), source_bytes=source.stat().st_size)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--original-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--deadline-utc', type=float, required=True)
    parser.add_argument('--status-file', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.epochs < 2 or args.epochs > 20:
        parser.error('epochs must be in [2,20] for this bounded comparison')
    repo = args.repo.resolve()
    original_root = args.original_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ['WF_STATUS_FILE'] = str(args.status_file.resolve())
    os.environ['WF_DEADLINE'] = str(args.deadline_utc)
    os.environ['MPLCONFIGDIR'] = str(output / 'matplotlib-cache')
    os.environ['PYTHONPATH'] = str(repo / 'src') + os.pathsep + os.environ.get('PYTHONPATH', '')
    report = dict(schema=1, status='running', epoch_horizon=args.epochs,
                  device=args.device, deadline_utc=args.deadline_utc,
                  baseline_commit=BASELINE, cases={}, initial_comparisons={})
    report_path = output / 'result.json'
    atomic_json(report_path, report)
    atomic_json(args.status_file, dict(status='starting', updated_utc=time.time()))
    try:
        import torch
        if args.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        report['environment'] = dict(torch=torch.__version__, cuda_build=torch.version.cuda,
            deepwave=package_version('deepwave'), gpu=torch.cuda.get_device_name() if args.device == 'cuda' else None,
            cudnn=torch.backends.cudnn.version(),
            driver=getattr(torch._C, '_cuda_getDriverVersion', lambda: None)() if args.device == 'cuda' else None)
        report['notebook_sources'] = {}
        for name in ('FWIGAN', 'FWIGAN_Noisy'):
            old_path = original_root / 'notebooks' / (name + '.ipynb')
            new_path = repo / 'notebooks' / (name + '.ipynb')
            for path in (old_path, new_path):
                if not path.is_file() or path.is_symlink():
                    raise ValueError(f'notebook source is absent or a symlink: {path}')
            report['notebook_sources'][name] = dict(old_transport_sha256=digest(old_path),
                new_raw_sha256=digest(new_path),
                old_numeric_cells={str(index): hashlib.sha256(cell_source(json.loads(old_path.read_text())['cells'][index]).encode()).hexdigest()
                                   for index in (8, 10, 12)})
        data, report['preparation'] = _prepare_shared(repo, output, args.device, args.deadline_utc)
        del data
        gc.collect()
        if args.device == 'cuda':
            torch.cuda.empty_cache()
            report['preparation']['parent_memory_after_release'] = dict(
                allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved())
        atomic_json(report_path, report)
        for name in ('FWIGAN', 'FWIGAN_Noisy'):
            check_deadline(args.deadline_utc)
            _write_workflow_config(repo, output, name, args.epochs, args.device)
            current = json.loads((repo / 'notebooks' / (name + '.ipynb')).read_text())
            frozen = json.loads((original_root / 'notebooks' / (name + '.ipynb')).read_text())
            new = build_new_notebook(current, name, repo, output, args.epochs, args.deadline_utc)
            old = build_old_notebook(frozen, name, repo, output, args.epochs, args.deadline_utc)
            report['initial_comparisons'][name + '_preflight'] = _preflight_pair(
                current, frozen, repo, output, name, args.deadline_utc)
            atomic_json(report_path, report)
            for kind, notebook in (('new', new), ('old', old)):
                key = name + '-' + kind
                report['current_case'] = key
                atomic_json(report_path, report)
                report['cases'][key] = _execute_notebook(notebook, repo, output,
                                                          key + '-executed.ipynb', args.deadline_utc)
                atomic_json(report_path, report)
            report['initial_comparisons'][name] = _initial_pair(output, name)
            atomic_json(report_path, report)
            if report['initial_comparisons'][name]['status'] != 'matched':
                raise RuntimeError(f'{name} full-run initialization diverged from verified common start')
        if time.time() + 60 < args.deadline_utc:
            _write_workflow_config(repo, output, 'FWI', args.epochs, args.device)
            report['cases']['FWI-new'] = _execute_notebook(
                build_fwi_notebook(repo, output, args.epochs, args.deadline_utc),
                repo, output, 'FWI-new-executed.ipynb', args.deadline_utc)
        else:
            report['cases']['FWI-new'] = dict(status='not_started', reason='insufficient deadline margin')
        report['status'] = 'completed'
    except Exception as error:
        report['status'] = 'deadline' if isinstance(error, TimeoutError) else 'failed'
        if 'current_case' in report and report['current_case'] not in report['cases']:
            report['cases'][report['current_case']] = dict(status=report['status'])
        report['error'] = dict(type=type(error).__name__, message=str(error),
                               traceback=traceback.format_exc(limit=12))
    finally:
        try:
            report['selected_checkpoint'] = _select_checkpoint(output)
        except Exception as error:
            report['selected_checkpoint'] = dict(status='selection_failed', error=str(error))
        atomic_json(report_path, report)
        atomic_json(output / 'artifact_manifest.json', _manifest(output))
        atomic_json(args.status_file, dict(status=report['status'], updated_utc=time.time(),
                                           result=str(report_path)))
    return 0 if report['status'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
