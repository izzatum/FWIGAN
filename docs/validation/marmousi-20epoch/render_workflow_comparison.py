#!/usr/bin/env python3
"""Render actual bounded notebook-workflow outputs; accepts trusted local Torch artifacts only.

The script never runs an inversion or fills missing results. A tiny CPU smoke bundle
is labelled as such and cannot be presented as the full Marmousi comparison.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch

NAMES = ('FWIGAN', 'FWIGAN_Noisy')
KINDS = ('old', 'new')
COLORS = {'FWIGAN-old': '#0072B2', 'FWIGAN-new': '#E69F00',
          'FWIGAN_Noisy-old': '#009E73', 'FWIGAN_Noisy-new': '#CC79A7'}
LABELS = {'FWIGAN-old': 'Original clean', 'FWIGAN-new': 'Package clean',
          'FWIGAN_Noisy-old': 'Original noisy', 'FWIGAN_Noisy-new': 'Package noisy'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_manifest(bundle, required_paths):
    manifest_path = bundle / 'artifact_manifest.json'
    if not manifest_path.is_file():
        return None, []
    manifest = read_json(manifest_path)
    indexed = {entry['path']: entry['sha256'] for entry in manifest.get('files', [])}
    verified = []
    for path in required_paths:
        relative = str(path.relative_to(bundle))
        if relative in indexed:
            if digest(path) != indexed[relative]:
                raise ValueError(f'artifact manifest SHA-256 mismatch: {relative}')
            verified.append(relative)
    return digest(manifest_path), verified


def safe_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def fmt(value, unit='', digits=4):
    number = safe_float(value)
    return '—' if number is None else f'{number:.{digits}g}{unit}'


def load_case(bundle, name, kind):
    key = f'{name}-{kind}'
    metrics_path = bundle / f'{key}-metrics.json'
    if not metrics_path.is_file():
        return None
    metrics = read_json(metrics_path)
    rows = metrics.get('rows', [])
    if not rows:
        return None
    epochs = [int(row['epoch']) for row in rows]
    if epochs != sorted(set(epochs)):
        raise ValueError(f'{metrics_path}: epochs must be unique and increasing')
    final_path = None
    if kind == 'old':
        candidate = bundle / f'{name}-old-final.pt'
        if candidate.is_file():
            final_path = candidate
    else:
        candidates = list((bundle / f'{name}-new-runs').glob('*/final.pt'))
        if len(candidates) > 1:
            raise ValueError(f'{key}: more than one final.pt')
        if candidates:
            final_path = candidates[0]
    final = torch.load(final_path, map_location='cpu') if final_path else None
    if final is not None:
        for field in ('velocity', 'source', 'sigma'):
            if field not in final or not torch.is_tensor(final[field]):
                raise ValueError(f'{final_path}: missing tensor {field}')
        if kind == 'old' and int(final['epoch']) != epochs[-1]:
            raise ValueError(f'{key}: final epoch disagrees with metric rows')
    return dict(key=key, name=name, kind=kind, rows=rows, final=final,
                final_path=final_path, metrics_path=metrics_path)


def final_row(case):
    return case['rows'][-1]


def updates(row, kind):
    if kind == 'old':
        actual = row.get('actual_updates') or {}
        skipped = row.get('skipped_updates') or {}
        return (int(actual.get('physical', 0)), int(skipped.get('physical', 0)),
                int(actual.get('critic', 0)), int(skipped.get('critic', 0)))
    counts = row.get('counters') or {}
    return (int(counts.get('physical_updates', 0)), int(counts.get('physical_skipped', 0)),
            int(counts.get('critic_updates', 0)), int(counts.get('critic_skipped', 0)))


def save_figure(fig, output, stem):
    fig.savefig(output / f'{stem}.png', dpi=180, bbox_inches='tight', facecolor='white')
    fig.savefig(output / f'{stem}.svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def setup_style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.titlesize': 10, 'axes.labelsize': 9,
                         'figure.titlesize': 12, 'savefig.pad_inches': .13,
                         'axes.spines.top': False, 'axes.spines.right': False})


def grid_extent(data):
    spacing = data.grid.spacing
    if isinstance(spacing, (tuple, list)):
        dz, dx = map(float, spacing)
    else:
        dz = dx = float(spacing)
    nz, nx = data.grid.shape
    return (0, nx * dx, nz * dz, 0), dz, dx


def as_array(tensor, shape=None):
    array = tensor.detach().cpu().numpy()
    if shape is not None and tuple(array.shape) != tuple(shape):
        raise ValueError(f'expected {shape}, got {array.shape}')
    if not np.isfinite(array).all():
        raise ValueError('nonfinite tensor in figure input')
    return array


def velocity_figures(data, cases, output, scope):
    models = {'truth': as_array(data.reference_velocity, data.grid.shape),
              'initial': as_array(data.initial_velocity, data.grid.shape)}
    for key, case in cases.items():
        if case['final'] is not None:
            models[key] = as_array(case['final']['velocity'], data.grid.shape)
    if len(models) < 4:
        return []
    extent, dz, dx = grid_extent(data)
    lower = min(float(np.min(a)) for a in models.values())
    upper = max(float(np.max(a)) for a in models.values())
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.8), sharex=True, sharey=True, constrained_layout=True)
    last = None
    for row, name in enumerate(NAMES):
        for col, (label, key) in enumerate((('Coarse truth', 'truth'), ('Shared initial', 'initial'),
                                            ('Original final', f'{name}-old'), ('Package final', f'{name}-new'))):
            ax = axes[row, col]
            if key in models:
                last = ax.imshow(models[key], cmap='viridis', vmin=lower, vmax=upper,
                                 origin='upper', extent=extent, aspect='auto', interpolation='nearest')
            else:
                ax.text(.5, .5, 'Final tensor unavailable', transform=ax.transAxes, ha='center', va='center')
            ax.set_title(label)
            ax.set_xlabel('x (m)')
            if col == 0:
                ax.set_ylabel(('Clean' if row == 0 else 'Noisy') + '\ndepth z (m)')
    if last is not None:
        fig.colorbar(last, ax=axes, label='Velocity (m/s)', shrink=.82)
    fig.suptitle(f'{scope}: shared velocity scale; final models at recorded epochs')
    save_figure(fig, output, 'velocity-models')
    differences = []
    for name in NAMES:
        old, new = models.get(f'{name}-old'), models.get(f'{name}-new')
        if old is not None and new is not None:
            differences.append(new - old)
    if not differences:
        return ['velocity-models']
    limit = max(float(np.max(np.abs(a))) for a in differences)
    all_exactly_equal = limit == 0
    if all_exactly_equal:
        limit = 1.0
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for row, name in enumerate(NAMES):
        old, new = models.get(f'{name}-old'), models.get(f'{name}-new')
        ax, profile = axes[row]
        if old is None or new is None:
            for target in (ax, profile):
                target.text(.5, .5, 'Paired final tensors unavailable', transform=target.transAxes,
                            ha='center', va='center')
            continue
        im = ax.imshow(new - old, cmap='RdBu_r', norm=TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit),
                       origin='upper', extent=extent, aspect='auto', interpolation='nearest')
        if all_exactly_equal:
            ax.text(.5, .5, 'All cell differences are exactly 0 m/s', transform=ax.transAxes,
                    ha='center', va='center', bbox=dict(facecolor='white', alpha=.9, edgecolor='none'))
        ax.set_title(('Clean' if row == 0 else 'Noisy') + ': package − original')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('Depth z (m)')
        depths = (np.arange(data.grid.shape[0]) + .5) * dz
        for label, array, color, style in [('Truth', models['truth'], '#222222', '-'),
                                           ('Initial', models['initial'], '#777777', '--'),
                                           ('Original', old, '#0072B2', '-'),
                                           ('Package', new, '#E69F00', '-')]:
            profile.plot(array.mean(axis=1), depths, label=label, color=color, linestyle=style)
        profile.invert_yaxis()
        profile.set_xlabel('Width-mean velocity (m/s)')
        profile.set_ylabel('Depth z (m)')
        profile.set_title(('Clean' if row == 0 else 'Noisy') + ': depth profile across all x')
        profile.legend(fontsize=7, ncol=2)
    if not all_exactly_equal:
        fig.colorbar(im, ax=[axes[0, 0], axes[1, 0]], label='Package − original velocity (m/s)', shrink=.82)
    fig.suptitle(f'{scope}: differences on one symmetric scale')
    save_figure(fig, output, 'velocity-differences')
    return ['velocity-models', 'velocity-differences']


def metric_figures(cases, output, scope):
    metrics = [('raw_residual_mse', 'Raw residual MSE (pressure-unit²)'),
               ('normalized_clean_mse', 'Clean legacy-normalized residual MSE (dimensionless)'),
               ('model_rmse', 'Coarse model RMSE (m/s)')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.1), constrained_layout=True)
    any_point = False
    for ax, (field, label) in zip(axes, metrics):
        for key in [f'{name}-{kind}' for name in NAMES for kind in KINDS]:
            case = cases.get(key)
            if case is None:
                continue
            points = [(int(row['epoch']), safe_float(row.get('evaluation', {}).get(field)))
                      for row in case['rows'] if 'evaluation' in row]
            points = [(x, y) for x, y in points if y is not None]
            if points:
                any_point = True
                ax.plot([x for x, _ in points], [y for _, y in points], marker='o', markersize=4,
                        linewidth=1.6, color=COLORS[key], linestyle='--' if case['kind']=='old' else '-',
                        label=LABELS[key])
        ax.set_xlabel('Completed epoch (evaluated milestones only)')
        ax.set_ylabel(label)
        ax.grid(alpha=.22)
    axes[0].legend(loc='best', fontsize=7)
    fig.suptitle(f'{scope}: independent evaluation metrics')
    if any_point:
        save_figure(fig, output, 'metric-histories')
        return ['metric-histories']
    plt.close(fig)
    return []


def sources_figure(data, cases, output, scope, full_marmousi):
    source = as_array(data.source).reshape(-1)
    original_true = source / .01 if full_marmousi else source
    initial = as_array(data.initial_source).reshape(-1) if data.initial_source is not None else None
    dt = float(data.sampling.dt)
    times = np.arange(len(source)) * dt
    waveforms = {}
    for key, case in cases.items():
        if case['final'] is not None:
            waveform = as_array(case['final']['source']).reshape(-1)
            if len(waveform) == len(source):
                waveforms[key] = waveform
    if not waveforms:
        return []
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.4), constrained_layout=True)
    ax = axes[0, 0]
    def normal_shape(values):
        return values / max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    ax.plot(times, normal_shape(original_true), color='#222222', linewidth=2,
            label='Observed 7 Hz shape' if full_marmousi else 'Known source shape')
    if initial is not None:
        ax.plot(times, normal_shape(initial), color='#777777', linestyle=':',
                label='Initial 10 Hz shape' if full_marmousi else 'Initial source shape')
    for key, waveform in waveforms.items():
        ax.plot(times, normal_shape(waveform), color=COLORS[key], linewidth=1.2,
                linestyle='--' if key.endswith('old') else '-', label=LABELS[key])
    if full_marmousi:
        ax.set_xlim(0, min(float(times[-1]), .6))
    ax.set(title='Waveform shape (first 0.6 s shown; full norm at right)' if full_marmousi else
           'Waveform shape (each normalized by own peak)', xlabel='Time (s)',
           ylabel='Relative coefficient (dimensionless)')
    ax.legend(fontsize=7, ncol=2)
    ax = axes[0, 1]
    values = [float(np.linalg.norm(original_true))]
    labels = ['True 7 Hz\nunscaled' if full_marmousi else 'Known source']
    colors = ['#222222']
    if full_marmousi:
        labels.append('Known inverse-grid\n0.01 × 7 Hz')
        values.append(float(np.linalg.norm(source)))
        colors.append('#777777')
    if initial is not None:
        labels.append('Initial 10 Hz' if full_marmousi else 'Initial source')
        values.append(float(np.linalg.norm(initial)))
        colors.append('#BBBBBB')
    for key, waveform in waveforms.items():
        labels.append(LABELS[key].replace(' ', '\n'))
        values.append(float(np.linalg.norm(waveform)))
        colors.append(COLORS[key])
    ax.bar(np.arange(len(values)), values, color=colors)
    ax.set_xticks(np.arange(len(values)), labels, rotation=20, ha='right', fontsize=7)
    if all(value > 0 for value in values) and max(values) / min(values) > 30:
        ax.set_yscale('log')
    ax.set(title='Absolute waveform L2 norms; distinct amplitude references',
           ylabel='Coefficient L2 norm (a.u.)')
    ax = axes[1, 0]
    for key in waveforms:
        row = final_row(cases[key]); metric = row.get('evaluation', {})
        value = safe_float(metric.get('source_correlation'))
        if value is not None:
            ax.scatter([LABELS[key]], [value], color=COLORS[key], s=70)
    ax.tick_params(axis='x', labelrotation=20, labelsize=7)
    ax.set(title='Source cosine correlation with ' + ('observed 7 Hz' if full_marmousi else 'known source'),
           ylabel='Cosine correlation (dimensionless)')
    ax.set_ylim(-1.05, 1.05)
    ax.grid(axis='y', alpha=.2)
    ax = axes[1, 1]
    for key in waveforms:
        row = final_row(cases[key]); metric = row.get('evaluation', {})
        sigma = safe_float(metric.get('sigma_signed'))
        if sigma is not None:
            ax.bar(LABELS[key], sigma, color=COLORS[key])
    ax.axhline(0, color='#222222', linewidth=.8)
    ax.axhline(.05, color='#666666', linestyle=':', linewidth=1,
               label='Configured noisy observation std 0.05 (not a recovery target)')
    ax.tick_params(axis='x', labelrotation=20, labelsize=7)
    ax.set(title='Learned signed fake-noise σ', ylabel='σ (normalized-data units)')
    ax.legend(fontsize=7)
    fig.suptitle(f'{scope}: source shape, amplitude and signed noise')
    save_figure(fig, output, 'source-and-sigma')
    return ['source-and-sigma']


def work_figure(cases, output, scope):
    ordered = [cases[key] for key in [f'{name}-{kind}' for name in NAMES for kind in KINDS] if key in cases]
    if not ordered:
        return []
    labels = [LABELS[c['key']] for c in ordered]
    colors = [COLORS[c['key']] for c in ordered]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.5), constrained_layout=True)
    for ax, position, title in [(axes[0, 0], 0, 'Physical optimizer calls'),
                                (axes[0, 1], 2, 'Critic optimizer calls')]:
        actual = [updates(final_row(c), c['kind'])[position] for c in ordered]
        skipped = [updates(final_row(c), c['kind'])[position + 1] for c in ordered]
        ax.bar(labels, actual, color=colors, label='Actual')
        ax.bar(labels, skipped, bottom=actual, color='#BDBDBD', label='Skipped attempt')
        ax.set(title=title, ylabel='Cumulative calls / attempts')
        ax.tick_params(axis='x', labelrotation=20, labelsize=7)
        ax.legend(fontsize=7)
    ax = axes[1, 0]
    seconds = [safe_float(final_row(c).get('elapsed_train_seconds')) for c in ordered]
    if all(x is not None for x in seconds):
        ax.bar(labels, seconds, color=colors)
        ax.set(title='Synchronized cumulative training time (excludes evaluation/checkpoints)',
               ylabel='Training time (s)')
    else:
        ax.text(.5, .5, 'Comparable training-time fields unavailable', transform=ax.transAxes,
                ha='center', va='center')
    ax.tick_params(axis='x', labelrotation=20, labelsize=7)
    ax = axes[1, 1]
    memory = [final_row(c).get('memory') for c in ordered]
    if all(isinstance(m, dict) and m.get('peak_allocated') is not None and m.get('peak_reserved') is not None
           for m in memory):
        x = np.arange(len(ordered))
        width = .38
        ax.bar(x-width/2, [m['peak_allocated']/2**30 for m in memory], width, color='#0072B2',
               label='Peak allocated')
        ax.bar(x+width/2, [m['peak_reserved']/2**30 for m in memory], width, color='#E69F00',
               label='Peak reserved')
        ax.set_xticks(x, labels, rotation=20, ha='right', fontsize=7)
        ax.set(title='Process-live cumulative CUDA allocator peaks', ylabel='Memory (GiB)')
        ax.legend(fontsize=7)
    else:
        ax.text(.5, .5, 'No CUDA allocator peak data\n(CPU smoke or incomplete bundle)',
                transform=ax.transAxes, ha='center', va='center')
        ax.set_xticks([])
    fig.suptitle(f'{scope}: completed work and measured resources')
    save_figure(fig, output, 'work-and-resources')
    return ['work-and-resources']


def scope_sentence(scope):
    if scope == 'Full-grid 20-epoch Marmousi GPU':
        return 'This is the completed full-grid, 20-epoch Marmousi GPU comparison.'
    if scope == 'Partial full-grid Marmousi GPU':
        return ('This is a partial full-grid Marmousi GPU run. The recorded cases and epochs '
                'do not establish a completed four-case, 20-epoch comparison.')
    if scope == 'Tiny CPU smoke':
        return 'This is a tiny CPU renderer check; its numerical values are not Marmousi findings.'
    return 'This is an unclassified partial artifact; no full Marmousi comparison is established.'


def report_text(bundle, data, cases, fwi_case, result, scope, full, calibrated_source, files, output):
    observed = as_array(data.traces)
    nz, nx = data.grid.shape
    extent, dz, dx = grid_extent(data)
    epochs = [int(final_row(case)['epoch']) for case in cases.values()]
    lines = [f'# {scope} notebook workflow comparison', '',
             f'**Evidence class:** {scope}. {scope_sentence(scope)}',
             f'**Bundle:** `{bundle}`. **Shared traces:** {list(observed.shape)} [shots, receivers, samples]; '
             f'inverse grid {nz} × {nx} cells at {dz:g} × {dx:g} m; dt={float(data.sampling.dt):g} s. '
             f'Last completed epoch {max(epochs)}; completed epoch values by case: {sorted(set(epochs))}.',
             f'**Shared observation SHA-256:** `{digest(bundle / "shared_observations.pt")}`.', '']
    if result:
        lines += [f'Workflow status: `{result.get("status", "unknown")}`; requested horizon: '
                  f'`{result.get("epoch_horizon", "unknown")}`; device: `{result.get("device", "unknown")}`; '
                  f'baseline notebook commit: `{result.get("baseline_commit", "unknown")}`.',
                  f'Preparation: `{json.dumps(result.get("preparation", {}), ensure_ascii=False)}`.',
                  'Initial preflight/full-run pair checks: ' + ', '.join(
                      f'`{name}={entry.get("status", "unknown")}`'
                      for name, entry in result.get('initial_comparisons', {}).items()) + '.',
                  'Notebook execution cases: ' + ', '.join(
                      f'`{name}={entry.get("status", "unknown")}`'
                      for name, entry in result.get('cases', {}).items()) + '.', '']
    else:
        lines += ['The bundle has no `result.json` or artifact manifest; source, lifecycle and transport claims '
                  'cannot be inferred from this artifact.', '']
    source_heading = ('Source norm / unscaled 7 Hz | Source norm / calibrated coarse' if calibrated_source else
                      'Source norm ratio (Marmousi-only) | Source norm ratio (Marmousi-only)')
    lines += ['| Case | Last epoch | Raw residual MSE (pressure-unit²) | Clean normalized residual MSE | Model RMSE (m/s) | Source cosine | ' + source_heading + ' | Signed σ | Physical actual / skipped | Critic actual / skipped |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |']
    for key in [f'{name}-{kind}' for name in NAMES for kind in KINDS]:
        case = cases.get(key)
        if not case:
            lines.append(f'| {LABELS[key]} | unavailable | — | — | — | — | — | — | — | — | — |')
            continue
        row = final_row(case)
        ev = row.get('evaluation', {})
        up = updates(row, case['kind'])
        lines.append('| ' + ' | '.join([LABELS[key], str(row['epoch']),
                     fmt(ev.get('raw_residual_mse')), fmt(ev.get('normalized_clean_mse')),
                     fmt(ev.get('model_rmse'), ' m/s'), fmt(ev.get('source_correlation')),
                     fmt(ev.get('source_norm_ratio_unscaled')) if calibrated_source else '—',
                     fmt(ev.get('source_norm_ratio_coarse')) if calibrated_source else '—',
                     fmt(ev.get('sigma_signed')), f'{up[0]} / {up[1]}', f'{up[2]} / {up[3]}']) + ' |')
    lines += ['', 'Raw residual MSE compares raw predicted and shared observed pressures. '
              '`normalized_clean_mse` compares legacy-normalized predicted and **clean** shared observed traces '
              'even for noisy cases; it excludes the once-injected noisy adversarial target. '
              'Sigma is signed and in normalized-data units; 0.05 is the configured observation-noise '
              'standard deviation, not an assured recovery target.', '']
    if result:
        lines += ['| Case | Synchronized training-only seconds | Notebook wall time (s) |',
                  '| --- | ---: | ---: |']
        for key in [f'{name}-{kind}' for name in NAMES for kind in KINDS]:
            case = cases.get(key)
            train = final_row(case).get('elapsed_train_seconds') if case else None
            wall = result.get('cases', {}).get(key, {}).get('seconds')
            lines.append(f'| {LABELS[key]} | {fmt(train)} | {fmt(wall)} |')
        lines += ['', 'Training-only seconds exclude evaluation and checkpoints; notebook wall time '
                  'includes the executed notebook cells and their lifecycle/I/O. A shorter training-only '
                  'bar does not establish a shorter whole-notebook run.', '']
    if fwi_case is not None:
        row = final_row(fwi_case)
        evaluated = row.get('evaluation', {})
        wall = result.get('cases', {}).get('FWI-new', {}).get('seconds') if result else None
        lines += ['### Conventional FWI companion', '',
                  f'The package-only FWI run reached epoch {row["epoch"]} on the same shared raw data '
                  'with the fixed calibrated inverse-grid 7 Hz source. '
                  f'Raw residual MSE: {fmt(evaluated.get("raw_residual_mse"))} pressure-unit²; '
                  f'coarse model RMSE: {fmt(evaluated.get("model_rmse"), " m/s")}; '
                  f'training-only time: {fmt(row.get("elapsed_train_seconds"), " s")}; '
                  f'notebook wall time: {fmt(wall, " s")}. '
                  'No original-notebook FWI counterpart was run, and this companion is not an '
                  'adversarial-versus-FWI convergence verdict.', '']
    if calibrated_source:
        lines += ['The fine-grid 7 Hz observation source is unscaled; the known coarse inverse-grid source '
                  'is 0.01 × that waveform. Both amplitude references are identified in the source figure. '
                  'FWIGAN begins from an unscaled 10 Hz initializer. Source cosine correlation measures shape, '
                  'while the two norm ratios measure amplitude against different references. '
                  'The waveform-shape panel shows the first 0.6 s for pulse readability; '
                  'amplitude norms use the full 6 s waveform.', '']
    else:
        lines += ['This artifact does not verify the 0.01 coarse-grid Marmousi source calibration. '
                  'Source norm ratio field names come from the common workflow helper and should not '
                  'be interpreted as Marmousi amplitude references here.', '']
    lines += ['Figures generated from actual bundle arrays and metric rows: ' + ', '.join(
              f'`{stem}.png`/`.svg`' for stem in files) + '.',
              'Velocity maps use one common m/s scale; difference maps use one symmetric m/s scale. '
              'Metric lines connect observed milestone values only. Cumulative CUDA allocator peaks are '
              'process-live snapshots, not per-epoch increments. Training time excludes evaluation and '
              'checkpoint I/O when present in both old and package rows.', '']
    if full:
        lines += ['These are selected-horizon outcomes, not full 401-epoch convergence or general '
                  'method superiority. Old and package paths need not be bit-identical on CUDA; '
                  'AMP skips, native reductions and scheduler behavior may differ. No source-identifiability '
                  'claim follows from a normalized-trace adversarial objective.', '']
    lines += [f'Figure-provenance JSON: `{(output / "figure-provenance.json").name}`.', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True, help='trusted local workflow artifact directory')
    parser.add_argument('--output-dir', type=Path, required=True, help='external report/figure directory')
    parser.add_argument('--repo', type=Path, default=Path('/workspace/FWIGAN'))
    args = parser.parse_args(argv)
    bundle = args.bundle.resolve()
    output = args.output_dir.resolve()
    repo = args.repo.resolve()
    if not bundle.is_dir():
        parser.error('bundle directory does not exist')
    if output == repo or repo in output.parents:
        parser.error('output directory must be outside the product repository')
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo / 'src'))
    from fwigan.data import load_observations
    observation_path = bundle / 'shared_observations.pt'
    if not observation_path.is_file():
        raise ValueError('shared_observations.pt is required; no synthetic data will be generated')
    data = load_observations(observation_path)
    result_path = bundle / 'result.json'
    result = read_json(result_path) if result_path.is_file() else None
    cases = {}
    for name in NAMES:
        for kind in KINDS:
            case = load_case(bundle, name, kind)
            if case is not None:
                cases[case['key']] = case
    fwi_case = load_case(bundle, 'FWI', 'new')
    if not cases:
        raise ValueError('no actual per-case metric rows found')
    full_geometry = (tuple(data.grid.shape) == (100, 310) and tuple(data.traces.shape) == (30, 60, 2000)
                     and result is not None and result.get('preparation', {}).get('fine_shape') == [1000, 3100])
    complete = (result is not None and result.get('status') == 'completed' and result.get('epoch_horizon') == 20
                and len(cases) == 4 and all(final_row(c)['epoch'] == 20 and c['final'] is not None
                                             for c in cases.values())
                and all(result.get('cases', {}).get(key, {}).get('status') == 'completed' for key in cases)
                and all(result.get('initial_comparisons', {}).get(name, {}).get('status') == 'matched'
                        for name in ('FWIGAN_preflight', 'FWIGAN', 'FWIGAN_Noisy_preflight', 'FWIGAN_Noisy')))
    full = bool(full_geometry and complete and result.get('device') == 'cuda')
    if full:
        scope = 'Full-grid 20-epoch Marmousi GPU'
    elif result and result.get('device') == 'cuda' and full_geometry:
        scope = 'Partial full-grid Marmousi GPU'
    elif result and result.get('device') == 'cpu':
        scope = 'Tiny CPU smoke'
    elif result is None and tuple(data.grid.shape) == (40, 70) and tuple(data.traces.shape) == (2, 60, 128):
        scope = 'Tiny CPU smoke'
    else:
        scope = 'Unclassified partial workflow'
    manifest_sha, verified_paths = verify_manifest(
        bundle, [observation_path] + ([result_path] if result else []) +
        [case['metrics_path'] for case in cases.values()] +
        [case['final_path'] for case in cases.values() if case['final_path'] is not None] +
        ([fwi_case['metrics_path']] if fwi_case else []) +
        ([fwi_case['final_path']] if fwi_case and fwi_case['final_path'] is not None else []))
    setup_style()
    files = []
    if data.reference_velocity is not None:
        files += velocity_figures(data, cases, output, scope)
    files += metric_figures(cases, output, scope)
    files += sources_figure(data, cases, output, scope, bool(full_geometry))
    files += work_figure(cases, output, scope)
    provenance = dict(schema=1, scope=scope, full_marmousi_comparison=full,
                      input_bundle=str(bundle), result_sha256=digest(result_path) if result else None,
                      artifact_manifest_sha256=manifest_sha, manifest_verified_inputs=verified_paths,
                      shared_observations_sha256=digest(observation_path),
                      case_metrics_sha256={key:digest(case['metrics_path']) for key,case in cases.items()},
                      final_tensors_sha256={key:digest(case['final_path']) for key,case in cases.items()
                                            if case['final_path'] is not None}, figures=files,
                      fwi_companion_metrics_sha256=digest(fwi_case['metrics_path']) if fwi_case else None,
                      limitations=['Tiny CPU smoke is renderer validation only'] if scope == 'Tiny CPU smoke' else
                      ['No full 401-epoch convergence or method-superiority claim'])
    (output / 'figure-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (output / 'comparison-report.md').write_text(report_text(
        bundle, data, cases, fwi_case, result, scope, full, bool(full_geometry), files, output))
    print(json.dumps(dict(scope=scope, figures=files, report=str(output / 'comparison-report.md'))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
