"""Plot *measured* CUDA `--suite all` FWIGAN benchmark JSON.

Usage: python examples/plot_gpu_benchmark.py --input REPORT.json --output-dir FIGURES

The script never runs a benchmark and refuses CPU, tiny-only, incomplete, or
missing-memory reports. It writes gpu-benchmark.svg and gpu-benchmark.png.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", type=Path, required=True, help="successful CUDA all-suite benchmark JSON")
parser.add_argument("--output-dir", "--outputdir", dest="output_dir", type=Path, required=True)
args = parser.parse_args()

try:
    report = json.loads(args.input.read_text())
except (OSError, json.JSONDecodeError) as error:
    parser.error(f"cannot read a complete benchmark JSON: {error}")
if report.get("schema") != 1 or report.get("device") != "cuda" or report.get("suite") != "all":
    parser.error("expected a successful schema-1 CUDA --suite all report")
workloads = report.get("workloads")
required = ("tiny-clean", "tiny-noisy", "representative-clean")
if not isinstance(workloads, dict) or any(name not in workloads for name in required):
    parser.error("CUDA all report must contain tiny-clean, tiny-noisy and representative-clean")
if not isinstance(report.get("gpu"), dict) or not report["gpu"].get("name"):
    parser.error("CUDA report lacks measured GPU identity")


def finite_nonnegative(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        parser.error(f"missing or invalid measured {label}")
    return float(value)


memory_keys = ("peak_allocated_bytes", "peak_reserved_bytes",
               "peak_allocated_increment_bytes", "peak_reserved_increment_bytes")
series = {}
for name in required:
    work = workloads[name]
    variants = {}
    for variant in ("original", "optimized"):
        windows = work.get("windows", {}).get(variant)
        if not isinstance(windows, list) or len(windows) < 2:
            parser.error(f"{name}/{variant} needs at least two measured windows")
        samples = []
        memory = {key: [] for key in memory_keys}
        for index, window in enumerate(windows):
            samples.append(finite_nonnegative(window.get("seconds_per_batch"),
                f"{name}/{variant}/window{index} seconds per group"))
            item = window.get("memory")
            if not isinstance(item, dict):
                parser.error(f"{name}/{variant}/window{index} has no measured CUDA memory")
            for key in memory_keys:
                memory[key].append(finite_nonnegative(item.get(key), f"{name}/{variant}/{key}"))
            if memory["peak_allocated_bytes"][-1] > memory["peak_reserved_bytes"][-1]:
                parser.error(f"{name}/{variant}/window{index}: allocated exceeds reserved peak")
        recorded = finite_nonnegative(work.get("median_seconds_per_batch", {}).get(variant),
                                      f"{name}/{variant} median")
        if not math.isclose(recorded, statistics.median(samples), rel_tol=1e-9, abs_tol=1e-12):
            parser.error(f"{name}/{variant} recorded median differs from raw windows")
        counts = work.get("untimed_counts", {}).get(variant, {})
        batches = work.get("config", {}).get("batches")
        if not isinstance(batches, int) or batches < 1:
            parser.error(f"{name}: invalid shot-group count")
        adjoints = finite_nonnegative(counts.get("solver_adjoint"), f"{name}/{variant} adjoints")
        if adjoints % batches:
            parser.error(f"{name}/{variant}: adjoint count cannot be divided by groups")
        variants[variant] = {"samples": samples, "median": recorded,
                             "memory": memory, "adjoints_per_group": adjoints / batches}
    series[name] = variants

# Import plotting libraries only after validating the measured report.
os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(args.output_dir / ".cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "svg.fonttype": "none", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})
fig, ((ax_time, ax_counts), (ax_absolute, ax_increment)) = plt.subplots(
    2, 2, figsize=(14.0, 8.5), layout="constrained")
names = list(required)
positions = np.arange(len(names), dtype=float)
labels = []
for name in names:
    geometry = workloads[name]["geometry"]
    nz, nx = geometry["model"]
    labels.append(f"{name.replace('-', ' ').title()}\n{nz}×{nx} · {geometry['shots']} shots")
variants = (("original", "Original frozen loop", "#707782", -0.19),
            ("optimized", "Package strategy", "#0072b2", 0.19))
gib = 1024 ** 3

for variant, label, color, offset in variants:
    group_x = positions + offset
    values = [series[name][variant]["median"] for name in names]
    ax_time.bar(group_x, values, width=.33, color=color, alpha=.82, label=label)
    for x, name in zip(group_x, names):
        samples = series[name][variant]["samples"]
        jitter = np.linspace(-.08, .08, len(samples))
        ax_time.scatter(x+jitter, samples, color=color, s=28, zorder=4,
                        edgecolors="white", linewidth=.55)
    counts = [series[name][variant]["adjoints_per_group"] for name in names]
    ax_counts.bar(group_x, counts, width=.33, color=color, alpha=.82)
    for x, count in zip(group_x, counts):
        ax_counts.annotate(f"{count:g}", (x, count), xytext=(0, 3),
                           textcoords="offset points", ha="center", va="bottom")
    for axis, allocated_key, reserved_key in (
        (ax_absolute, "peak_allocated_bytes", "peak_reserved_bytes"),
        (ax_increment, "peak_allocated_increment_bytes", "peak_reserved_increment_bytes"),
    ):
        allocated = [statistics.median(series[name][variant]["memory"][allocated_key])/gib
                     for name in names]
        reserved = [statistics.median(series[name][variant]["memory"][reserved_key])/gib
                    for name in names]
        axis.bar(group_x, allocated, width=.33, color=color, alpha=.82)
        axis.scatter(group_x, reserved, marker="D", s=40, color=color,
                     edgecolors="white", linewidth=.55, zorder=4)

for ax in (ax_time, ax_counts, ax_absolute, ax_increment):
    ax.set_xticks(positions, labels)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
ax_time.set(title="Warmed training wall time · dots are repetitions",
            ylabel="Seconds per shot group")
ax_counts.set(title="Instrumented DeepWave reverse solves",
              ylabel="Adjoint calls per shot group")
ax_absolute.set(title="Absolute CUDA peak · bars allocated, diamonds reserved",
                ylabel="Peak memory (GiB)")
ax_increment.set(title="Increase above pre-window baseline",
                 ylabel="Peak increment (GiB)")
ax_time.legend(loc="upper right", frameon=False, fontsize=9)
cycles = sorted({workloads[name]["config"]["critic_cycles"] for name in names})
fig.suptitle(f"Measured synthetic CUDA benchmark · {report['gpu']['name']} · "
             f"{','.join(map(str, cycles))} critic cycles\n"
             "Matched optimizer work; timing and memory depend on this device and workload",
             fontsize=12)
fig.text(.5, -.012, "Memory bars and diamonds are medians of recorded peaks across repetitions. "
         "These are training-loop measurements, not inversion convergence or full Marmousi results.",
         ha="center", fontsize=9, color="#4b5563")
args.output_dir.mkdir(parents=True, exist_ok=True)
for extension in ("svg", "png"):
    destination = args.output_dir / f"gpu-benchmark.{extension}"
    fig.savefig(destination, dpi=190, bbox_inches="tight")
    if extension == "svg":
        destination.write_text("\n".join(line.rstrip() for line in destination.read_text().splitlines()) + "\n")
plt.close(fig)
print(args.output_dir / "gpu-benchmark.svg")
