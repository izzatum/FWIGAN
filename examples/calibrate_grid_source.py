"""Bounded CPU-only source cell-area calibration for pinned DeepWave 0.0.9.

python examples/calibrate_grid_source.py --output runs/grid-source-calibration.json
The fitted gain is a diagnostic only; production uses (3/30)**2 exactly.
This interior off-source homogeneous probe does not establish full Marmousi
agreement, surface/colocated source behavior, or inversion convergence.
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import deepwave
import torch
from deepwave.scalar.scalar import Timestep

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(2)
started = time.monotonic()
nt, dt, velocity = 256, 0.003, 2100.0
source = deepwave.wavelets.ricker(7.0, nt, dt, 1 / 7.0).reshape(nt, 1, 1)
sources = torch.tensor([[[300.0, 900.0]]])
offsets = torch.tensor([300.0, 450.0, 600.0])
receivers = torch.stack((torch.full_like(offsets, 300.0), 900 + offsets), -1)[None]
outputs, runs = {}, {}
for name, shape, dx, pml in [
    ("coarse", (40, 70), 30.0, 10),
    ("fine", (391, 691), 3.0, 100),
]:
    t0 = time.monotonic()
    vp = torch.full(shape, velocity)
    timestep = Timestep(dt, torch.tensor([dx, dx]), velocity)
    with torch.no_grad():
        prop = deepwave.scalar.Propagator({"vp": vp}, dx, pml_width=pml, survey_pad=None)
        traces = prop(source, sources, receivers, dt).detach().clone()
    outputs[name] = traces
    runs[name] = {
        "shape": shape, "dx_m": dx, "pml_cells_per_side": pml,
        "pml_width_m": pml * dx, "inner_dt_s": timestep.inner_dt,
        "step_ratio": timestep.step_ratio, "elapsed_s": time.monotonic() - t0,
        "peak_amplitude": traces.abs().max().item(),
        "finite": bool(torch.isfinite(traces).all()),
        "requires_grad": traces.requires_grad,
    }
    print(name, runs[name], flush=True)

def compare(coarse, fine):
    coarse, fine = coarse.double().flatten(), fine.double().flatten()
    gain = torch.dot(coarse, fine) / torch.dot(coarse, coarse)
    return {
        "best_coarse_to_fine_gain": gain.item(),
        "raw_relative_l2": (torch.linalg.vector_norm(coarse - fine) / torch.linalg.vector_norm(fine)).item(),
        "best_gain_relative_l2": (torch.linalg.vector_norm(gain * coarse - fine) / torch.linalg.vector_norm(fine)).item(),
        "cell_area_gain_relative_l2": (torch.linalg.vector_norm(0.01 * coarse - fine) / torch.linalg.vector_norm(fine)).item(),
        "coarse_to_fine_norm_ratio": (torch.linalg.vector_norm(coarse) / torch.linalg.vector_norm(fine)).item(),
        "correlation_cosine": (torch.dot(coarse, fine) / torch.linalg.vector_norm(coarse) / torch.linalg.vector_norm(fine)).item(),
    }

times = torch.arange(nt) * dt
direct = (times[:, None] - offsets[None] / velocity - 1 / 7.0).abs() <= 0.1
result = {
    "schema": 1, "description": "Interior homogeneous off-source raw grid calibration; not full Marmousi equivalence",
    "python_version": platform.python_version(),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "deepwave_python_sha256": hashlib.sha256(Path(deepwave.scalar.scalar.__file__).read_bytes()).hexdigest(),
    "deepwave_version": "0.0.9", "torch_version": torch.__version__,
    "device": "cpu", "threads": 2, "nt": nt, "dt_s": dt,
    "frequency_hz": 7.0, "peak_time_s": 1 / 7.0, "velocity_m_per_s": velocity,
    "source_m": sources.tolist(), "receivers_m": receivers.tolist(),
    "physical_domain_sample_extents_m": [1170.0, 2070.0],
    "expected_cell_area_gain": (3 / 30) ** 2, "runs": runs,
    "all_samples": compare(outputs["coarse"], outputs["fine"]),
    "direct_arrival_window": compare(outputs["coarse"][:, 0][direct], outputs["fine"][:, 0][direct]),
    "per_receiver": [compare(outputs["coarse"][:, 0, i], outputs["fine"][:, 0, i]) for i in range(3)],
    "elapsed_s": time.monotonic() - started,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
