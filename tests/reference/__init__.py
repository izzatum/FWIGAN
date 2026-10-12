"""Frozen notebook definitions; never import the extracted implementation here."""
import json
from pathlib import Path
import torch
import numpy as np
import deepwave as dw
from torch import nn

FROZEN = json.loads(Path(__file__).with_name('notebook_cells.json').read_text())
_namespace = {'torch': torch, 'np': np, 'dw': dw, 'nn': nn}
for _index in ('8', '10'):
    exec(FROZEN['notebooks']['notebooks/FWIGAN.ipynb'][_index]['source'], _namespace)
data_normalization = _namespace['data_normalization']
Generator = _namespace['Generator']
Critic = _namespace['Critic']
gradient_penalty = _namespace['gradient_penalty']
add_noise = _namespace['add_noise']
