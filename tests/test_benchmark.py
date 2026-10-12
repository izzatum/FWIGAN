"""Pinned CUDA startup contracts checked without allocating a GPU."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch


class MetadataQueryReached(Exception):
    pass


def test_cuda_benchmark_memory_metadata_queries_current_device(tmp_path, monkeypatch):
    """The real pinned index validator rejects cuda without an explicit index."""
    path=Path(__file__).resolve().parents[1]/'examples'/'benchmark.py'
    spec=importlib.util.spec_from_file_location('benchmark_device_regression',path)
    benchmark=importlib.util.module_from_spec(spec);spec.loader.exec_module(benchmark)
    monkeypatch.setattr(sys,'argv',[str(path),'--device','cuda','--suite','all',
        '--output',str(tmp_path/'benchmark.json')])
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'current_device',lambda:3)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda device:'validation GPU')
    monkeypatch.setattr(torch,'set_num_threads',lambda count:None)
    monkeypatch.setattr(torch,'set_num_interop_threads',lambda count:None)
    monkeypatch.setattr(torch.backends.cudnn,'deterministic',torch.backends.cudnn.deterministic)
    monkeypatch.setattr(torch.backends.cudnn,'benchmark',torch.backends.cudnn.benchmark)
    class Runtime:
        def cudaMemGetInfo(self,index):
            # Keep real Torch1.13 mem_get_info and its strict index validator.
            # Only the hardware runtime is replaced; cuda:0 must also fail
            # when the selected current device is 3.
            assert index==3
            raise MetadataQueryReached
    monkeypatch.setattr(torch.cuda,'cudart',lambda:Runtime())
    with pytest.raises(MetadataQueryReached):benchmark.main()
    assert not (tmp_path/'benchmark.json').exists()
