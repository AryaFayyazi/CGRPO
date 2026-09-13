import os
import random
import numpy as np
import torch
import subprocess
from typing import Any, Dict, List, Optional


def total_free_gpu_mib() -> List[int]:
    """Return free GPU memory (MiB) for each visible CUDA device."""
    if not torch.cuda.is_available():
        return []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
            capture_output=True, text=True, check=True,
        )
        all_free = [int(x.strip().split()[0]) for x in result.stdout.strip().splitlines() if x.strip()]
        vis_env = os.environ.get("CUDA_VISIBLE_DEVICES")
        if vis_env:
            try:
                vis_phys = [int(x) for x in vis_env.split(",") if x.strip()]
                return [all_free[i] for i in vis_phys if i < len(all_free)]
            except (ValueError, IndexError):
                pass
        return all_free[: torch.cuda.device_count()]
    except Exception:
        return [int(torch.cuda.mem_get_info(i)[0] / 1024 / 1024) for i in range(torch.cuda.device_count())]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}


def pick_cuda_devices(num: int = 1) -> List[int]:
    """Return a list of ``num`` GPU indices with the most **free** memory.

    The return values are *logical* device indices as seen by PyTorch,
    i.e. 0..``torch.cuda.device_count()-1``.  We respect
    ``CUDA_VISIBLE_DEVICES`` by mapping physical GPU IDs to logical ones.

    On systems with ``nvidia-smi`` we query the driver for free memory and
    choose the GPUs with the largest available memory.  This tends to
    avoid crowded devices.  If the query fails we fall back to a simple
    round‑robin over the visible GPUs.

    Example
    -------
    >>> pick_cuda_devices(2)
    [1, 0]  # logical indices, not necessarily the same as physical IDs
    """
    if not torch.cuda.is_available():
        return []
    # figure out visible physical devices
    vis_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis_env:
        try:
            vis_phys = [int(x) for x in vis_env.split(",") if x.strip() != ""]
        except ValueError:
            vis_phys = []
    else:
        vis_phys = list(range(torch.cuda.device_count()))

    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        all_mems = [int(x.strip().split()[0]) for x in output.stdout.strip().splitlines() if x.strip()]
        # select memory values only for visible physical GPUs, preserving order
        visible_mems = [(i, all_mems[i]) for i in vis_phys if i < len(all_mems)]
        # sort by free memory descending
        visible_mems.sort(key=lambda pair: pair[1], reverse=True)
        # convert physical IDs to logical indices
        logical_order = []
        for phys, _ in visible_mems:
            if phys in vis_phys:
                logical_order.append(vis_phys.index(phys))
        return logical_order[:min(num, len(logical_order))]
    except Exception:
        # fall back to first ``num`` visible GPUs
        return list(range(min(num, torch.cuda.device_count())))


def batch_to_examples(batch: Any) -> List[Dict]:
    """Convert a dataset slice to a list of example dictionaries.

    HuggingFace ``Dataset`` slicing (e.g. ``ds[0:4]``) returns a
    *dictionary of lists* rather than a list of dictionaries.  Iterating
    over that object yields the column names (strings), which breaks any
    code that expects ``for ex in batch:`` to give a dict.  This helper
    normalizes both representations into ``[{{...}}, ...]``.
    """
    if isinstance(batch, dict):
        # determine number of examples from first column
        if not batch:
            return []
        length = len(next(iter(batch.values())))
        return [ {k: batch[k][i] for k in batch} for i in range(length) ]
    elif isinstance(batch, list):
        return batch
    else:
        try:
            return list(batch)
        except Exception:
            raise TypeError(f"Cannot convert batch of type {type(batch)} to examples")
