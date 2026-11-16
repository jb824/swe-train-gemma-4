"""Stop numerical failures before updates or portable artifacts are accepted."""

import json
from pathlib import Path


def require_finite(named_tensors, stage, *, device=None, distributed=True):
  """Inspect local shards; synchronize one failure flag, never gather the base."""
  import torch
  from torch.distributed.tensor import DTensor
  checks = {}
  for name, tensor in named_tensors:
    if tensor is None:
      continue
    value = torch.as_tensor(tensor).detach()
    if isinstance(value, DTensor):
      value = value.to_local()
    if value.is_sparse:
      value = value.coalesce().values()
    checks.setdefault(value.device, []).append((name, torch.isfinite(value).all()))
  # One device-to-host synchronization per device on the healthy path, rather
  # than one for every projection tensor. Only transfer names after a failure.
  bad = []
  for entries in checks.values():
    if not torch.stack([flag for _, flag in entries]).all().item():
      bad.extend(name for name, flag in entries if not flag.item())
  failed = bool(bad)
  if distributed and torch.distributed.is_initialized():
    if device is None:
      device = torch.device('cuda', torch.cuda.current_device()) if torch.distributed.get_backend() == 'nccl' else 'cpu'
    flag = torch.tensor(int(failed), device=device)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
    failed = bool(flag.item())
  if failed:
    detail = ', '.join(bad[:3]) if bad else 'another distributed rank'
    raise FloatingPointError(f'{stage}: NaN/Inf detected in {detail}. Training/artifact creation stopped.')


def check_adapter_file(path):
  """Validate saved tensors on CPU without requiring a training/Torch install."""
  import numpy as np
  from safetensors import safe_open
  path = Path(path)
  with safe_open(path, framework='np') as weights:
    names = list(weights.keys())
    if not names:
      raise ValueError(f'Empty adapter: {path}')
    # NumPy has no native BF16 dtype. Inspect its exponent bits in the Rust-
    # validated safetensors data region; both Inf and NaN have all exponent bits
    # set. Keep the file memory-mapped rather than duplicating the full adapter.
    with path.open('rb') as handle:
      size = int.from_bytes(handle.read(8), 'little')
      header = json.loads(handle.read(size))
    raw = np.memmap(path, mode='r', dtype=np.uint8)
    bad = []
    for name in names:
      if weights.get_slice(name).get_dtype() == 'BF16':
        start, end = header[name]['data_offsets']
        value = raw[8 + size + start:8 + size + end].view('<u2')
        finite = ((value & 0x7F80) != 0x7F80).all()
      else:
        finite = np.isfinite(weights.get_tensor(name)).all()
      if not finite:
        bad.append(name)
    if bad:
      raise FloatingPointError(f'Adapter {path}: NaN/Inf detected in {", ".join(bad[:3])}. Artifact rejected.')
