import json
from types import SimpleNamespace

import pytest

from swegemma_training.data import prepare
from swegemma_training.modeling import Collator
from swegemma_training.prepared import prepared_datasets
from swegemma_training.reference import prepared_rows
from swegemma_training.resources import StartupMonitor, host_memory
from swegemma_training.runner import configure_cpu_offload


@pytest.mark.parametrize('mode', ['turn', 'trajectory', 'windowed'])
def test_disk_backed_tokens_preserve_examples_and_collation(traces, tokenizer, mode):
  import torch
  from datasets.table import MemoryMappedTable
  traces.data.sequence_mode = mode
  manifest = prepare(traces, tokenizer, schemas=[])
  previous = prepared_rows(traces)
  datasets = prepared_datasets(traces, manifest)
  collate = Collator(tokenizer.pad_token_id)
  for split, dataset in datasets.items():
    assert isinstance(dataset.data, MemoryMappedTable)
    expected = [{key: row[key] for key in ('input_ids', 'attention_mask', 'labels', 'example_id')}
                for row in previous[split]]
    assert list(dataset) == expected
    batch = collate(list(dataset))
    reference = collate(expected)
    for key in batch:
      torch.testing.assert_close(batch[key], reference[key])
    assert all(row['example_id'] == f'{split}-{index:06d}' for index, row in enumerate(dataset))
  # Reuse reads the same bounded, disk-backed token cache.
  assert prepared_datasets(traces, manifest)['train'].cache_files == datasets['train'].cache_files


def test_host_memory_units_and_missing_proc_files(tmp_path):
  (tmp_path / 'self').mkdir()
  (tmp_path / 'self/status').write_text('Name:\tpython\nVmRSS:\t100 kB\nVmSwap:\t23 kB\nVmPin:\t8 kB\n')
  (tmp_path / 'meminfo').write_text('MemTotal: 1000 kB\nMemAvailable: 200 kB\nShmem: 300 kB\n')
  assert host_memory(tmp_path) == {
      'process_rss_bytes': 102400, 'process_swap_bytes': 23552, 'process_pinned_bytes': 8192,
      'system_total_bytes': 1024000, 'system_available_bytes': 204800, 'system_shared_bytes': 307200}
  assert host_memory(tmp_path / 'missing') == {}


def test_startup_failure_preserves_phase_and_memory(tmp_path, monkeypatch):
  monkeypatch.setenv('RANK', '0')
  with pytest.raises(RuntimeError, match='setup failed'):
    with StartupMonitor(tmp_path, enabled=False) as monitor:
      monitor.set_phase('preparing_distributed_model')
      raise RuntimeError('setup failed')
  record = json.loads((tmp_path / 'startup.json').read_text())
  assert record['event'] == 'failed' and record['phase'] == 'preparing_distributed_model'
  assert record['error_type'] == 'RuntimeError'
  assert 'process_rss_bytes' in record['host_memory']
  assert monitor.thread is None and monitor.stop_event.is_set()


def test_startup_hands_off_without_overwriting_training_progress(tmp_path, monkeypatch):
  monkeypatch.setenv('RANK', '0')
  with StartupMonitor(tmp_path, enabled=False) as monitor:
    monitor.set_phase('preparing_distributed_model')
    monitor.finish()
    assert not monitor.active and monitor.thread is None
    record = (tmp_path / 'startup.json').read_text()
  assert (tmp_path / 'startup.json').read_text() == record
  assert json.loads(record)['event'] == 'complete'
  assert not (tmp_path / 'progress.json').exists()


def test_nonzero_rank_does_not_write_startup_files(tmp_path, monkeypatch):
  monkeypatch.setenv('RANK', '1')
  with StartupMonitor(tmp_path, enabled=False) as monitor:
    monitor.set_phase('loading_base_and_adapter')
  assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('pin_memory', [True, False])
def test_cpu_offload_policy_applied_before_preparation(config, pin_memory):
  from torch.distributed.fsdp import CPUOffloadPolicy
  plugin = SimpleNamespace(fsdp_version=2, cpu_offload=CPUOffloadPolicy())
  trainer = SimpleNamespace(accelerator=SimpleNamespace(state=SimpleNamespace(fsdp_plugin=plugin)))
  previous_identity = config.fingerprint()
  config.memory.offload = 'cpu'
  pinned_identity = config.fingerprint()
  config.memory.cpu_offload_pin_memory = pin_memory
  configure_cpu_offload(trainer, config)
  assert isinstance(plugin.cpu_offload, CPUOffloadPolicy)
  assert plugin.cpu_offload.pin_memory is pin_memory
  if not pin_memory:
    assert config.fingerprint() != pinned_identity
  config.memory.offload = 'none'
  config.memory.cpu_offload_pin_memory = True
  assert config.fingerprint() == previous_identity


def test_cpu_offload_requires_matching_plugin(config):
  config.memory.offload = 'cpu'
  trainer = SimpleNamespace(accelerator=SimpleNamespace(state=SimpleNamespace(fsdp_plugin=None)))
  with pytest.raises(RuntimeError, match='FSDP2'):
    configure_cpu_offload(trainer, config)
