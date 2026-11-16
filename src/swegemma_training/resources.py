"""Host memory snapshots and startup progress without querying the GPU driver."""

import json
import os
from pathlib import Path
import threading
import time

from .artifacts import write_json


def host_memory(proc: Path = Path('/proc')) -> dict:
  result = {}
  for source, names in (
      ('self/status', {'VmRSS': 'process_rss_bytes', 'VmHWM': 'process_peak_rss_bytes',
                       'VmSwap': 'process_swap_bytes', 'VmLck': 'process_locked_bytes',
                       'VmPin': 'process_pinned_bytes'}),
      ('meminfo', {'MemTotal': 'system_total_bytes', 'MemAvailable': 'system_available_bytes',
                   'Shmem': 'system_shared_bytes', 'SwapTotal': 'system_swap_total_bytes',
                   'SwapFree': 'system_swap_free_bytes'}),
  ):
    try:
      for line in (proc / source).read_text().splitlines():
        key, _, value = line.partition(':')
        if key in names:
          result[names[key]] = int(value.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
      pass
  return result


class StartupMonitor:
  """Keep evidence of initialization stalls, including FSDP setup before callbacks."""

  def __init__(self, run, *, enabled=True, heartbeat_seconds=30):
    self.run = Path(run)
    self.enabled = enabled
    self.heartbeat_seconds = heartbeat_seconds
    self.started = time.monotonic()
    self.phase_started = self.started
    self.phase = 'initializing'
    self.active = int(os.environ.get('RANK', '0')) == 0
    self.stop_event = threading.Event()
    self.lock = threading.Lock()
    self.thread = None

  def __enter__(self):
    if self.active:
      self.set_phase(self.phase)
      self.thread = threading.Thread(target=self._heartbeat, daemon=True)
      self.thread.start()
    return self

  def _heartbeat(self):
    while not self.stop_event.wait(self.heartbeat_seconds):
      with self.lock:
        self._report('heartbeat')

  def _report(self, event, **extra):
    record = dict(event=event, phase=self.phase, pid=os.getpid(),
                  phase_elapsed_seconds=time.monotonic() - self.phase_started,
                  startup_elapsed_seconds=time.monotonic() - self.started,
                  host_memory=host_memory(), **extra)
    write_json(self.run / 'startup.json', record)
    with (self.run / 'startup.jsonl').open('a') as handle:
      handle.write(json.dumps(record) + '\n')
    if self.enabled:
      memory = record['host_memory']
      details = '; '.join(f'{label}={memory[key] / 2**30:.1f} GiB' for key, label in (
          ('process_rss_bytes', 'RSS'), ('system_available_bytes', 'host available'),
          ('system_shared_bytes', 'host shared')) if key in memory)
      print(f'[startup] {self.phase} ({record["phase_elapsed_seconds"]:.1f}s); {details}', flush=True)

  def set_phase(self, phase):
    if self.active:
      with self.lock:
        self.phase = phase
        self.phase_started = time.monotonic()
        self._report('phase')

  def finish(self):
    if self.active:
      self.stop_event.set()
      if self.thread is not None:
        self.thread.join()
        self.thread = None
      with self.lock:
        self._report('complete')
      self.active = False

  def __exit__(self, error_type, error, traceback):
    if self.active:
      self.stop_event.set()
      if self.thread is not None:
        self.thread.join()
        self.thread = None
      with self.lock:
        self._report('failed' if error else 'complete',
                     **({'error_type': error_type.__name__, 'message': str(error)} if error else {}))
      self.active = False
