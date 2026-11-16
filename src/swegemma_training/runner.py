"""Run lifecycle, TRL integration, and portable adapter artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid

import yaml

from .artifacts import check_resume, export_adapter, latest_checkpoint, latest_release, publish, write_json
from .config import Config, digest
from .harness import read_harness, snapshot_harness
from .resources import StartupMonitor, host_memory


def versions() -> dict:
  result = {}
  for name in ('torch', 'transformers', 'trl', 'peft', 'accelerate', 'compressed-tensors'):
    try:
      result[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
      result[name] = None
  return result


def doctor(config: Config, *, smoke=False) -> dict:
  contract = read_harness(config)
  report = {'versions': versions(), 'runtime': {k: contract[k] for k in (
      'context_length', 'completion_tokens', 'thinking_tokens')},
      'offload': config.memory.offload, 'processes': config.memory.processes,
      'bitsandbytes_used': False, 'gpu_training_verified': False}
  report.update(sequence_mode=config.data.sequence_mode, objective=config.objective.name,
                loss_normalization='optimizer_step_supervised_tokens' if config.token_weighted else 'microbatch_mean')
  try:
    import torch
    report['backend'] = 'rocm' if torch.version.hip else 'cuda'
    report['devices'] = [{'name': torch.cuda.get_device_name(i),
                           'vram_bytes': torch.cuda.get_device_properties(i).total_memory}
                          for i in range(torch.cuda.device_count())]
  except Exception as error:
    report['devices'] = []
    report['device_error'] = str(error)
  report['ready_for_training'] = all(report['versions'].values()) and len(report['devices']) >= config.memory.processes
  report['initial_host_weight_estimate_bytes'] = 24_000_000_000 * config.memory.processes
  report['host_memory'] = host_memory()
  report['cpu_offload_pin_memory'] = config.memory.cpu_offload_pin_memory if config.memory.offload == 'cpu' else None
  report['host_loading_note'] = (
      'Each worker initially dequantizes a full text base on CPU. FSDP setup can temporarily retain '
      'the original weights alongside the offloaded weights; the initial estimate is not peak RAM. '
      'Pinned offload weights cannot be swapped out. Allow RAM for the desktop and other processes.')
  if smoke:
    if not report['devices']:
      raise RuntimeError('No accessible GPU for a model smoke test')
    if config.memory.offload == 'cpu':
      raise ValueError('CPU offload uses distributed training hooks. Use train for this profile; doctor --smoke is GPU-only.')
    from .modeling import attach_adapter, load_base
    model, ranks = attach_adapter(load_base(config), config)
    model.to('cuda').train()
    if config.memory.activation_checkpointing:
      model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    ids = torch.tensor([[2, 42, 43, 44]], device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
      loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    if not torch.isfinite(loss):
      raise RuntimeError('Nonfinite smoke loss')
    report.update(smoke_loss=float(loss), adapter_parameters=sum(r['parameters'] for r in ranks),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  smoke_tokens=4, full_context_verified=False)
  return report


def load_prepared(config: Config) -> dict:
  manifest = json.loads((config.data.prepared / 'manifest.json').read_text())
  original = manifest.copy()
  fingerprint = original.pop('fingerprint')
  if digest(original) != fingerprint:
    raise ValueError('Prepared manifest changed')
  if read_harness(config)['fingerprint'] != manifest['harness']['fingerprint']:
    raise ValueError('Harness changed since preparation; prepare a new dataset')
  if config.model.source != manifest['base_source']:
    raise ValueError('Student source changed since tokenization')
  if config.model.revision not in {manifest['requested_base_revision'], manifest['base_revision']}:
    raise ValueError('Student revision changed since tokenization')
  for split, checksum in manifest['dataset_hashes'].items():
    if digest((config.data.prepared / f'{split}.jsonl').read_text()) != checksum:
      raise ValueError(f'Prepared {split} data changed')
  if set(manifest['splits']['train']) & set(manifest['splits']['validation']):
    raise ValueError('Task IDs overlap between splits')
  from .sequences import sequence_contract
  expected = sequence_contract(config)
  if manifest.get('sequence', {'mode': 'turn', 'window': None, 'mask_version': 1}) != expected:
    raise ValueError('Sequence construction changed; prepare a new dataset')
  if config.data.format == 'harness_jsonl':
    from .harness_jsonl import preparation_identity
    if manifest.get('source_preparation') != preparation_identity(config):
      raise ValueError('JSONL source files or prompt override changed; prepare a new dataset')
  elif manifest.get('source_format', 'atif') != 'atif':
    raise ValueError('Prepared source format differs from data.format')
  return manifest


def launch(config: Config, resume: str | None = None) -> Path:
  manifest = load_prepared(config)
  if config.objective.name == 'soft':
    from .reference import validate_cache
    validate_cache(config, manifest)
  if resume and config.adapter.warm_start:
    raise ValueError('Choose checkpoint resume or adapter.warm_start, not both')
  checkpoint = latest_checkpoint(config.output.directory) if resume == 'latest' else (
      Path(resume).expanduser().resolve() if resume else None
  )
  if checkpoint:
    run = checkpoint.parents[1]
  else:
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    run = config.output.directory / name
    run.mkdir(parents=True, exist_ok=False)
    (run / 'resolved.yaml').write_text(yaml.safe_dump(config.model_dump(mode='json'), sort_keys=False))
    snapshot_harness(manifest['harness'], run / 'harness')
    write_json(run / 'data-manifest.json', manifest)
  command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
             '--nproc_per_node', str(config.memory.processes),
             '--log-dir', str(run / 'worker-logs'), '--tee', '3', '-m',
             'swegemma_training.worker', '--config', str(run / 'resolved.yaml'),
             '--run', str(run)]
  # Resume overrides must be validated, not silently replaced by the old YAML.
  if checkpoint:
    candidate = run / f'resume-{uuid.uuid4().hex[:8]}.yaml'
    candidate.write_text(yaml.safe_dump(config.model_dump(mode='json'), sort_keys=False))
    command[command.index('--config') + 1] = str(candidate)
    command += ['--resume', str(checkpoint)]
  environment = os.environ.copy()
  for key in ('PYTHONFAULTHANDLER', 'PYTHONUNBUFFERED', 'TORCH_SHOW_CPP_STACKTRACES'):
    environment.setdefault(key, '1')
  try:
    subprocess.run(command, check=True, env=environment)
  except subprocess.CalledProcessError as error:
    checkpoints = sorted(p.parent for p in (run / 'checkpoints').glob('*/complete.json'))
    write_json(run / 'launcher-failure.json', {
        'torchrun_exit_code': error.returncode,
        'worker_logs': str(run / 'worker-logs'),
        'complete_checkpoints': [str(path) for path in checkpoints],
        'note': 'For a native abort, inspect worker stderr and the matching kernel log; Python exception handling may not run.',
    })
    print(f'Worker logs: {run / "worker-logs"}', file=sys.stderr, flush=True)
    if checkpoints:
      print(f'{len(checkpoints)} complete checkpoint(s) remain in this run.', file=sys.stderr, flush=True)
    else:
      print('No complete checkpoint was saved in this run; its updates cannot be resumed.', file=sys.stderr, flush=True)
    raise
  completed = json.loads((run / 'completed.json').read_text())
  # Worker processes exit first, releasing GPU allocations before network I/O.
  if not completed['interrupted']:
    for target in ('hf', 'kaggle'):
      if getattr(config.publish, target).enabled:
        try:
          publish(config, target, latest_release(run))
        except Exception as error:
          print(f'{target} upload failed ({type(error).__name__}); local release is saved. Retry with publish.', flush=True)
  return run


def callback_type():
  import torch
  from transformers import TrainerCallback

  class RunCallback(TrainerCallback):
    def __init__(self, run, identity, *, progress_enabled=True, heartbeat_seconds=30, startup=None):
      self.run, self.identity = Path(run), identity
      self.stopping = False
      self.handlers = {}
      self.started = time.monotonic()
      self.trainer = None
      self.resume_tokens = 0
      self.completed_microbatches = 0
      self.progress_enabled = progress_enabled
      self.heartbeat_seconds = heartbeat_seconds
      self.heartbeat_stop = threading.Event()
      self.heartbeat_thread = None
      self.status_lock = threading.Lock()
      self.startup = startup

    def report_progress(self, args, state, event):
      if not state.is_world_process_zero:
        return
      with self.status_lock:
        now = time.monotonic()
        phase = getattr(self.trainer, '_swe_phase', 'waiting_for_batch')
        batch_started = getattr(self.trainer, '_swe_microbatch_started', None) if phase in {
            'forward_and_loss', 'backward'
        } else None
        phase_started = getattr(self.trainer, '_swe_phase_started', self.started)
        record = {'event': event, 'optimizer_step': state.global_step,
                  'total_optimizer_steps': state.max_steps,
                  'completed_microbatches_this_session': self.completed_microbatches,
                  'gradient_accumulation': args.gradient_accumulation_steps,
                  'phase': phase,
                  'phase_elapsed_seconds': now - phase_started,
                  'sequence_tokens': getattr(self.trainer, '_swe_sequence_tokens', None),
                  'microbatch_elapsed_seconds': now - batch_started if batch_started else None,
                  'session_elapsed_seconds': now - self.started, 'host_memory': host_memory()}
        record.update({key: self.identity[key] for key in (
            'sequence_mode', 'objective', 'loss_normalization', 'reference_cache') if key in self.identity})
        write_json(self.run / 'progress.json', record)
        if self.progress_enabled:
          from tqdm.auto import tqdm
          duration = f'; current microbatch {record["microbatch_elapsed_seconds"]:.1f}s' if batch_started else ''
          tqdm.write(f'[training] {event}: step {state.global_step}/{state.max_steps}; '
                      f'{self.completed_microbatches} microbatches complete; '
                      f'phase={phase} ({record["phase_elapsed_seconds"]:.1f}s); '
                      f'sequence_tokens={record["sequence_tokens"]}{duration}')
          sys.stdout.flush()

    def supervised_tokens(self, args):
      value = getattr(self.trainer, 'supervised_tokens', 0)
      counter = torch.tensor(value, device=args.device, dtype=torch.long)
      if torch.distributed.is_initialized():
        torch.distributed.all_reduce(counter)
      return counter.item() + self.resume_tokens

    def on_train_begin(self, args, state, control, **kwargs):
      if self.startup is not None:
        self.startup.finish()
      def stop(signum, frame):
        self.stopping = True
      for name in (signal.SIGINT, signal.SIGTERM):
        self.handlers[name] = signal.signal(name, stop)
      self.report_progress(args, state, 'started')
      if state.is_world_process_zero:
        def heartbeat():
          while not self.heartbeat_stop.wait(self.heartbeat_seconds):
            self.report_progress(args, state, 'heartbeat')
        self.heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        self.heartbeat_thread.start()

    def restore(self):
      self.heartbeat_stop.set()
      if self.heartbeat_thread is not None:
        self.heartbeat_thread.join(timeout=1)
        self.heartbeat_thread = None
      for name, handler in self.handlers.items():
        signal.signal(name, handler)
      self.handlers = {}

    def on_substep_end(self, args, state, control, **kwargs):
      self.completed_microbatches += 1
      self.trainer._swe_phase = 'between_microbatches'
      self.trainer._swe_phase_started = time.monotonic()
      self.report_progress(args, state, 'microbatch_complete')

    def on_step_end(self, args, state, control, **kwargs):
      self.completed_microbatches += 1
      self.trainer._swe_phase = 'optimizer_step_complete'
      self.trainer._swe_phase_started = time.monotonic()
      self.report_progress(args, state, 'optimizer_step_complete')
      stopping = torch.tensor(int(self.stopping), device=args.device)
      if torch.distributed.is_initialized():
        torch.distributed.all_reduce(stopping, op=torch.distributed.ReduceOp.MAX)
      if stopping.item():
        self.stopping = True
        control.should_save = True
        control.should_training_stop = True
      # Always create a resumable checkpoint at the last scheduled step too.
      if state.global_step >= state.max_steps:
        control.should_save = True
      return control

    def on_save(self, args, state, control, **kwargs):
      tokens = self.supervised_tokens(args)
      if torch.distributed.is_initialized():
        torch.distributed.barrier()
      if state.is_world_process_zero:
        path = Path(args.output_dir) / f'checkpoint-{state.global_step}'
        write_json(path / 'complete.json', {'identity': self.identity,
                                           'step': state.global_step,
                                           'supervised_tokens': tokens})
      self.trainer._swe_phase = 'checkpoint_complete'
      self.trainer._swe_phase_started = time.monotonic()
      self.report_progress(args, state, 'checkpoint_complete')

    def on_log(self, args, state, control, logs=None, **kwargs):
      from numbers import Real
      from .numerics import require_finite
      require_finite(((key, value) for key, value in (logs or {}).items()
                      if isinstance(value, Real) or torch.is_tensor(value)),
                     'Logged metrics', device=args.device)
      tokens = self.supervised_tokens(args)
      if state.is_world_process_zero:
        record = dict(logs or {}, step=state.global_step,
                      elapsed_seconds=time.monotonic() - self.started,
                      supervised_tokens=tokens,
                      supervised_tokens_per_second=(tokens - self.resume_tokens) / max(0.001, time.monotonic() - self.started))
        if torch.cuda.is_available():
          record.update(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        with (self.run / 'metrics.jsonl').open('a') as handle:
          handle.write(json.dumps(record) + '\n')

    def on_train_end(self, args, state, control, **kwargs):
      self.restore()
      self.trainer._swe_phase = 'interrupted' if self.stopping else 'training_complete'
      self.report_progress(args, state, self.trainer._swe_phase)

  return RunCallback


def trainer_type(base):
  from transformers import Trainer

  class HardDistillationTrainer(base):
    supervised_tokens = 0

    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      clip = self.accelerator.clip_grad_norm_

      def checked_clip(*clip_args, **clip_kwargs):
        from .numerics import require_finite
        require_finite(((name, p.grad) for name, p in self.model.named_parameters() if p.requires_grad),
                       'Before gradient clipping', device=self.args.device)
        norm = clip(*clip_args, **clip_kwargs)
        require_finite([('grad_norm', norm)], 'Gradient clipping', device=self.args.device)
        return norm

      self.accelerator.clip_grad_norm_ = checked_clip

    def create_optimizer(self):
      optimizer = super().create_optimizer()
      if not getattr(optimizer, '_swe_numerical_hooks', False):
        from .numerics import require_finite

        def before_update(opt, args, kwargs):
          require_finite(((name, p.grad) for name, p in self.model.named_parameters()
                          if p.requires_grad), 'Before optimizer update', device=self.args.device)

        def after_update(opt, args, kwargs):
          require_finite(((name, p) for name, p in self.model.named_parameters()
                          if p.requires_grad), 'After optimizer update', device=self.args.device)
          self._check_optimizer_state('Optimizer state after update', opt)

        optimizer.register_step_pre_hook(before_update)
        optimizer.register_step_post_hook(after_update)
        optimizer._swe_numerical_hooks = True
      return optimizer

    def _check_optimizer_state(self, stage, optimizer=None):
      import torch
      from .numerics import require_finite
      optimizer = optimizer if optimizer is not None else self.optimizer
      if optimizer is not None:
        require_finite(((f'optimizer.{index}.{key}', value)
                        for index, state in enumerate(optimizer.state.values())
                        for key, value in state.items() if torch.is_tensor(value)),
                       stage, device=self.args.device)

    def _load_optimizer_and_scheduler(self, checkpoint):
      result = super()._load_optimizer_and_scheduler(checkpoint)
      from .numerics import require_finite
      require_finite(((name, p) for name, p in self.model.named_parameters() if p.requires_grad),
                     'Restored adapter weights', device=self.args.device)
      self._check_optimizer_state('Restored optimizer state')
      return result

    def save_model(self, output_dir=None, _internal_call=False):
      from .numerics import require_finite
      require_finite(((name, p) for name, p in self.model.named_parameters() if p.requires_grad),
                     'Before model save', device=self.args.device)
      self._swe_phase = 'saving_model'
      self._swe_phase_started = time.monotonic()
      if getattr(self.accelerator, 'is_fsdp2', False):
        import torch
        from peft import PeftModel
        from transformers.trainer import TRAINING_ARGS_NAME
        from .modeling import save_lora_adapter, trainable_state_dict
        if not isinstance(self.model, PeftModel):
          raise ValueError('This FSDP2 save path requires a PEFT adapter model')
        # All ranks must participate in collecting adapter shards. Avoid the
        # generic Accelerator.get_state_dict path, which gathers the frozen base.
        state_dict = trainable_state_dict(self.model)
        if self.args.should_save:
          destination = Path(output_dir or self.args.output_dir)
          # PEFT 0.21 selects adapter keys using named_modules(), whose names
          # still include activation checkpoint wrappers. Write our restricted
          # projection LoRA format directly to avoid an empty PEFT export.
          save_lora_adapter(self.model, state_dict, destination)
          if self.processing_class is not None:
            self.processing_class.save_pretrained(destination)
          torch.save(self.args, destination / TRAINING_ARGS_NAME)
          if _internal_call:
            from accelerate.utils.constants import FSDP_MODEL_NAME
            # Keep Accelerate's native resume format, with adapter parameters
            # only. The immutable base is reconstructed before checkpoint load.
            torch.save(state_dict, destination / f'{FSDP_MODEL_NAME}.bin')
      else:
        super().save_model(output_dir, _internal_call=_internal_call)

    def _save_optimizer_and_scheduler(self, output_dir):
      self._check_optimizer_state('Before optimizer save')
      self._swe_phase = 'saving_optimizer_and_scheduler'
      self._swe_phase_started = time.monotonic()
      if getattr(self.accelerator, 'is_fsdp2', False):
        import torch
        from accelerate.utils import save_fsdp_optimizer
        from transformers.trainer import SCHEDULER_NAME
        # Trainer's generic FSDP branch saves model weights again. Our model
        # and native adapter checkpoint were already written by save_model().
        save_fsdp_optimizer(self.accelerator.state.fsdp_plugin, self.accelerator,
                            self.optimizer, self.model, output_dir)
        if self.args.should_save:
          torch.save(self.lr_scheduler.state_dict(), Path(output_dir) / SCHEDULER_NAME)
        return
      return super()._save_optimizer_and_scheduler(output_dir)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
      # TRL's default additional entropy/accuracy metrics require dense logits.
      # Keep its dataset/PEFT integration and use HF's public loss path instead.
      if model.training:
        self._swe_phase = 'forward_and_loss'
        self._swe_phase_started = time.monotonic()
        self._swe_microbatch_started = time.monotonic()
        self._swe_sequence_tokens = inputs['input_ids'].shape[-1]
        self.supervised_tokens += inputs['labels'][:, 1:].ne(-100).sum().item()
      result = Trainer.compute_loss(self, model, inputs, return_outputs=return_outputs,
                                    num_items_in_batch=num_items_in_batch)
      from .numerics import require_finite
      loss = result[0] if return_outputs else result
      require_finite([('loss', loss)], 'Training loss' if model.training else 'Evaluation loss',
                     device=self.args.device)
      if model.training:
        self._swe_phase = 'backward'
        self._swe_phase_started = time.monotonic()
      return result

  return HardDistillationTrainer


def sft_config_kwargs(config: Config, run: Path, world: int) -> dict:
  """Translate recipe settings to the supported TRL training arguments."""
  fsdp = world > 1 or config.memory.offload == 'cpu'
  return dict(
      output_dir=str(run / 'checkpoints'),
      num_train_epochs=config.training.epochs,
      per_device_train_batch_size=config.training.microbatch_size,
      per_device_eval_batch_size=1,
      gradient_accumulation_steps=config.training.gradient_accumulation,
      learning_rate=config.optimizer.learning_rate, weight_decay=config.optimizer.weight_decay,
      max_grad_norm=config.optimizer.max_grad_norm, lr_scheduler_type=config.optimizer.scheduler,
      warmup_steps=config.optimizer.warmup_ratio, optim='adamw_torch',
      bf16=True, gradient_checkpointing=config.memory.activation_checkpointing and not fsdp,
      gradient_checkpointing_kwargs={'use_reentrant': False},
      fsdp='full_shard auto_wrap' if fsdp else '',
      fsdp_config={'version': 2, 'cpu_offload': config.memory.offload == 'cpu',
                   'transformer_layer_cls_to_wrap': ['Gemma4UnifiedTextDecoderLayer'],
                   'activation_checkpointing': config.memory.activation_checkpointing,
                   'cpu_ram_efficient_loading': False, 'state_dict_type': 'FULL_STATE_DICT'} if fsdp else None,
      max_length=config.runtime.context_length, packing=False, padding_free=False,
      dataset_kwargs={'skip_prepare_dataset': True},
      prediction_loss_only=True, remove_unused_columns=False,
      eval_strategy='steps', eval_steps=config.training.eval_steps,
      save_strategy='steps', save_steps=config.training.save_steps,
      save_total_limit=config.training.save_total_limit,
      logging_steps=config.training.logging_steps,
      logging_first_step=True, logging_nan_inf_filter=False, disable_tqdm=not config.progress.enabled,
      report_to=['tensorboard'] if config.training.tensorboard else [],
      include_num_input_tokens_seen='all', seed=config.training.seed,
      dataloader_num_workers=0, average_tokens_across_devices=config.token_weighted,
  )


def configure_cpu_offload(trainer, config: Config):
  """Apply the FSDP2 public policy before Accelerate prepares any parameters."""
  if config.memory.offload != 'cpu':
    return
  from torch.distributed.fsdp import CPUOffloadPolicy
  plugin = getattr(trainer.accelerator.state, 'fsdp_plugin', None)
  if plugin is None or plugin.fsdp_version != 2:
    raise RuntimeError('CPU offload requires an FSDP2 plugin before model preparation')
  plugin.cpu_offload = CPUOffloadPolicy(pin_memory=config.memory.cpu_offload_pin_memory)


def train_worker(config: Config, run: Path, resume: Path | None = None):
  with StartupMonitor(run, enabled=config.progress.enabled,
                      heartbeat_seconds=config.progress.heartbeat_seconds) as startup:
    return _train_worker(config, run, resume, startup)


def _train_worker(config: Config, run: Path, resume: Path | None, startup):
  import torch
  from transformers import AutoTokenizer, set_seed
  try:
    from trl import SFTConfig, SFTTrainer
  except ImportError as error:
    raise RuntimeError('Install the training[train] dependencies, including TRL, in this interpreter') from error
  from .modeling import Collator, attach_adapter, load_base

  startup.set_phase('validating_prepared_data')
  manifest = load_prepared(config)
  cache = None
  if config.objective.name == 'soft':
    from .reference import validate_cache
    cache = validate_cache(config, manifest)
  local_rank = int(os.environ.get('LOCAL_RANK', '0'))
  rank = int(os.environ.get('RANK', '0'))
  world = int(os.environ.get('WORLD_SIZE', '1'))
  if not torch.cuda.is_available():
    raise RuntimeError('No accessible CUDA/ROCm GPU. Run swe-train doctor first.')
  torch.cuda.set_device(local_rank)
  if not torch.cuda.is_bf16_supported():
    raise RuntimeError('This training profile requires BF16 support')
  if world != config.memory.processes:
    raise ValueError('Launched process count does not match memory.processes')
  identity = {'config': config.fingerprint(), 'dataset': manifest['fingerprint'],
              'world_size': world, 'versions': versions()}
  if config.token_weighted:
    identity.update(sequence_mode=config.data.sequence_mode,
                    loss_normalization='optimizer_step_supervised_tokens', objective=config.objective.name)
  if cache:
    identity['reference_cache'] = cache['fingerprint']
  if resume:
    check_resume(resume, identity)
  elif config.adapter.warm_start:
    provenance = json.loads((config.adapter.warm_start / 'provenance.json').read_text())
    if provenance['base_source'] != config.model.source or provenance['base_revision'] != manifest['base_revision']:
      raise ValueError('Warm-start adapter base source/revision differs')
  # Validate trainer arguments before loading/decompressing the full checkpoint.
  args = SFTConfig(**sft_config_kwargs(config, run, world))
  set_seed(config.training.seed)
  if not config.progress.enabled or rank != 0:
    from transformers.utils.logging import disable_progress_bar
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bar()
    disable_progress_bars()
  startup.set_phase('loading_tokenizer')
  tokenizer = AutoTokenizer.from_pretrained(config.model.source, revision=manifest['base_revision'],
                                            local_files_only=config.model.local_files_only)
  if digest({'vocab': tokenizer.get_vocab(), 'template': tokenizer.chat_template}) != manifest['tokenizer_sha256']:
    raise ValueError('Student tokenizer changed since preparation')
  if tokenizer.pad_token_id is None:
    raise ValueError('Tokenizer must define a padding token')
  # Convert one row at a time to memory-mapped Arrow before loading the dense base.
  # Full provenance/messages remain in the immutable JSONL rather than Python RAM.
  startup.set_phase('loading_prepared_tokens')
  from .prepared import prepared_datasets
  datasets = prepared_datasets(config, manifest)
  # Pin base loading to the revision used for tokenization, not a moving main.
  config.model.revision = manifest['base_revision']
  startup.set_phase('loading_base_and_adapter')
  model, ranks = attach_adapter(load_base(config), config)
  if rank == 0:
    write_json(run / 'environment.json', doctor(config))
    write_json(run / 'adapter-parameters.json', ranks)
    print(f'Adapter parameters: {sum(r["parameters"] for r in ranks):,}', flush=True)
  callback = callback_type()(run, identity, progress_enabled=config.progress.enabled,
                              heartbeat_seconds=config.progress.heartbeat_seconds, startup=startup)
  startup.set_phase('creating_trainer')
  trainer = trainer_type(SFTTrainer)(
      model=model, args=args, processing_class=tokenizer,
      train_dataset=datasets['train'], eval_dataset=datasets['validation'],
      data_collator=Collator(tokenizer.pad_token_id, config.objective.soft.cache_dir if cache else None),
      callbacks=[callback],
  )
  configure_cpu_offload(trainer, config)
  trainer.model_accepts_loss_kwargs = config.token_weighted
  callback.trainer = trainer
  if resume:
    callback.resume_tokens = json.loads((resume / 'complete.json').read_text()).get('supervised_tokens', 0)
  try:
    startup.set_phase('preparing_distributed_model')
    result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    evaluation = {} if callback.stopping else trainer.evaluate()
    trainer.save_model(str(run / 'adapter'))
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
      provenance = {'schema_version': 1, 'stage': 'sft', 'base_source': config.model.source,
                    'sequence_mode': config.data.sequence_mode, 'objective': config.objective.name,
                    'sequence': manifest.get('sequence', {'mode': 'turn', 'window': None, 'mask_version': 1}),
                    'objective_parameters': config.objective.soft.model_dump(exclude={'cache_dir'}) if cache else None,
                    'loss_normalization': 'optimizer_step_supervised_tokens' if config.token_weighted else 'microbatch_mean',
                    'reference_cache': cache['fingerprint'] if cache else None,
                    'source_format': manifest.get('source_format', 'atif'),
                    'training_prompts': manifest.get('training_prompts'),
                    'verification_policy': manifest.get('verification_policy', 'resolved_and_exit_zero'),
                    'adapter_layout': getattr(model.get_base_model(), '_swe_source_model_type', 'gemma4_unified_text'),
                    'base_revision': manifest['base_revision'], 'identity': identity,
                    'harness_fingerprint': manifest['harness']['fingerprint'],
                    'runtime': {k: manifest['harness'][k] for k in (
                        'context_length', 'completion_tokens', 'thinking_tokens')},
                    'interrupted': callback.stopping, 'metrics': result.metrics | evaluation}
      write_json(run / 'adapter/provenance.json', provenance)
      release = export_adapter(run / 'adapter', run / 'releases' /
                               f'step-{trainer.state.global_step}-{uuid.uuid4().hex[:8]}', provenance)
      write_json(run / 'latest-release.json', {'path': str(release.relative_to(run))})
      write_json(run / 'completed.json', provenance)
    trainer.accelerator.wait_for_everyone()
  except Exception as error:
    trainer._swe_phase = 'failed'
    trainer._swe_phase_started = time.monotonic()
    callback.report_progress(args, trainer.state, 'failed')
    if rank == 0:
      write_json(run / 'failure.json', {'error_type': type(error).__name__,
                 'message': str(error), 'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                 'optimizer_step': trainer.state.global_step,
                 'completed_microbatches': callback.completed_microbatches,
                 'sequence_tokens': getattr(trainer, '_swe_sequence_tokens', None),
                 'note': 'Context and offload settings were not changed automatically.'})
    raise
  finally:
    callback.restore()
