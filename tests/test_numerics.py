import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from swegemma_training.artifacts import check_resume, export_adapter, file_sha256, publish, write_json
from swegemma_training.config import Config, digest
from swegemma_training.data import prepare
from swegemma_training.modeling import Collator, attach_adapter, training_model_class
from swegemma_training.numerics import check_adapter_file, require_finite
from swegemma_training.objectives import construct_targets, probability_floor, reference_support, sparse_soft_loss
from swegemma_training.reference import build_cache, prepared_rows, validate_cache
from swegemma_training.runner import sft_config_kwargs, trainer_type
from test_training import tiny_config


@pytest.fixture(params=[(mode, objective) for mode in ('turn', 'trajectory', 'windowed')
                       for objective in ('sft', 'soft')], ids=lambda pair: '-'.join(pair))
def regime(request, traces, tokenizer, tmp_path):
  """Exercise prepared sequence masks, real TRL, and both model loss paths."""
  from datasets import Dataset
  from trl import SFTConfig, SFTTrainer
  traces.data.sequence_mode, traces.objective.name = request.param
  traces.data.window.max_tokens = 256
  traces.objective.soft.cache_dir = tmp_path / 'reference'
  manifest = prepare(traces, tokenizer, schemas=[])
  cache_dir = None
  if traces.objective.name == 'soft':
    def predict(row):
      gold = torch.tensor([v for v in row['labels'][1:] if v != -100])
      return reference_support(torch.zeros(len(gold), 256), gold, 32)
    build_cache(traces, manifest, predict)
    cache_dir = traces.objective.soft.cache_dir
  tiny = tiny_config()
  tiny.max_position_embeddings = 4096
  model = training_model_class()(tiny)
  model.token_weighted = traces.token_weighted
  model.loss_chunk_tokens = 16
  model.requires_grad_(False)
  model, _ = attach_adapter(model, traces)
  rows = prepared_rows(traces)['train']
  rows = [{k: row[k] for k in ('input_ids', 'attention_mask', 'labels', 'example_id')} for row in rows]
  args = SFTConfig(output_dir=str(tmp_path / 'checkpoints'), use_cpu=True, bf16=False,
      max_steps=2, save_steps=1, max_grad_norm=1, report_to=[], disable_tqdm=True,
      per_device_train_batch_size=1, gradient_accumulation_steps=2,
      dataset_kwargs={'skip_prepare_dataset': True}, remove_unused_columns=False,
      average_tokens_across_devices=traces.token_weighted, logging_nan_inf_filter=False)
  trainer = trainer_type(SFTTrainer)(model=model, args=args, train_dataset=Dataset.from_list(rows),
      eval_dataset=Dataset.from_list(rows), processing_class=tokenizer, data_collator=Collator(0, cache_dir))
  trainer.model_accepts_loss_kwargs = traces.token_weighted
  return traces, trainer, model


@pytest.mark.parametrize('failure', ['nan_loss', 'inf_loss', 'nan_gradient', 'inf_gradient', 'overflowing_norm'])
def test_nonfinite_training_stops_before_first_optimizer_update(regime, tmp_path, failure):
  _, trainer, model = regime
  original = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
  if failure.endswith('loss'):
    def corrupt_output(module, args, output):
      output.loss = output.loss * float('nan' if failure == 'nan_loss' else 'inf')
    model.register_forward_hook(corrupt_output)
  else:
    parameter = next(p for p in model.parameters() if p.requires_grad)
    value = 1e38 if failure == 'overflowing_norm' else float('nan' if failure == 'nan_gradient' else 'inf')
    parameter.register_hook(lambda grad: torch.full_like(grad, value))
    if failure != 'overflowing_norm':
      trainer.args.max_grad_norm = 0  # Also cover the optimizer guard without clipping.
  with pytest.raises(FloatingPointError, match='NaN/Inf'):
    trainer.train()
  assert trainer.state.global_step == 0
  assert not list(tmp_path.rglob('checkpoint-*'))
  for name, parameter in model.named_parameters():
    if parameter.requires_grad:
      torch.testing.assert_close(parameter, original[name], rtol=0, atol=0)


@pytest.mark.parametrize('corruption', ['parameter', 'state'])
def test_corrupt_optimizer_update_keeps_last_good_checkpoint(regime, tmp_path, corruption):
  _, trainer, model = regime

  class BrokenAdam(torch.optim.AdamW):
    updates = 0
    @torch.no_grad()
    def step(self, closure=None):
      result = super().step(closure)
      self.updates += 1
      if self.updates == 2:
        parameter = self.param_groups[0]['params'][0]
        tensor = parameter if corruption == 'parameter' else self.state[parameter]['exp_avg']
        tensor.fill_(float('nan'))
      return result

  trainer.optimizer = BrokenAdam([p for p in model.parameters() if p.requires_grad], lr=0.01)
  with pytest.raises(FloatingPointError, match='fter update' if corruption == 'state' else 'After optimizer update'):
    trainer.train()
  assert trainer.state.global_step == 1
  checkpoint = tmp_path / 'checkpoints/checkpoint-1'
  check_adapter_file(checkpoint / 'adapter_model.safetensors')
  assert not (checkpoint.parent / 'checkpoint-2').exists()


def test_finite_training_evaluation_and_export_for_each_regime(regime, tmp_path):
  config, trainer, _ = regime
  trainer.train()
  assert trainer.state.global_step == 2
  assert torch.isfinite(torch.tensor(trainer.evaluate()['eval_loss']))
  require_finite(((name, p) for name, p in trainer.model.named_parameters() if p.requires_grad), 'Test')
  source = tmp_path / 'checkpoints/checkpoint-2'
  release = export_adapter(source, tmp_path / 'release', {})
  check_adapter_file(release / 'adapter_model.safetensors')
  assert sft_config_kwargs(config, tmp_path, 1)['logging_nan_inf_filter'] is False


def test_nonfinite_evaluation_is_rejected(regime):
  _, trainer, model = regime
  def corrupt_output(module, args, output):
    if not module.training:
      output.loss = output.loss * float('nan')
  model.register_forward_hook(corrupt_output)
  with pytest.raises(FloatingPointError, match='Evaluation loss'):
    trainer.evaluate()


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_legacy_nonfinite_adapter_cannot_resume_export_or_publish(config, tmp_path, value):
  folder = tmp_path / 'bad'
  folder.mkdir()
  save_file({'lora_A.weight': torch.tensor([[value]])}, folder / 'adapter_model.safetensors')
  (folder / 'adapter_config.json').write_text('{}')
  (folder / 'complete.json').write_text(json.dumps({'identity': {'fixed': True}}))
  (folder / 'trainer_state.json').write_text('{}')
  with pytest.raises(FloatingPointError, match='Adapter'):
    check_adapter_file(folder / 'adapter_model.safetensors')
  with pytest.raises(FloatingPointError):
    check_resume(folder, {'fixed': True})
  with pytest.raises(FloatingPointError):
    export_adapter(folder, tmp_path / 'release', {})
  assert not (tmp_path / 'release').exists()
  config.adapter.warm_start = folder
  with pytest.raises(FloatingPointError):
    attach_adapter(training_model_class()(tiny_config()), config)
  config.publish.hf.enabled = True
  config.publish.hf.repo_id = 'owner/model'
  with pytest.raises(FloatingPointError):
    publish(config, 'hf', folder)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32, torch.float64])
def test_saved_tensor_check_needs_no_torch(dtype, tmp_path, monkeypatch):
  import builtins
  valid = tmp_path / 'valid.safetensors'
  invalid = tmp_path / 'invalid.safetensors'
  save_file({'weight': torch.tensor([0., -1., 1e-20], dtype=dtype)}, valid)
  save_file({'weight': torch.tensor([float('nan'), float('inf'), -float('inf')], dtype=dtype)}, invalid)
  original_import = builtins.__import__

  def no_torch(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
      raise AssertionError('Artifact inspection must not require Torch')
    return original_import(name, *args, **kwargs)

  monkeypatch.setattr(builtins, '__import__', no_torch)
  check_adapter_file(valid)
  with pytest.raises(FloatingPointError):
    check_adapter_file(invalid)


@pytest.mark.parametrize('existing_destination', [False, True])
def test_export_cli_does_not_reuse_invalid_release(config, tmp_path, existing_destination):
  import yaml
  from typer.testing import CliRunner
  from swegemma_training.cli import app
  run = tmp_path / 'run'
  source = run / 'adapter'
  release = run / 'releases/existing'
  source.mkdir(parents=True)
  release.mkdir(parents=True)
  write_json(source / 'provenance.json', {'test': True})
  write_json(release / 'provenance.json', {'test': True})
  write_json(run / 'latest-release.json', {'path': 'releases/existing'})
  save_file({'weight': torch.tensor([float('nan')])}, release / 'adapter_model.safetensors')
  config_path = tmp_path / 'config.yaml'
  config_path.write_text(yaml.safe_dump(config.model_dump(mode='json')))
  command = ['export', '--config', str(config_path), '--run', str(run)]
  if existing_destination:
    command += ['--destination', str(release)]
  result = CliRunner().invoke(app, command)
  assert isinstance(result.exception, FloatingPointError)


def test_finite_check_includes_other_rank_failures(monkeypatch):
  monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
  def remote_failure(flag, op):
    flag.fill_(1)
  monkeypatch.setattr(torch.distributed, 'all_reduce', remote_failure)
  with pytest.raises(FloatingPointError, match='another distributed rank'):
    require_finite([('healthy', torch.ones(2))], 'Guard', device='cpu')


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('field', ['learning_rate', 'weight_decay', 'max_grad_norm'])
def test_config_rejects_nonfinite_optimizer_values(field, value):
  with pytest.raises(ValueError):
    Config.model_validate({'optimizer': {field: value}})
  with pytest.raises(ValueError):
    Config.model_validate({'adapter': {'alpha_pattern': {'q_proj': value}}})


@pytest.mark.parametrize('rstar', [0.0, 0.5, 1.0])
@pytest.mark.parametrize('top_k', [1, 4])
def test_soft_saturated_probabilities_and_underflow_remain_finite(rstar, top_k):
  logits = torch.tensor([[10000., -10000., -10000., -10000.],
                         [-10000., 10000., -10000., -10000.],
                         [0., -80., -40., -20.]])
  gold = torch.tensor([0, 0, 1])  # Reference gold probabilities include exact 1 and 0.
  support = reference_support(logits, gold, top_k)
  floor = probability_floor(support['gold_probs'], rstar)
  q, tail = construct_targets(support, gold, floor)
  require_finite([('q', q), ('tail', tail)], 'Targets')
  torch.testing.assert_close(q.sum(-1) + tail, torch.ones(3))
  student = torch.tensor([[80., -80., 0., -40.]]*3, requires_grad=True)
  loss = sparse_soft_loss(student, support['ids'], q, tail)
  loss.backward()
  require_finite([('loss', loss), ('gradient', student.grad)], 'SoFT objective')
  if rstar == 1:
    torch.testing.assert_close(loss, torch.nn.functional.cross_entropy(student, gold, reduction='sum'))


def test_soft_already_certain_reference_has_no_division_by_zero():
  support = reference_support(torch.tensor([[10000., -10000.]]), torch.tensor([0]), 2)
  floor = probability_floor(support['gold_probs'], 0.5)
  q, tail = construct_targets(support, torch.tensor([0]), floor)
  torch.testing.assert_close(q, torch.tensor([[1., 0., 0.]]))
  assert tail.item() == 0


def test_nonfinite_metric_cannot_be_written_as_success(tmp_path):
  from types import SimpleNamespace
  from swegemma_training.runner import callback_type
  callback = callback_type()(tmp_path, {}, progress_enabled=False)
  with pytest.raises(FloatingPointError, match='Logged metrics'):
    callback.on_log(SimpleNamespace(device='cpu'), None, None, logs={'eval_loss': float('nan')})
  assert not (tmp_path / 'metrics.jsonl').exists()


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_reference_logits_and_probabilities_are_rejected(value):
  with pytest.raises(FloatingPointError, match='reference logits'):
    reference_support(torch.tensor([[0., value]]), torch.tensor([0]), 1)
  support = reference_support(torch.zeros(1, 2), torch.tensor([0]), 1)
  support['base_probs'][0, 0] = value
  with pytest.raises(FloatingPointError, match='reference probabilities'):
    construct_targets(support, torch.tensor([0]), 0.5)


@pytest.mark.parametrize('corruption', ['reference', 'cached_target', 'mass'])
def test_invalid_reference_or_cache_never_becomes_training_input(traces, tokenizer, tmp_path, corruption):
  traces.objective.name = 'soft'
  traces.objective.soft.cache_dir = tmp_path / 'reference'
  manifest = prepare(traces, tokenizer, schemas=[])

  def predict(row):
    gold = torch.tensor([v for v in row['labels'][1:] if v != -100])
    support = reference_support(torch.zeros(len(gold), 256), gold, 32)
    if corruption == 'reference':
      support['base_probs'][0, 0] = float('inf')
    return support

  if corruption == 'reference':
    with pytest.raises(FloatingPointError, match='SoFT reference'):
      build_cache(traces, manifest, predict)
    assert not traces.objective.soft.cache_dir.exists()
    return
  result = build_cache(traces, manifest, predict)
  name = next(iter(result['files']))
  shard = traces.objective.soft.cache_dir / name
  targets = load_file(shard)
  targets['soft_probs'][0, 0] = float('nan') if corruption == 'cached_target' else 0.9
  save_file(targets, shard)
  # Simulate an old cache with a valid checksum for numerically invalid data.
  result['files'][name] = file_sha256(shard)
  result.pop('fingerprint')
  result['fingerprint'] = digest(result)
  write_json(traces.objective.soft.cache_dir / 'manifest.json', result)
  error = FloatingPointError if corruption == 'cached_target' else ValueError
  with pytest.raises(error, match='SoFT cache|target mass'):
    validate_cache(traces, manifest)


def test_corrupt_restored_optimizer_is_rejected_before_forward(config, tmp_path):
  from transformers import Trainer, TrainingArguments
  rows = [{'input_ids': [2, 10, 11, 1], 'attention_mask': [1]*4, 'labels': [-100, -100, 11, 1]}]

  def make_trainer(destination):
    model = training_model_class()(tiny_config())
    model.requires_grad_(False)
    model, _ = attach_adapter(model, config)
    args = TrainingArguments(output_dir=str(destination), use_cpu=True, max_steps=2,
                             save_steps=1, report_to=[], disable_tqdm=True)
    trainer = trainer_type(Trainer)(model=model, args=args, train_dataset=rows, data_collator=Collator(0))
    trainer.model_accepts_loss_kwargs = False
    return trainer

  original = make_trainer(tmp_path / 'original')
  original.train()
  checkpoint = tmp_path / 'original/checkpoint-1'
  optimizer_path = checkpoint / 'optimizer.pt'
  state = torch.load(optimizer_path, weights_only=True)
  next(iter(state['state'].values()))['exp_avg'].fill_(float('inf'))
  torch.save(state, optimizer_path)
  restored = make_trainer(tmp_path / 'restored')
  with pytest.raises(FloatingPointError, match='Restored optimizer state'):
    restored.train(resume_from_checkpoint=str(checkpoint))
  assert restored.supervised_tokens == 0
  assert not list((tmp_path / 'restored').rglob('checkpoint-*'))


def distributed_finite_worker(rank, rendezvous):
  from datetime import timedelta
  from torch.distributed.device_mesh import init_device_mesh
  from torch.distributed.tensor import DTensor, Shard
  torch.distributed.init_process_group('gloo', rank=rank, world_size=2,
      init_method=f'file://{rendezvous}', timeout=timedelta(seconds=30))
  try:
    mesh = init_device_mesh('cpu', (2,))
    local = torch.tensor([1.0 if rank == 0 else float('nan')])
    shard = DTensor.from_local(local, mesh, [Shard(0)])
    with pytest.raises(FloatingPointError, match='NaN/Inf'):
      require_finite([('shard', shard)], 'Distributed test', device='cpu')
    require_finite([('healthy', torch.ones(1))], 'Healthy distributed test', device='cpu')
  finally:
    torch.distributed.destroy_process_group()


def test_two_rank_shard_failure_reaches_both_ranks(tmp_path, monkeypatch):
  import socket
  try:
    interfaces = {name for _, name in socket.if_nameindex()}
    with socket.socket() as probe:
      probe.bind(('127.0.0.1', 0))
  except OSError as error:
    pytest.skip(f'Local sockets unavailable for Gloo: {error}')
  if 'lo' in interfaces:
    monkeypatch.setenv('GLOO_SOCKET_IFNAME', 'lo')
  torch.multiprocessing.spawn(distributed_finite_worker,
      args=(str(tmp_path / 'rendezvous'),), nprocs=2, join=True)
