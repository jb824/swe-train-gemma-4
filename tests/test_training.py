import copy
import json

import pytest
import torch
import torch.nn.functional as F

from swegemma_training.artifacts import check_resume, export_adapter, publish
from swegemma_training.modeling import Collator, attach_adapter, chunked_cross_entropy, load_base, trainable_state_dict, training_model_class
from swegemma_training.runner import callback_type, sft_config_kwargs, trainer_type


def tiny_config():
  from transformers import Gemma4UnifiedTextConfig
  return Gemma4UnifiedTextConfig(
      vocab_size=256, hidden_size=32, intermediate_size=64,
      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
      head_dim=8, global_head_dim=8, max_position_embeddings=128,
      layer_types=['sliding_attention', 'full_attention'],
      sliding_window=32, final_logit_softcapping=10.0,
  )


def test_chunked_loss_and_gradients_match_dense_softcapped_ce():
  torch.manual_seed(1)
  hidden = torch.randn(2, 7, 9, requires_grad=True)
  weights = torch.randn(17, 9)
  labels = torch.randint(0, 17, (2, 7))
  labels[:, :3] = -100
  loss = chunked_cross_entropy(hidden, weights, labels, chunk_tokens=2, softcap=3.0)
  loss.backward()
  gradient = hidden.grad.clone()
  hidden.grad = None
  logits = torch.tanh(F.linear(hidden[:, :-1], weights) / 3.0) * 3.0
  expected = F.cross_entropy(logits.reshape(-1, 17), labels[:, 1:].reshape(-1), ignore_index=-100)
  expected.backward()
  torch.testing.assert_close(loss, expected)
  torch.testing.assert_close(gradient, hidden.grad)


def test_warmup_ratio_uses_fractional_warmup_steps(config, tmp_path):
  from transformers import TrainingArguments
  config.memory.offload = 'none'
  config.optimizer.warmup_ratio = 0.03
  kwargs = sft_config_kwargs(config, tmp_path, 1)
  # Exercise the actual Transformers constructor and scheduler conversion on CPU.
  kwargs = {key: value for key, value in kwargs.items()
            if key in TrainingArguments.__dataclass_fields__}
  args = TrainingArguments(**(kwargs | {'use_cpu': True, 'bf16': False}))
  assert args.get_warmup_steps(101) == 4
  assert args.get_warmup_steps(20) == 1
  config.optimizer.warmup_ratio = 0
  kwargs = sft_config_kwargs(config, tmp_path, 1)
  assert kwargs['warmup_steps'] == 0


@pytest.mark.parametrize('checkpoint_wrapped', [False, True])
def test_trainable_state_excludes_frozen_shared_weights(checkpoint_wrapped):
  from peft import LoraConfig, get_peft_model
  from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

  class SharedWeights(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.left = torch.nn.Linear(32, 32, bias=False)
      self.right = torch.nn.Linear(32, 32, bias=False)
      self.right.weight = self.left.weight
      self.target = torch.nn.Linear(32, 32, bias=False)

  model = get_peft_model(SharedWeights(), LoraConfig(r=2, target_modules=['target']))
  expected = {name: parameter.detach().clone()
              for name, parameter in model.named_parameters() if parameter.requires_grad}
  if checkpoint_wrapped:
    # Nested wrappers exercise each prefix removal while keeping frozen aliases.
    model.base_model.model.target = checkpoint_wrapper(checkpoint_wrapper(model.base_model.model.target))
    assert any('_checkpoint_wrapped_module.' in name for name, _ in model.named_parameters())
  saved = trainable_state_dict(model)
  assert set(saved) == set(expected)
  assert all(tensor.device.type == 'cpu' for tensor in saved.values())
  for name, parameter in expected.items():
    torch.testing.assert_close(saved[name], parameter)


@pytest.mark.parametrize('checkpoint_wrapped', [False, True])
def test_actual_trl_arguments_and_training(config, tokenizer, tmp_path, checkpoint_wrapped):
  import inspect
  from datasets import Dataset
  from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing
  from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedTextDecoderLayer
  trl = pytest.importorskip('trl')
  for offload, world in (('cpu', 1), ('none', 2), ('none', 4)):
    config.memory.offload = offload
    config.memory.processes = world
    inspect.signature(trl.SFTConfig).bind(**sft_config_kwargs(config, tmp_path, world))
  config.memory.offload = 'none'
  config.memory.processes = 1
  kwargs = sft_config_kwargs(config, tmp_path, 1)
  args = trl.SFTConfig(**(kwargs | {
      'use_cpu': True, 'bf16': False, 'max_steps': 2,
      'gradient_accumulation_steps': 2, 'disable_tqdm': True,
      'save_strategy': 'no', 'eval_strategy': 'no',
  }))
  assert args.get_warmup_steps(101) == 4
  base = training_model_class()(tiny_config())
  initial_base = copy.deepcopy(base.state_dict())
  base.requires_grad_(False)
  model, _ = attach_adapter(base, config)
  original = {name: value.detach().clone() for name, value in model.named_parameters()}
  data = Dataset.from_list([
      {'input_ids': [2, 10, 11, 1], 'attention_mask': [1, 1, 1, 1], 'labels': [-100, -100, 11, 1]},
      {'input_ids': [2, 12, 13, 1], 'attention_mask': [1, 1, 1, 1], 'labels': [-100, -100, 13, 1]},
  ])
  callback = callback_type()(tmp_path, {}, progress_enabled=False)
  trainer = trainer_type(trl.SFTTrainer)(model=model, args=args,
      processing_class=tokenizer, train_dataset=data,
      data_collator=Collator(tokenizer.pad_token_id), callbacks=[callback])
  callback.trainer = trainer
  trainer.model_accepts_loss_kwargs = False
  result = trainer.train()
  assert trainer.state.global_step == 2 and torch.isfinite(torch.tensor(result.training_loss))
  status = json.loads((tmp_path / 'progress.json').read_text())
  assert status['optimizer_step'] == 2 and status['completed_microbatches_this_session'] == 4
  assert status['phase'] == 'training_complete' and callback.heartbeat_thread is None
  assert any(not torch.equal(original[name], value) for name, value in model.named_parameters()
             if 'lora_B' in name)
  for name, value in model.named_parameters():
    if 'lora_' not in name:
      torch.testing.assert_close(value, original[name], rtol=0, atol=0)

  if checkpoint_wrapped:
    apply_activation_checkpointing(model, check_fn=lambda layer: isinstance(layer, Gemma4UnifiedTextDecoderLayer))
    assert any('_checkpoint_wrapped_module.' in name for name, _ in model.named_parameters())

  class SaveAccelerator:
    is_fsdp2 = True

    def __init__(self, accelerator):
      from types import SimpleNamespace
      from torch.distributed.fsdp import FullOptimStateDictConfig, FullStateDictConfig, StateDictType
      self.accelerator = accelerator
      self.state = SimpleNamespace(fsdp_plugin=SimpleNamespace(
          fsdp_version=2, state_dict_type=StateDictType.FULL_STATE_DICT,
          state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
          optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
      ))

    def __getattr__(self, name):
      return getattr(self.accelerator, name)

    def get_state_dict(self, *args, **kwargs):
      raise AssertionError('Adapter saving must not gather the frozen base')

  trainer.accelerator = SaveAccelerator(trainer.accelerator)
  destination = tmp_path / 'adapter-save'
  trainer.save_model(str(destination))
  from peft import PeftModel
  from safetensors.torch import load_file
  saved = load_file(str(destination / 'adapter_model.safetensors'))
  assert saved and all('lora_' in name for name in saved)
  fresh = training_model_class()(tiny_config())
  fresh.load_state_dict(initial_base)
  restored = PeftModel.from_pretrained(fresh, destination)
  model.eval()
  restored.eval()
  ids = torch.tensor([[2, 10, 11, 1]])
  torch.testing.assert_close(model(input_ids=ids, labels=ids).loss,
                             restored(input_ids=ids, labels=ids).loss)

  # Exercise Accelerate's real native adapter/optimizer resume format on CPU.
  from accelerate.utils import load_fsdp_model, load_fsdp_optimizer
  from accelerate.utils.constants import FSDP_MODEL_NAME
  native = tmp_path / 'native-checkpoint'
  trainer.save_model(str(native), _internal_call=True)
  trainer._save_optimizer_and_scheduler(str(native))
  raw = torch.load(native / f'{FSDP_MODEL_NAME}.bin', weights_only=True)
  assert raw and all('lora_' in name for name in raw)
  fresh = training_model_class()(tiny_config())
  fresh.load_state_dict(initial_base)
  fresh.requires_grad_(False)
  fresh, _ = attach_adapter(fresh, config)
  next_trainer = trainer_type(trl.SFTTrainer)(model=fresh, args=args,
      processing_class=tokenizer, train_dataset=data,
      data_collator=Collator(tokenizer.pad_token_id))
  next_trainer.create_optimizer_and_scheduler(num_training_steps=2)
  if checkpoint_wrapped:
    apply_activation_checkpointing(fresh, check_fn=lambda layer: isinstance(layer, Gemma4UnifiedTextDecoderLayer))
  acceleration = SaveAccelerator(next_trainer.accelerator)
  plugin = acceleration.state.fsdp_plugin
  load_fsdp_model(plugin, acceleration, fresh, native, adapter_only=True)
  load_fsdp_optimizer(plugin, acceleration, next_trainer.optimizer, fresh, native)
  next_trainer.lr_scheduler.load_state_dict(torch.load(native / 'scheduler.pt', weights_only=True))
  fresh.eval()
  torch.testing.assert_close(model(input_ids=ids, labels=ids).loss,
                             fresh(input_ids=ids, labels=ids).loss)
  torch.testing.assert_close(trainer.optimizer.state_dict(), next_trainer.optimizer.state_dict())
  assert next_trainer.lr_scheduler.state_dict() == trainer.lr_scheduler.state_dict()


def test_heartbeat_reports_phase_before_first_optimizer_step(tmp_path, monkeypatch):
  import signal
  import threading
  import time
  from types import SimpleNamespace
  from swegemma_training import runner
  heartbeat_seen = threading.Event()
  write = runner.write_json

  def record(path, value):
    write(path, value)
    if value.get('event') == 'heartbeat':
      heartbeat_seen.set()

  monkeypatch.setattr(runner, 'write_json', record)
  callback = callback_type()(tmp_path, {}, progress_enabled=False, heartbeat_seconds=0.01)
  callback.trainer = SimpleNamespace(_swe_phase='backward', _swe_sequence_tokens=24000,
                                     _swe_microbatch_started=time.monotonic())
  args = SimpleNamespace(gradient_accumulation_steps=8)
  state = SimpleNamespace(global_step=0, max_steps=18, is_world_process_zero=True)
  previous = {name: signal.getsignal(name) for name in (signal.SIGINT, signal.SIGTERM)}
  try:
    callback.on_train_begin(args, state, None)
    assert heartbeat_seen.wait(timeout=2)
    status = json.loads((tmp_path / 'progress.json').read_text())
    assert status['phase'] == 'backward' and status['optimizer_step'] == 0
    assert status['sequence_tokens'] == 24000 and status['completed_microbatches_this_session'] == 0
    assert status['microbatch_elapsed_seconds'] >= 0
    callback.trainer._swe_phase = 'saving_model'
    callback.trainer._swe_phase_started = time.monotonic()
    callback.report_progress(args, state, 'heartbeat')
    status = json.loads((tmp_path / 'progress.json').read_text())
    assert status['phase'] == 'saving_model' and status['microbatch_elapsed_seconds'] is None
  finally:
    callback.restore()
  assert callback.heartbeat_thread is None
  assert all(signal.getsignal(name) == handler for name, handler in previous.items())


def test_real_gemma_adapters_train_save_reload(config, tmp_path):
  from peft import PeftModel
  cls = training_model_class()
  base = cls(tiny_config())
  frozen = copy.deepcopy(base.state_dict())
  base.requires_grad_(False)
  model, ranks = attach_adapter(base, config)
  model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
  assert len(ranks) == 14
  assert {r['rank'] for r in ranks} == {8, 12, 16, 24}
  ids = torch.tensor([[2, 10, 11, 12, 1]])
  model.train()
  loss = model(input_ids=ids, labels=ids).loss
  assert torch.isfinite(loss)
  loss.backward()
  assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if 'lora_B' in n)
  optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
  optimizer.step()
  for name, value in model.get_base_model().state_dict().items():
    if 'lora_' not in name:
      original = name.replace('.base_layer', '')
      torch.testing.assert_close(value, frozen[original])
  model.eval()
  expected = model(input_ids=ids, labels=ids).loss.detach()
  path = tmp_path / 'adapter'
  model.save_pretrained(path)
  fresh = cls(tiny_config())
  fresh.load_state_dict(frozen)
  loaded = PeftModel.from_pretrained(fresh, path).eval()
  torch.testing.assert_close(loaded(input_ids=ids, labels=ids).loss, expected)


def test_real_trainer_checkpoint_resume_matches_uninterrupted(config, tokenizer, tmp_path):
  from transformers import Trainer, TrainingArguments, set_seed
  cls = training_model_class()
  set_seed(12)
  base = cls(tiny_config())
  original = copy.deepcopy(base.state_dict())
  rows = [{'input_ids': [2, 10, 11, 12, 1], 'attention_mask': [1]*5,
           'labels': [-100, -100, 11, 12, 1]}] * 4
  identity = {'config': 'fixed', 'world_size': 1}

  def run(folder, resume=None, stop=False):
    set_seed(12)
    model = cls(tiny_config())
    model.load_state_dict(original)
    model.requires_grad_(False)
    model, _ = attach_adapter(model, config)
    args = TrainingArguments(output_dir=str(folder), use_cpu=True, max_steps=2,
        per_device_train_batch_size=1, save_steps=1, logging_steps=1,
        learning_rate=0.001, report_to=[], disable_tqdm=True,
        remove_unused_columns=False, seed=12)
    callback = callback_type()(folder.parent, identity)
    callback.stopping = stop
    if resume:
      callback.resume_tokens = json.loads((resume / 'complete.json').read_text())['supervised_tokens']
    trainer = trainer_type(Trainer)(model=model, args=args, train_dataset=rows,
        data_collator=Collator(0), callbacks=[callback])
    callback.trainer = trainer
    trainer.model_accepts_loss_kwargs = False
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    return {n: p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}

  uninterrupted = run(tmp_path / 'first/checkpoints')
  checkpoint = tmp_path / 'first/checkpoints/checkpoint-1'
  check_resume(checkpoint, identity)
  resumed = run(tmp_path / 'second/checkpoints', checkpoint)
  for name in uninterrupted:
    torch.testing.assert_close(uninterrupted[name], resumed[name])
  with pytest.raises(ValueError, match='changed'):
    check_resume(checkpoint, {'config': 'changed', 'world_size': 1})
  run(tmp_path / 'interrupted/checkpoints', stop=True)
  assert not (tmp_path / 'interrupted/checkpoints/checkpoint-2').exists()
  marker = json.loads((tmp_path / 'interrupted/checkpoints/checkpoint-1/complete.json').read_text())
  assert marker['supervised_tokens'] == 3
  resumed = run(tmp_path / 'continued/checkpoints', tmp_path / 'interrupted/checkpoints/checkpoint-1')
  for name in uninterrupted:
    torch.testing.assert_close(uninterrupted[name], resumed[name])
  marker = json.loads((tmp_path / 'continued/checkpoints/checkpoint-2/complete.json').read_text())
  assert marker['supervised_tokens'] == 6


def test_publishing_is_disabled_before_any_network_access(config, tmp_path):
  with pytest.raises(ValueError, match='disabled'):
    publish(config, 'hf', tmp_path / 'missing')
  with pytest.raises(ValueError, match='disabled'):
    publish(config, 'kaggle', tmp_path / 'missing')


def test_export_excludes_optimizer_and_base_weights(config, tmp_path):
  model = training_model_class()(tiny_config())
  model.requires_grad_(False)
  model, _ = attach_adapter(model, config)
  source = tmp_path / 'adapter'
  model.save_pretrained(source)
  (source / 'optimizer.pt').write_bytes(b'not for upload')
  release = export_adapter(source, tmp_path / 'release', {'base_source': 'test'})
  assert (release / 'adapter_model.safetensors').is_file()
  assert not (release / 'optimizer.pt').exists()
  assert (release / 'provenance.json').is_file()


def test_unified_checkpoint_maps_text_weights_without_loading_modal_towers(config, tmp_path):
  from safetensors.torch import save_file
  from transformers import Gemma4UnifiedConfig
  base = training_model_class()(tiny_config())
  folder = tmp_path / 'unified'
  folder.mkdir()
  unified = Gemma4UnifiedConfig(text_config=tiny_config().to_dict())
  unified.save_pretrained(folder)
  weights = {n.replace('model.', 'model.language_model.', 1) if n.startswith('model.') else n:
             p.detach().clone() for n,p in base.state_dict().items()}
  save_file(weights, folder / 'model.safetensors', metadata={'format': 'pt'})
  config.model.source = str(folder)
  config.model.local_files_only = True
  config.runtime.context_length = 64
  model = load_base(config)
  assert not any('vision' in n or 'audio' in n for n,_ in model.named_parameters())
  for name, parameter in model.state_dict().items():
    torch.testing.assert_close(parameter, base.state_dict()[name].to(torch.bfloat16))


def test_packed_unified_checkpoint_preserves_quantization_metadata(config, tmp_path):
  from compressed_tensors.compressors.pack_quantized import PackedQuantizationCompressor
  from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
  from safetensors.torch import save_file
  from transformers import Gemma4UnifiedConfig

  base = training_model_class()(tiny_config()).to(torch.bfloat16)
  scheme = QuantizationScheme(
      targets=['Linear'], format='pack-quantized',
      weights=QuantizationArgs(num_bits=4, type='int', strategy='group',
                               group_size=8, symmetric=True),
  )
  weights = {name: value.detach().clone() for name, value in base.state_dict().items()}
  expected = copy.deepcopy(weights)
  for name, module in base.named_modules():
    if not isinstance(module, torch.nn.Linear) or name == 'lm_head':
      continue
    weight = weights.pop(f'{name}.weight')
    scale = weight.reshape(weight.shape[0], -1, 8).abs().amax(-1).clamp_min(1e-5) / 7
    packed = PackedQuantizationCompressor.compress({'weight': weight, 'weight_scale': scale}, scheme)
    weights.update({f'{name}.{key}': value for key, value in packed.items()})
    expected[f'{name}.weight'] = PackedQuantizationCompressor.decompress(packed, scheme)['weight']

  unified = Gemma4UnifiedConfig(text_config=tiny_config().to_dict())
  assert getattr(unified.get_text_config(), 'quantization_config', None) is None
  unified.quantization_config = {
      'quant_method': 'compressed-tensors', 'format': 'pack-quantized',
      'quantization_status': 'compressed', 'ignore': ['lm_head'],
      'config_groups': {'group_0': scheme.model_dump(mode='json')},
  }
  folder = tmp_path / 'packed-unified'
  folder.mkdir()
  unified.save_pretrained(folder)
  weights = {name.replace('model.', 'model.language_model.', 1) if name.startswith('model.') else name:
             value for name, value in weights.items()}
  save_file(weights, folder / 'model.safetensors', metadata={'format': 'pt'})
  config.model.source = str(folder)
  config.model.local_files_only = True
  config.runtime.context_length = 64
  loaded = load_base(config)
  for name, value in expected.items():
    torch.testing.assert_close(loaded.state_dict()[name], value.to(torch.bfloat16))
  assert not any('weight_packed' in name for name in loaded.state_dict())
  assert not any(parameter.requires_grad for parameter in loaded.parameters())
  adapted, _ = attach_adapter(loaded, config)
  ids = torch.tensor([[2, 10, 11, 1]])
  loss = adapted(input_ids=ids, labels=ids).loss
  loss.backward()
  assert torch.isfinite(loss)
  assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
             for name, parameter in adapted.named_parameters() if 'lora_B' in name)


def test_real_12b_projection_shapes_match_rank_preset(config):
  from transformers import AutoConfig
  try:
    text = AutoConfig.from_pretrained(config.model.source, local_files_only=True).get_text_config()
  except OSError:
    pytest.skip('12B config not available locally')
  with torch.device('meta'):
    base = training_model_class()(text)
    base.requires_grad_(False)
    _, ranks = attach_adapter(base, config)
  assert sum(r['parameters'] for r in ranks) == 65_585_152
  assert len(ranks) == 328


def test_failed_upload_preserves_release_and_can_be_retried(config, tmp_path, monkeypatch):
  model = training_model_class()(tiny_config())
  model.requires_grad_(False)
  model, _ = attach_adapter(model, config)
  model.save_pretrained(tmp_path / 'adapter')
  release = export_adapter(tmp_path / 'adapter', tmp_path / 'release', {'base_source': 'test'})
  manifest = (release / 'release.json').read_bytes()
  config.publish.hf.enabled = True
  config.publish.hf.repo_id = 'owner/private-adapter'

  class OfflineHub:
    def create_repo(self, repo, private, exist_ok):
      assert repo == 'owner/private-adapter' and private and exist_ok

    def upload_folder(self, **kwargs):
      raise OSError('Offline')

  monkeypatch.setattr('huggingface_hub.HfApi', OfflineHub)
  with pytest.raises(OSError, match='Offline'):
    publish(config, 'hf', release)
  assert (release / 'release.json').read_bytes() == manifest
  assert (release / 'adapter_model.safetensors').is_file()
  assert json.loads((tmp_path / 'release-publish-hf.json').read_text())['status'] == 'failed'

  class OnlineHub(OfflineHub):
    def upload_folder(self, **kwargs):
      assert 'adapter_model.safetensors' in kwargs['allow_patterns']
      assert 'optimizer.pt' not in kwargs['allow_patterns']
      return 'published-commit'

  monkeypatch.setattr('huggingface_hub.HfApi', OnlineHub)
  assert publish(config, 'hf', release)['status'] == 'uploaded'


def test_unified_release_namespace_roundtrips_into_text_training(config, tmp_path):
  from safetensors.torch import load_file
  base = training_model_class()(tiny_config())
  original = copy.deepcopy(base.state_dict())
  base.requires_grad_(False)
  model, _ = attach_adapter(base, config)
  with torch.no_grad():
    for name, parameter in model.named_parameters():
      if 'lora_B' in name:
        parameter.fill_(0.01)
  source = tmp_path / 'adapter'
  model.save_pretrained(source)
  release = export_adapter(source, tmp_path / 'release', {'adapter_layout': 'gemma4_unified'})
  assert all('model.language_model.layers.' in key for key in load_file(str(release / 'adapter_model.safetensors')))
  config.adapter.warm_start = release
  base = training_model_class()(tiny_config())
  base.load_state_dict(original)
  base.requires_grad_(False)
  loaded, _ = attach_adapter(base, config)
  ids = torch.tensor([[2, 10, 11, 1]])
  model.eval()
  loaded.eval()
  torch.testing.assert_close(model(input_ids=ids, labels=ids).loss,
                             loaded(input_ids=ids, labels=ids).loss)
