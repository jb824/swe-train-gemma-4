"""Differentiable Gemma text backbone, bounded logits, and projection LoRA."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import Config


def chunked_cross_entropy(hidden, weight, labels, *, chunk_tokens=128, softcap=None):
  """Exact shifted CE; recompute chunk logits during backward to bound memory."""
  states = hidden[:, :-1].reshape(-1, hidden.shape[-1])
  targets = labels[:, 1:].reshape(-1)
  count = targets.ne(-100).sum()
  if count.item() == 0:
    raise ValueError('Batch has no supervised tokens')

  def chunk_loss(x, w, y):
    logits = F.linear(x, w)
    if softcap is not None:
      logits = torch.tanh(logits / softcap) * softcap
    return F.cross_entropy(logits.float(), y, ignore_index=-100, reduction='sum')

  total = hidden.sum() * 0
  for start in range(0, len(targets), chunk_tokens):
    target = targets[start:start + chunk_tokens]
    if not target.ne(-100).any():
      continue
    total = total + checkpoint(chunk_loss, states[start:start + chunk_tokens], weight,
                               target, use_reentrant=False)
  return total / count


def training_model_class():
  from transformers import Gemma4UnifiedForCausalLM
  from transformers.modeling_outputs import CausalLMOutputWithPast

  class TrainingGemma(Gemma4UnifiedForCausalLM):
    loss_chunk_tokens = 128

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                num_items_in_batch=None, soft_ids=None, soft_probs=None, soft_tail=None,
                reference_targets=None, reference_top_k=32, **kwargs):
      if labels is None and reference_targets is None:
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
      # This forward stays inside PEFT and the distributed model wrapper. Only
      # the native frozen head projection is chunked; LoRAs live in the backbone.
      kwargs.pop('num_items_in_batch', None)
      kwargs.pop('logits_to_keep', None)
      kwargs.pop('use_cache', None)
      outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                           use_cache=False, **kwargs)
      if reference_targets is not None:
        from .reference import reference_distribution
        return reference_distribution(outputs.last_hidden_state, self.lm_head.weight,
            reference_targets, reference_top_k, self.loss_chunk_tokens, self.config.final_logit_softcapping)
      if getattr(self, 'token_weighted', False) or soft_ids is not None:
        from .objectives import chunked_objective
        loss = chunked_objective(outputs.last_hidden_state, self.lm_head.weight, labels,
            chunk_tokens=self.loss_chunk_tokens, softcap=self.config.final_logit_softcapping,
            denominator=num_items_in_batch, soft_ids=soft_ids, soft_probs=soft_probs, soft_tail=soft_tail)
        return CausalLMOutputWithPast(loss=loss)
      loss = chunked_cross_entropy(
          outputs.last_hidden_state, self.lm_head.weight, labels,
          chunk_tokens=self.loss_chunk_tokens,
          softcap=self.config.final_logit_softcapping,
      )
      return CausalLMOutputWithPast(loss=loss)

  return TrainingGemma


def load_base(config: Config):
  from transformers import AutoConfig, CompressedTensorsConfig

  source = config.model.source
  root = AutoConfig.from_pretrained(source, revision=config.model.revision,
                                    local_files_only=config.model.local_files_only)
  if root.model_type not in {'gemma4_unified', 'gemma4_unified_text'}:
    raise ValueError(f'Expected Gemma 4 unified/text config, got {root.model_type}')
  text = copy.deepcopy(root.get_text_config())
  quant = getattr(root, 'quantization_config', None)
  options = {}
  if quant:
    quant = quant.to_dict() if hasattr(quant, 'to_dict') else dict(quant)
    if quant.get('quant_method') != 'compressed-tensors':
      raise ValueError('Only compressed-tensors or dense BF16 base checkpoints are supported')
    quant.update(dequantize=True, run_compressed=False)
    # Transformers infers pre_quantized from the supplied model config, not
    # from the quantization_config loading argument. The text config must keep
    # the unified checkpoint's metadata when loading its packed text weights.
    text.quantization_config = copy.deepcopy(quant)
    options['quantization_config'] = CompressedTensorsConfig.from_dict(quant)
  if text.max_position_embeddings < config.runtime.context_length:
    raise ValueError('Student cannot accommodate the configured runtime context')
  model_class = training_model_class()
  # Explicit key mapping avoids loading vision/audio towers or random text
  # weights when using a unified multimodal checkpoint as a text-only model.
  if root.model_type == 'gemma4_unified':
    options['key_mapping'] = {r'^model\.language_model\.': 'model.'}
  model, info = model_class.from_pretrained(
      source, config=text, revision=config.model.revision,
      local_files_only=config.model.local_files_only,
      dtype=torch.bfloat16, attn_implementation=config.model.attention,
      output_loading_info=True, **options,
  )
  if info.get('missing_keys') or info.get('mismatched_keys'):
    raise ValueError(f'Text weights did not load completely: {info}')
  model.loss_chunk_tokens = config.memory.loss_chunk_tokens
  model.token_weighted = config.token_weighted
  model._swe_source_model_type = root.model_type
  model.requires_grad_(False)
  model.config.use_cache = False
  return model


def attach_adapter(model, config: Config):
  from peft import LoraConfig, PeftModel, get_peft_model

  if config.adapter.warm_start:
    from .numerics import check_adapter_file
    check_adapter_file(config.adapter.warm_start / 'adapter_model.safetensors')
    provenance_path = config.adapter.warm_start / 'provenance.json'
    provenance = json.loads(provenance_path.read_text()) if provenance_path.is_file() else {}
    mapping = {r'^model\.language_model\.': 'model.'} if provenance.get('adapter_layout') == 'gemma4_unified' else None
    model = PeftModel.from_pretrained(model, config.adapter.warm_start,
                                      is_trainable=True, key_mapping=mapping)
  else:
    ranks = config.adapter.rank_pattern
    alphas = {name: config.adapter.alpha_pattern.get(
        name, rank * config.adapter.alpha_multiplier
    ) for name, rank in ranks.items()}
    model = get_peft_model(model, LoraConfig(
        task_type='CAUSAL_LM', r=next(iter(ranks.values())),
        lora_alpha=next(iter(alphas.values())),
        rank_pattern=ranks, alpha_pattern=alphas,
        target_modules=list(ranks), lora_dropout=config.adapter.dropout,
        bias='none', use_rslora=False,
    ))
  report = []
  found = set()
  for name, module in model.named_modules():
    if hasattr(module, 'lora_A'):
      projection = name.rsplit('.', 1)[-1]
      found.add(projection)
      a, b = module.lora_A['default'].weight, module.lora_B['default'].weight
      if a.shape[0] != config.adapter.rank_pattern.get(projection):
        raise ValueError(f'Loaded adapter rank does not match config at {name}')
      expected_alpha = config.adapter.alpha_pattern.get(
          projection, a.shape[0] * config.adapter.alpha_multiplier
      )
      if module.lora_alpha['default'] != expected_alpha:
        raise ValueError(f'Loaded adapter alpha does not match config at {name}')
      report.append({'module': name, 'rank': a.shape[0], 'alpha': expected_alpha,
                     'parameters': a.numel() + b.numel()})
  if found != set(config.adapter.rank_pattern):
    raise ValueError(f'Some requested projections are absent: {set(config.adapter.rank_pattern) - found}')
  if any(p.requires_grad and 'lora_' not in n for n, p in model.named_parameters()):
    raise ValueError('Non-adapter weights unexpectedly require gradients')
  return model, report


@torch.no_grad()
def trainable_state_dict(model):
  """Gather only adapter parameters, including CPU-offloaded FSDP2 shards."""
  from torch.distributed.tensor import DTensor

  # Filter before any distributed gather, including aliases of frozen/tied
  # embeddings. FSDP2 state_dict() returns local DTensors without a full gather.
  # Activation checkpoint wrappers strip their prefix from state_dict keys,
  # but traversal from a parent model retains it in named_parameters(). Use
  # the same canonical names as PyTorch's checkpoint save/load hooks.
  names = {name.replace('_checkpoint_wrapped_module.', '')
           for name, parameter in model.named_parameters(remove_duplicate=False)
           if parameter.requires_grad}
  local = model.state_dict()
  missing = names - local.keys()
  if missing:
    raise ValueError(f'{len(missing)} trainable parameters absent from state dict; '
                     f'first names: {sorted(missing)[:5]}')
  rank_zero = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
  result = {}
  for name in sorted(names):
    value = local[name].detach()
    if isinstance(value, DTensor):
      # Offloaded local shards must return to their mesh device for NCCL/RCCL
      # collectives. Every rank participates; only rank zero retains CPU copies.
      if value.device.type != value.device_mesh.device_type:
        value = value.to(value.device_mesh.device_type)
      value = value.full_tensor()
    if rank_zero:
      result[name] = value.cpu()
  from .numerics import require_finite
  require_finite(result.items(), 'Gathered adapter weights')
  return result


def save_lora_adapter(model, state_dict, destination):
  """Write gathered projection LoRAs without inspecting wrapped module names."""
  from peft import LoraConfig
  from safetensors.torch import save_file

  if set(model.peft_config) != {'default'}:
    raise ValueError('Adapter saving requires one default LoRA adapter')
  config = copy.deepcopy(model.peft_config['default'])
  if not isinstance(config, LoraConfig) or config.bias != 'none' or config.modules_to_save:
    raise ValueError('Adapter saving supports projection LoRAs without bias or modules_to_save')
  if not state_dict:
    raise ValueError('Refusing to save an empty adapter')
  from .numerics import require_finite
  require_finite(state_dict.items(), 'Adapter save', distributed=False)
  weights = {}
  for name, value in state_dict.items():
    if not name.endswith(('.lora_A.default.weight', '.lora_B.default.weight')):
      raise ValueError(f'Unexpected projection LoRA parameter: {name}')
    if value.ndim != 2 or not value.numel():
      raise ValueError(f'Incomplete projection LoRA tensor: {name} {tuple(value.shape)}')
    weights[name.removesuffix('.default.weight') + '.weight'] = value.contiguous()
  destination = Path(destination)
  destination.mkdir(parents=True, exist_ok=True)
  model.create_or_update_model_card(destination)
  save_file(weights, destination / 'adapter_model.safetensors', metadata={'format': 'pt'})
  config.inference_mode = True
  config.save_pretrained(destination)


class Collator:
  def __init__(self, pad_token_id, cache_dir=None):
    self.pad_token_id = pad_token_id
    self.cache_dir = cache_dir

  def __call__(self, rows):
    size = max(len(r['input_ids']) for r in rows)
    result = {key: [] for key in ('input_ids', 'attention_mask', 'labels')}
    for row in rows:
      for key, pad in (('input_ids', self.pad_token_id), ('attention_mask', 0), ('labels', -100)):
        result[key].append(row[key] + [pad] * (size - len(row[key])))
    result = {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}
    if self.cache_dir is not None:
      from safetensors.torch import load_file
      cached = [load_file(Path(self.cache_dir) / f'{row["example_id"]}.safetensors') for row in rows]
      for key in ('soft_ids', 'soft_probs', 'soft_tail'):
        result[key] = torch.cat([row[key] for row in cached])
    return result
