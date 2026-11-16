"""Immutable SoFT reference targets; no reference model is loaded during training."""

from collections import defaultdict
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile

import torch
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm

from .artifacts import file_sha256, write_json
from .config import digest
from .numerics import require_finite
from .objectives import construct_targets, probability_floor, project, reference_support


def prepared_rows(config):
  return {split: [dict(json.loads(line), example_id=f'{split}-{index:06d}')
                 for index, line in enumerate((config.data.prepared / f'{split}.jsonl').read_text().splitlines())]
          for split in ('train', 'validation')}


def cache_identity(config, manifest):
  from .runner import versions
  warm = config.adapter.warm_start
  local_base = Path(config.model.source)
  local_weights = None
  if local_base.is_dir():
    local_weights = {str(path.relative_to(local_base)): file_sha256(path)
                     for pattern in ('*.safetensors', '*.bin', 'config.json')
                     for path in sorted(local_base.rglob(pattern))}
  return {'dataset': manifest['fingerprint'], 'base_source': manifest['base_source'],
          'base_revision': manifest['base_revision'], 'tokenizer': manifest['tokenizer_sha256'],
          'local_base_files': local_weights,
          'adapter': config.adapter.model_dump(mode='json'),
          'warm_start_sha256': file_sha256(warm / 'adapter_model.safetensors') if warm else None,
          'seed': config.training.seed, 'objective': config.objective.soft.model_dump(exclude={'cache_dir'}),
          'versions': versions(), 'format': 1}


def validate_cache(config, manifest):
  root = config.objective.soft.cache_dir
  path = root / 'manifest.json'
  if not path.is_file():
    raise ValueError('SoFT requires a complete reference cache; run swe-train cache-reference first')
  cached = json.loads(path.read_text())
  original = dict(cached)
  fingerprint = original.pop('fingerprint')
  if digest(original) != fingerprint or cached['identity'] != cache_identity(config, manifest):
    raise ValueError('SoFT reference cache is stale or its identity changed; choose a new cache_dir')
  expected = {row['example_id'] + '.safetensors' for rows in prepared_rows(config).values() for row in rows}
  if set(cached['files']) != expected:
    raise ValueError('SoFT cache does not cover the prepared examples')
  for name, checksum in cached['files'].items():
    if Path(name).name != name or file_sha256(root / name) != checksum:
      raise ValueError(f'SoFT cache shard changed: {name}')
    validate_targets(load_file(root / name), name)
  return cached


def validate_targets(targets, name):
  """Checksums prove identity; these checks also reject invalid legacy caches."""
  require_finite(targets.items(), f'SoFT cache {name}', distributed=False)
  ids, probs, tail = (targets[k] for k in ('soft_ids', 'soft_probs', 'soft_tail'))
  if ids.dtype != torch.int64 or ids.ndim != 2 or probs.shape != ids.shape or tail.shape != ids.shape[:1]:
    raise ValueError(f'Invalid SoFT target shapes/dtypes: {name}')
  if not ids.numel() or (ids < 0).any() or (probs < 0).any() or (tail < 0).any():
    raise ValueError(f'Invalid SoFT target support/probabilities: {name}')
  if not torch.allclose(probs.sum(-1) + tail, torch.ones_like(tail), atol=1e-5, rtol=1e-5):
    raise ValueError(f'SoFT target mass does not sum to one: {name}')


def reference_distribution(hidden, weight, labels, top_k, chunk_tokens, softcap):
  keep = labels[:, 1:].reshape(-1).ne(-100)
  states = hidden[:, :-1].reshape(-1, hidden.shape[-1])[keep]
  gold = labels[:, 1:].reshape(-1)[keep]
  if not len(gold):
    raise ValueError('Reference batch has no supervised tokens')
  pieces = []
  for start in range(0, len(gold), chunk_tokens):
    logits = project(states[start:start + chunk_tokens], weight, softcap)
    pieces.append(reference_support(logits, gold[start:start + chunk_tokens], top_k))
  return {key: torch.cat([piece[key] for piece in pieces]) for key in pieces[0]}


def build_cache(config, manifest, predictor, accelerator=None):
  """All model-sharding ranks predict identical rows; only rank zero writes."""
  root = config.objective.soft.cache_dir
  writer = accelerator is None or accelerator.is_main_process
  if root.exists():
    return validate_cache(config, manifest)
  root.parent.mkdir(parents=True, exist_ok=True)
  rows = prepared_rows(config)
  identity = cache_identity(config, manifest)
  manager = tempfile.TemporaryDirectory(dir=root.parent) if writer else nullcontext(None)
  with manager as temporary:
    stage = Path(temporary) / 'cache' if writer else None
    if writer:
      stage.mkdir()
    groups = defaultdict(list)
    for split_rows in rows.values():
      for row in tqdm(split_rows, desc='Caching frozen reference', unit='example',
                      disable=not writer or not config.progress.enabled):
        support = predictor(row)
        require_finite(support.items(), f'SoFT reference {row["example_id"]}', distributed=False)
        if writer:
          path = stage / f'{row["example_id"]}.safetensors'
          save_file(support, path)
          groups[row['instance_id']].append(row)
    stats = {}
    if writer:
      for iid, examples in tqdm(groups.items(), desc='Calibrating SoFT', unit='trajectory',
                                disable=not config.progress.enabled):
        probs = torch.cat([load_file(stage / f'{r["example_id"]}.safetensors')['gold_probs'] for r in examples])
        floor = probability_floor(probs, config.objective.soft.rstar)
        realized = float((floor - probs.double()).clamp_min(0).sum() / (1 - probs.double()).sum().clamp_min(1e-12))
        stats[iid] = {'floor': floor, 'rstar': config.objective.soft.rstar,
                      'realized_budget': realized, 'supervised_tokens': len(probs)}
        for row in examples:
          path = stage / f'{row["example_id"]}.safetensors'
          support = load_file(path)
          gold = torch.tensor([v for v in row['labels'][1:] if v != -100])
          if len(gold) != len(support['ids']):
            raise ValueError('Reference cache position mismatch')
          q, tail = construct_targets(support, gold, floor)
          targets = {'soft_ids': support['ids'], 'soft_probs': q, 'soft_tail': tail,
                     'base_probs': support['base_probs'], 'gold_probs': support['gold_probs'],
                     'base_tail': (1 - support['base_probs'].sum(-1)).clamp_min(0)}
          validate_targets(targets, row['example_id'])
          save_file(targets, path)
      cached = {'identity': identity, 'calibration': stats,
                'files': {f.name: file_sha256(f) for f in stage.glob('*.safetensors')}}
      cached['fingerprint'] = digest(cached)
      write_json(stage / 'manifest.json', cached)
      stage.rename(root)
    if accelerator is not None:
      accelerator.wait_for_everyone()
  return validate_cache(config, manifest)


def launch_reference(config):
  import os
  import subprocess
  import sys
  import yaml
  from .runner import load_prepared
  manifest = load_prepared(config)
  if config.objective.name != 'soft':
    raise ValueError('cache-reference requires objective.name: soft')
  if config.objective.soft.cache_dir.exists():
    return validate_cache(config, manifest)
  # Keep torchrun's logs outside the immutable cache so failed initialization
  # never produces what looks like a completed cache directory.
  root = config.objective.soft.cache_dir
  root.parent.mkdir(parents=True, exist_ok=True)
  logs = Path(tempfile.mkdtemp(prefix='reference-logs-', dir=root.parent))
  path = logs / 'config.yaml'
  path.write_text(yaml.safe_dump(config.model_dump(mode='json'), sort_keys=False))
  subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
      '--nproc_per_node', str(config.memory.processes), '--log-dir', str(logs), '--tee', '3',
      '-m', 'swegemma_training.reference', '--config', str(path)], check=True,
      env=os.environ | {'PYTHONUNBUFFERED': '1'})
  return validate_cache(config, manifest)


def cache_worker(config):
  import os
  from transformers import AutoTokenizer, Trainer, set_seed
  from trl import SFTConfig
  from .modeling import Collator, attach_adapter, load_base
  from .runner import configure_cpu_offload, load_prepared, sft_config_kwargs
  if not torch.cuda.is_available():
    raise RuntimeError('Reference caching requires an accessible CUDA/ROCm GPU')
  torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
  manifest = load_prepared(config)
  if config.adapter.warm_start:
    provenance = json.loads((config.adapter.warm_start / 'provenance.json').read_text())
    if provenance['base_source'] != config.model.source or provenance['base_revision'] != manifest['base_revision']:
      raise ValueError('Warm-start reference base source/revision differs')
  if config.objective.soft.cache_dir.exists():
    return validate_cache(config, manifest)
  set_seed(config.training.seed)
  config = config.model_copy(deep=True)
  config.model.revision = manifest['base_revision']
  model, _ = attach_adapter(load_base(config), config)
  tokenizer = AutoTokenizer.from_pretrained(config.model.source, revision=manifest['base_revision'],
                                            local_files_only=config.model.local_files_only)
  if digest({'vocab': tokenizer.get_vocab(), 'template': tokenizer.chat_template}) != manifest['tokenizer_sha256']:
    raise ValueError('Reference tokenizer changed since preparation')
  kwargs = sft_config_kwargs(config, config.objective.soft.cache_dir.parent / 'reference-runtime',
                              config.memory.processes)
  kwargs.update(gradient_checkpointing=False, report_to=[], eval_strategy='no')
  if kwargs['fsdp_config']:
    kwargs['fsdp_config']['activation_checkpointing'] = False
  trainer = Trainer(model=model, args=SFTConfig(**kwargs), processing_class=tokenizer,
                    data_collator=Collator(tokenizer.pad_token_id))
  configure_cpu_offload(trainer, config)
  trainer.create_optimizer()
  model, _ = trainer.accelerator.prepare(model, trainer.optimizer)
  model.eval()

  @torch.no_grad()
  def predict(row):
    batch = Collator(tokenizer.pad_token_id)([row])
    batch = {key: value.to(trainer.accelerator.device) for key, value in batch.items()}
    labels = batch.pop('labels')
    with trainer.accelerator.autocast():
      return model(**batch, reference_targets=labels, reference_top_k=config.objective.soft.top_k)
  return build_cache(config, manifest, predict, trainer.accelerator)


if __name__ == '__main__':
  import argparse
  from .config import load_config
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=Path, required=True)
  cache_worker(load_config(parser.parse_args().config))
