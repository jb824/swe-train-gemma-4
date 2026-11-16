"""Local artifacts and explicit, independently retryable publishing."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile

from .config import Config


def file_sha256(path: Path) -> str:
  with path.open('rb') as handle:
    return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path: Path, value) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + '.tmp')
  temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
  temporary.replace(path)


def latest_checkpoint(root: Path) -> Path:
  paths = [p.parent for p in root.glob('*/checkpoints/checkpoint-*/complete.json')]
  if not paths:
    raise FileNotFoundError(f'No complete resumable checkpoints under {root}')
  return max(paths, key=lambda p: (p.parents[1].name, int(p.name.split('-')[-1])))


def latest_release(run: Path) -> Path:
  value = json.loads((run / 'latest-release.json').read_text())
  path = (run / value['path']).resolve()
  if not path.is_relative_to(run.resolve()):
    raise ValueError('Release pointer leaves its run directory')
  return path


def check_resume(path: Path, identity: dict) -> None:
  marker = path / 'complete.json'
  if not marker.is_file() or not (path / 'trainer_state.json').is_file():
    raise ValueError('Checkpoint is incomplete; select one with complete.json and trainer_state.json')
  saved = json.loads(marker.read_text())
  if saved['identity'] != identity:
    raise ValueError('Resume config/data/base/environment/topology changed. Use adapter.warm_start for an explicit optimizer reset.')
  from .numerics import check_adapter_file
  check_adapter_file(path / 'adapter_model.safetensors')


def export_adapter(source: Path, destination: Path, provenance: dict) -> Path:
  # PEFT saves its default adapter at the root. No full base-model or optimizer
  # state is included in a portable release.
  if not (source / 'adapter_config.json').is_file() or not (source / 'adapter_model.safetensors').is_file():
    raise ValueError(f'Not a safetensors PEFT adapter: {source}')
  if destination.exists():
    raise FileExistsError(f'Release already exists: {destination}')
  from .numerics import check_adapter_file
  check_adapter_file(source / 'adapter_model.safetensors')
  destination.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
    stage = Path(tmp) / 'release'
    stage.mkdir()
    for pattern in ('adapter_config.json', 'adapter_model.safetensors',
                    '*token*', '*vocab*', '*merges*', '*chat_template*',
                    'special_tokens_map.json', 'generation_config.json'):
      for path in source.glob(pattern):
        if path.is_file():
          shutil.copy2(path, stage / path.name)
    if provenance.get('adapter_layout') == 'gemma4_unified':
      from safetensors.torch import load_file, save_file
      weights = load_file(str(stage / 'adapter_model.safetensors'))
      prefix = 'base_model.model.model.'
      converted = {}
      for name, tensor in weights.items():
        if not name.startswith(prefix) or name.startswith(prefix + 'language_model.'):
          raise ValueError(f'Unexpected text-training adapter key: {name}')
        converted[prefix + 'language_model.' + name[len(prefix):]] = tensor
      save_file(converted, str(stage / 'adapter_model.safetensors'), metadata={'format': 'pt'})
    write_json(stage / 'provenance.json', provenance)
    manifest = {p.name: file_sha256(p) for p in stage.iterdir() if p.is_file()}
    write_json(stage / 'release.json', {'schema_version': 1, 'files': manifest,
                                      'inference_validation_required': True})
    stage.rename(destination)
  return destination


def publish(config: Config, target: str, release: Path, *, version=False) -> dict:
  remote = getattr(config.publish, target)
  if not remote.enabled:
    raise ValueError(f'Publishing to {target} is disabled in the configuration')
  from .numerics import check_adapter_file
  check_adapter_file(release / 'adapter_model.safetensors')
  manifest_file = release / 'release.json'
  manifest = json.loads(manifest_file.read_text())
  for name, checksum in manifest['files'].items():
    path = release / name
    if Path(name).name != name or not path.is_file() or file_sha256(path) != checksum:
      raise ValueError(f'Release integrity check failed: {name}')
  status_path = release.parent / f'{release.name}-publish-{target}.json'
  try:
    if target == 'hf':
      from huggingface_hub import HfApi
      api = HfApi()
      api.create_repo(remote.repo_id, private=True, exist_ok=True)
      # Match an allowlist to avoid uploading unrelated files added to a release.
      result = api.upload_folder(repo_id=remote.repo_id, folder_path=release,
                                 allow_patterns=[*manifest['files'], 'release.json'])
      status = {'status': 'uploaded', 'target': target, 'url': str(result)}
    elif target == 'kaggle':
      with tempfile.TemporaryDirectory(prefix='sft-kaggle-') as tmp:
        stage = Path(tmp)
        for name in [*manifest['files'], 'release.json']:
          shutil.copy2(release / name, stage / name)
        write_json(stage / 'dataset-metadata.json', {
            'id': remote.repo_id, 'title': remote.repo_id.split('/')[-1],
            'licenses': [{'name': 'other'}],
            'description': 'Gemma PEFT adapter; see provenance.json for base revision and training data fingerprints.',
        })
        version = version or remote.mode == 'version'
        command = ['kaggle', 'datasets', 'version' if version else 'create',
                   '-p', str(stage), '--keep-tabular']
        if version:
          command += ['-m', 'SFT adapter release']
        subprocess.run(command, check=True)
      status = {'status': 'uploaded', 'target': target,
                'url': f'https://www.kaggle.com/datasets/{remote.repo_id}'}
    else:
      raise ValueError(f'Unknown publication target: {target}')
    write_json(status_path, status)
    return status
  except Exception as error:
    write_json(status_path, {'status': 'failed', 'target': target,
                             'error_type': type(error).__name__,
                             'local_release': str(release)})
    raise
