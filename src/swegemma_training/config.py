"""Strict file configuration; importing this module does not import Torch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
import yaml


class Section(BaseModel):
  model_config = ConfigDict(extra='forbid', validate_default=True, allow_inf_nan=False)


class Model(Section):
  source: str = 'google/gemma-4-12B-it-qat-w4a16-ct'
  revision: str = 'main'
  local_files_only: bool = False
  attention: Literal['sdpa', 'eager'] = 'sdpa'


class Window(Section):
  max_tokens: int = Field(default=8192, ge=2)
  history_tokens: int = Field(default=2048, ge=0)


class Data(Section):
  format: Literal['atif', 'harness_jsonl'] = 'atif'
  train_file: Path | None = None
  validation_file: Path | None = None
  prompt_override: Path | None = None
  traces: Path = Path('./teacher-traces')
  outcomes: Path = Path('./teacher-results')
  prepared: Path = Path('./prepared')
  verified_passing_only: Literal[True] = True
  validation_fraction: float = Field(default=0.1, gt=0, lt=1)
  seed: int = 42
  sequence_mode: Literal['turn', 'trajectory', 'windowed'] = 'turn'
  window: Window = Field(default_factory=Window)


class Soft(Section):
  rstar: float = Field(default=0.5, ge=0, le=1)
  top_k: int = Field(default=32, ge=1)
  cache_dir: Path = Path('./reference-cache')


class Objective(Section):
  name: Literal['sft', 'soft'] = 'sft'
  soft: Soft = Field(default_factory=Soft)


class Harness(Section):
  submission_dir: Path = Path.home() / 'dev/kaggle/gemma-4-developer-agent_nemotron_v3.5/sample_submission'
  data_root: Path | None = None
  sandbox: Literal['subprocess', 'docker'] = 'subprocess'


class Evaluation(Section):
  compare_base: bool = True
  task_ids: list[str] | None = None
  port: int = Field(default=8000, ge=1024, le=65535)
  tool_call_parser: str = 'gemma4'
  reasoning_parser: str = 'gemma4'


class Runtime(Section):
  context_length: int = Field(default=32768, ge=2)
  # Completion and reasoning settings are resolved from the harness YAML.


PROJECTIONS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')


class Adapter(Section):
  name: str = 'main_lora'
  rank_pattern: dict[str, int] = Field(default_factory=lambda: dict(zip(
      PROJECTIONS, (12, 8, 8, 12, 24, 16, 16)
  )))
  alpha_multiplier: float = Field(default=2, gt=0)
  alpha_pattern: dict[str, float] = Field(default_factory=dict)
  dropout: float = Field(default=0, ge=0, lt=1)
  warm_start: Path | None = None

  @model_validator(mode='after')
  def validate_projections(self):
    if set(self.rank_pattern) != set(PROJECTIONS):
      raise ValueError(f'rank_pattern must specify all seven projections: {PROJECTIONS}')
    if any(isinstance(v, bool) or v < 1 for v in self.rank_pattern.values()):
      raise ValueError('Projection ranks must be positive integers')
    if set(self.alpha_pattern) - set(PROJECTIONS) or any(v <= 0 for v in self.alpha_pattern.values()):
      raise ValueError('alpha_pattern has invalid projections or nonpositive scaling')
    if Path(self.name).name != self.name or self.name in {'', '.', '..'}:
      raise ValueError('Adapter name must be a safe path component')
    return self


class Memory(Section):
  offload: Literal['none', 'cpu'] = 'cpu'
  cpu_offload_pin_memory: bool = True
  processes: int = Field(default=1, ge=1)
  activation_checkpointing: bool = True
  loss_chunk_tokens: int = Field(default=128, ge=1)


class Optimizer(Section):
  learning_rate: float = Field(default=1e-4, gt=0)
  weight_decay: float = Field(default=0, ge=0)
  max_grad_norm: float = Field(default=1, gt=0)
  scheduler: Literal['cosine', 'linear', 'constant'] = 'cosine'
  warmup_ratio: float = Field(default=0.03, ge=0, lt=1)


class Training(Section):
  epochs: float = Field(default=1, gt=0)
  microbatch_size: int = Field(default=1, ge=1)
  gradient_accumulation: int = Field(default=8, ge=1)
  seed: int = 42
  save_steps: int = Field(default=1, ge=1)
  eval_steps: int = Field(default=100, ge=1)
  save_total_limit: int = Field(default=3, ge=1)
  logging_steps: int = Field(default=1, ge=1)
  tensorboard: bool = False


class Output(Section):
  directory: Path = Path('./runs')


class Progress(Section):
  enabled: bool = True
  heartbeat_seconds: float = Field(default=30, ge=1)


class Remote(Section):
  enabled: bool = False
  repo_id: str | None = None
  mode: Literal['create', 'version'] = 'create'

  @model_validator(mode='after')
  def require_id(self):
    if self.enabled and (not self.repo_id or len(self.repo_id.split('/')) != 2):
      raise ValueError('An enabled publisher requires owner/repo in repo_id')
    return self


class Publish(Section):
  hf: Remote = Field(default_factory=Remote)
  kaggle: Remote = Field(default_factory=Remote)


class Config(Section):
  model: Model = Field(default_factory=Model)
  data: Data = Field(default_factory=Data)
  harness: Harness = Field(default_factory=Harness)
  evaluation: Evaluation = Field(default_factory=Evaluation)
  runtime: Runtime = Field(default_factory=Runtime)
  adapter: Adapter = Field(default_factory=Adapter)
  memory: Memory = Field(default_factory=Memory)
  optimizer: Optimizer = Field(default_factory=Optimizer)
  training: Training = Field(default_factory=Training)
  output: Output = Field(default_factory=Output)
  progress: Progress = Field(default_factory=Progress)
  publish: Publish = Field(default_factory=Publish)
  objective: Objective = Field(default_factory=Objective)

  @model_validator(mode='after')
  def validate_window(self):
    if self.data.sequence_mode == 'windowed' and self.data.window.max_tokens > self.runtime.context_length:
      raise ValueError('window.max_tokens exceeds the harness context length')
    if self.data.format == 'harness_jsonl':
      if self.data.train_file is None or self.data.validation_file is None:
        raise ValueError('harness_jsonl requires data.train_file and data.validation_file')
      if self.data.sequence_mode != 'trajectory':
        raise ValueError('harness_jsonl currently requires data.sequence_mode: trajectory')
    elif any(getattr(self.data, name) is not None for name in ('train_file', 'validation_file', 'prompt_override')):
      raise ValueError('train_file, validation_file, and prompt_override require data.format: harness_jsonl')
    return self

  @property
  def token_weighted(self):
    return self.data.sequence_mode != 'turn' or self.objective.name == 'soft'

  def fingerprint(self) -> str:
    # Publishing, output location, and console settings do not change gradients.
    value = self.model_dump(mode='json', exclude={'publish', 'progress', 'output', 'evaluation'})
    # Preserve checkpoint identities for the previous pinned-offload default.
    if self.memory.cpu_offload_pin_memory:
      value['memory'].pop('cpu_offload_pin_memory')
    if self.data.format == 'atif':
      for key in ('format', 'train_file', 'validation_file', 'prompt_override'):
        value['data'].pop(key)
    # Keep identities of existing turn-SFT checkpoints stable.
    if self.data.sequence_mode == 'turn':
      value['data'].pop('sequence_mode')
      value['data'].pop('window')
    if self.objective.name == 'sft':
      value.pop('objective')
    return digest(value)


def digest(value) -> str:
  return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_config(path: Path, overrides: list[str] = ()) -> Config:
  path = path.expanduser().resolve()
  raw = yaml.safe_load(path.read_text()) or {}
  if not isinstance(raw, dict):
    raise ValueError('The configuration must be a YAML mapping')
  for item in overrides:
    key, separator, value = item.partition('=')
    if not separator or not key or any(not k for k in key.split('.')):
      raise ValueError(f'Override must be dotted.key=YAML_VALUE: {item}')
    node = raw
    parts = key.split('.')
    for part in parts[:-1]:
      node = node.setdefault(part, {})
      if not isinstance(node, dict):
        raise ValueError(f'Override traverses a non-mapping: {item}')
    node[parts[-1]] = yaml.safe_load(value)
  config = Config.model_validate(raw)
  for section in (config.data, config.harness, config.output, config.adapter, config.objective.soft):
    for key in type(section).model_fields:
      value = getattr(section, key)
      if isinstance(value, Path):
        value = value.expanduser()
        setattr(section, key, (path.parent / value).resolve() if not value.is_absolute() else value.resolve())
  if config.model.source.startswith(('.', '/', '~')):
    source = Path(config.model.source).expanduser()
    config.model.source = str((path.parent / source).resolve())
  return config
