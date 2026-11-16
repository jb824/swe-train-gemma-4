"""Read the single-agent harness and snapshot its executable prompt contract."""

from __future__ import annotations

from pathlib import Path
import os

import yaml

from .config import Config, digest


def read_harness(config: Config) -> dict:
  root = config.harness.submission_dir.resolve()
  active = set()
  files = {}

  def read(path):
    path = path.resolve()
    if not path.is_relative_to(root) or path in active:
      raise ValueError(f'Unsafe or cyclic harness include: {path}')
    active.add(path)
    text = path.read_text()
    files[str(path.relative_to(root))] = text
    try:
      if path.suffix not in {'.yaml', '.yml'}:
        return text

      class Loader(yaml.SafeLoader):
        pass

      Loader.add_constructor('!include', lambda loader, node: read(
          path.parent / loader.construct_scalar(node)
      ))
      return yaml.load(text, Loader=Loader)
    finally:
      active.remove(path)

  agent = read(root / 'agent.yaml')
  tools = agent.get('tools', [])
  if not tools or not all(isinstance(t, str) for t in tools) or agent.get('sub_agents'):
    raise ValueError('This recipe requires one agent with named Python tools')
  sampling = agent.get('generate_content_config', {})
  completion = sampling.get('max_output_tokens')
  thinking = sampling.get('thinking_config', {}).get('thinking_budget')
  if not isinstance(completion, int) or not isinstance(thinking, int):
    raise ValueError('Harness must declare max_output_tokens and thinking_budget')
  if not 0 <= thinking < completion < config.runtime.context_length:
    raise ValueError('Require thinking < completion < context_length')
  value = {'agent': agent, 'sampling': sampling, 'files': files,
           'context_length': config.runtime.context_length,
           'completion_tokens': completion, 'thinking_tokens': thinking}
  value['fingerprint'] = digest(value)
  return value


def snapshot_harness(contract: dict, destination: Path) -> None:
  destination.mkdir(parents=True, exist_ok=False)
  for name, text in contract['files'].items():
    path = destination / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def tool_schemas(contract: dict) -> list[dict]:
  os.environ.setdefault('LITELLM_LOCAL_MODEL_COST_MAP', 'True')
  from transformers.utils.chat_template_utils import get_json_schema
  from swegemma.tools import create_tools

  available = create_tools(None)
  result = []
  for name in contract['agent']['tools']:
    if name not in available:
      raise ValueError(f'Installed SWEGemma does not provide {name}')
    schema = get_json_schema(available[name])
    schema['function'].pop('return', None)
    result.append(schema)
  return result
