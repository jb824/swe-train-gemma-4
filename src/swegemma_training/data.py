"""Versioned hard-distillation examples with auditable exclusions and splits."""

from __future__ import annotations

from collections import defaultdict, deque
import copy
import json
from pathlib import Path
import random
import re
import tempfile

from tqdm.auto import tqdm

from .config import Config, digest
from .harness import read_harness, tool_schemas


def json_records(path: Path):
  if path.suffix == '.jsonl':
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
  value = json.loads(path.read_text())
  return value if isinstance(value, list) else [value]


def outcome_index(path: Path) -> dict:
  if path.suffix == '.log':
    text = path.read_text()
    try:
      events = json.loads(text)
      text = ''.join(e.get('data', '') for e in events
                     if e.get('stream_name') == 'stdout')
    except json.JSONDecodeError:
      pass
    rows = []
    current = None
    for line in text.splitlines():
      started = re.search(r'\[\d+/\d+\] Evaluating ([A-Za-z0-9_.-]+) ', line)
      if started:
        current = started[1]
      finished = re.search(r'resolved=(True|False).*?exit_code=(-?\d+)', line)
      if current and finished:
        duration = re.search(r'duration=([0-9.]+)s', line)
        rows.append({'instance_id': current, 'resolved': finished[1] == 'True',
                     'test_exit_code': int(finished[2]),
                     'duration_seconds': float(duration[1]) if duration else None})
        current = None
    if not rows:
      raise ValueError('Log has no evaluation summaries with resolved and exit_code')
    result = {}
    for row in rows:
      if row['instance_id'] in result:
        raise ValueError('Repeated task attempts in outcome log; select a single run')
      result[row['instance_id']] = row
    return result
  paths = sorted(path.rglob('task_results.jsonl')) if path.is_dir() else [path]
  if not paths:
    raise ValueError(f'No task_results.jsonl found in {path}')
  result = {}
  for item in paths:
    for row in json_records(item):
      key = row.get('instance_id') or row.get('task_id') or row.get('id')
      if not key:
        raise ValueError(f'Outcome without a task ID: {item}')
      if key in result and result[key] != row:
        raise ValueError(f'Ambiguous repeated outcome for {key}; select one collection run')
      result[key] = row
  return result


def trace_id(path: Path, document: dict) -> str:
  key = document.get('instance_id') or document.get('task_id')
  if not key:
    key = (document.get('extra') or {}).get('instance_id')
  if key:
    return str(key)
  # SWEGemma saves trace_<instance_id>.json under results/traces.
  return path.stem.removeprefix('trace_')


def atif_messages(document: dict, contract: dict) -> list[dict]:
  """Reconstruct one agent's recorded transcript; reject ambiguous tool matches."""
  agent = contract['agent']
  messages = [{'role': 'system', 'content': agent['instruction']}]
  pending = defaultdict(deque)
  seen_calls = set()
  previous_key = None
  for step in document.get('steps', []):
    extra = step.get('extra') or {}
    source = step.get('source')
    if extra.get('event_type') == 'compaction':
      raise ValueError('Compacted ATIF cannot prove the original model context; supply uncompacted traces')
    if extra.get('event_type') == 'system_instruction':
      if (step.get('message') or '').strip() != agent['instruction'].strip():
        raise ValueError('Recorded system instruction differs from configured harness')
    if source == 'user' or extra.get('event_type') == 'continuation_nudge':
      if pending:
        raise ValueError('User message arrived before outstanding tool responses')
      messages.append({'role': 'user', 'content': step.get('message', '')})
      previous_key = None
    elif source == 'agent':
      if extra.get('author', agent['name']) != agent['name']:
        raise ValueError('Trace belongs to a different or multi-agent harness')
      if not any(m['role'] == 'user' for m in messages):
        raise ValueError('Trace has no recorded initial user prompt')
      key = extra.get('timestamp')
      if key is None:
        key = step.get('llm_call_count')
      merge = key is not None and key == previous_key and messages[-1]['role'] == 'assistant'
      if not merge:
        if pending:
          raise ValueError('Assistant response arrived before outstanding tool responses')
        messages.append({'role': 'assistant', 'content': ''})
      message = messages[-1]
      field = 'reasoning' if extra.get('event_type') == 'thinking' else 'content'
      message[field] = message.get(field, '') + (step.get('message') or '')
      for call in step.get('tool_calls') or []:
        call_extra = call.get('extra') or {}
        if call_extra.get('author', agent['name']) != agent['name']:
          raise ValueError('Mixed-author ATIF tool calls cannot be used by a single agent')
        name, call_id = call['function_name'], call['tool_call_id']
        if name not in agent['tools'] or call_id in seen_calls:
          raise ValueError(f'Unknown tool or duplicate call ID: {name}/{call_id}')
        arguments = call.get('arguments', {})
        if isinstance(arguments, str):
          arguments = json.loads(arguments)
        if not isinstance(arguments, dict):
          raise ValueError('Tool arguments must be JSON objects')
        message.setdefault('tool_calls', []).append({
            'id': call_id, 'type': 'function',
            'function': {'name': name, 'arguments': arguments},
        })
        seen_calls.add(call_id)
        pending[name].append(call_id)
      previous_key = key
    observation = step.get('observation')
    if observation:
      detail = observation.get('extra') or {}
      name = detail.get('tool_name')
      if name is None and len(pending) == 1:
        name = next(iter(pending))
      if name not in pending or not pending[name]:
        raise ValueError(f'Unmatched tool response: {name}')
      call_id = pending[name].popleft()
      if not pending[name]:
        del pending[name]
      content = observation.get('content', '')
      messages.append({'role': 'tool', 'name': name, 'tool_call_id': call_id,
                       'content': content if isinstance(content, str) else json.dumps(content)})
      previous_key = None
  if pending:
    raise ValueError('Trace ends with outstanding tool calls')
  for message in messages:
    if message['role'] == 'assistant' and message.get('reasoning') and not (
        message.get('content') or message.get('tool_calls')
    ):
      raise ValueError('Unmerged thinking-only event; response boundaries are ambiguous')
  return messages


def render_target(tokenizer, history: list, target: dict, tools: list) -> dict:
  """Use exact token-prefix matching; never guess masks from character offsets."""
  options = dict(tools=tools, enable_thinking=True, tokenize=True, return_dict=False)
  prompt = tokenizer.apply_chat_template(history, add_generation_prompt=True, **options)
  full = tokenizer.apply_chat_template(history + [target], add_generation_prompt=False, **options)
  if hasattr(prompt, 'tolist'):
    prompt, full = prompt.tolist(), full.tolist()
  untrimmed_full = full
  response_marker = tokenizer.encode('<|tool_response>', add_special_tokens=False)
  if target.get('tool_calls') and response_marker and full[-len(response_marker):] == response_marker:
    full = full[:-len(response_marker)]
  boundary = len(prompt)
  if full[:boundary] != prompt:
    boundary = next((i for i, (left, right) in enumerate(zip(prompt, full))
                     if left != right), min(len(prompt), len(full)))
    # Gemma continues the model turn after a tool result. Its generation-only
    # suffix can add a turn closer absent from a completed conversation. Permit
    # only that small control-token suffix; any rewritten history is rejected.
    suffix = tokenizer.decode(prompt[boundary:], skip_special_tokens=False)
    if len(prompt) - boundary > 32 or not re.fullmatch(
        r'(?:<turn\|>|<\|channel>|thought|analysis|final|\s)*', suffix
    ):
      raise ValueError('Student chat template rewrites the prompt prefix')
  if len(full) <= boundary:
    raise ValueError('Empty assistant target after rendering')
  # Token IDs alone are insufficient if a template silently drops reasoning.
  for field in ('reasoning',):
    if target.get(field):
      without = copy.deepcopy(target)
      without.pop(field)
      if tokenizer.apply_chat_template(history + [without], add_generation_prompt=False,
                                       **options) == untrimmed_full:
        raise ValueError('Student template silently discarded recorded reasoning')
  return {'input_ids': full, 'attention_mask': [1] * len(full),
          'labels': [-100] * boundary + full[boundary:],
          'prompt_tokens': boundary, 'completion_tokens': len(full) - boundary}


def prepare(config: Config, tokenizer=None, schemas=None) -> dict:
  contract = read_harness(config)
  revision = config.model.revision
  if tokenizer is None:
    from transformers import AutoConfig, AutoTokenizer
    model_config = AutoConfig.from_pretrained(
        config.model.source, revision=revision,
        local_files_only=config.model.local_files_only,
    )
    revision = getattr(model_config, '_commit_hash', None) or revision
    if not Path(config.model.source).is_dir():
      from huggingface_hub import hf_hub_download
      config_path = Path(hf_hub_download(
          config.model.source, 'config.json', revision=revision,
          local_files_only=config.model.local_files_only,
      ))
      if config_path.parent.parent.name == 'snapshots':
        revision = config_path.parent.name
      if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Could not resolve an immutable model revision; specify model.revision explicitly')
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.source, revision=revision,
        local_files_only=config.model.local_files_only,
    )
  schemas = tool_schemas(contract) if schemas is None else schemas
  if config.data.format == 'harness_jsonl':
    from .harness_jsonl import prepare_jsonl
    return prepare_jsonl(config, tokenizer, revision, schemas, contract)
  outcomes = outcome_index(config.data.outcomes)
  paths = sorted(config.data.traces.rglob('*.json')) if config.data.traces.is_dir() else [config.data.traces]
  accepted = defaultdict(list)
  rejected = []
  coverage = {}
  fingerprints = {}
  for path in tqdm(paths, desc='Preparing teacher traces', unit='trace', disable=not config.progress.enabled):
    try:
      document = json.loads(path.read_text())
    except json.JSONDecodeError:
      rejected.append({'instance_id': path.stem, 'reason': 'invalid_trace_json'})
      continue
    if not isinstance(document, dict) or not str(document.get('schema_version', '')).startswith('ATIF'):
      continue
    iid = trace_id(path, document)
    if iid in fingerprints:
      raise ValueError(f'Multiple traces for {iid}; select one collection run')
    fingerprints[iid] = digest(document)
    outcome = outcomes.get(iid)
    if not outcome or outcome.get('resolved') is not True or outcome.get('test_exit_code') != 0 or outcome.get('error'):
      rejected.append({'instance_id': iid, 'reason': 'not_verified_passing'})
      continue
    try:
      messages = atif_messages(document, contract)
      rows = []
      for index, message in enumerate(messages):
        if message['role'] != 'assistant':
          continue
        row = render_target(tokenizer, messages[:index], message, schemas)
        reasoning_tokens = len(tokenizer.encode(message.get('reasoning', ''), add_special_tokens=False))
        if config.data.sequence_mode == 'turn' and len(row['input_ids']) > contract['context_length']:
          raise ValueError('student_context_overflow')
        if row['completion_tokens'] > contract['completion_tokens']:
          raise ValueError('student_completion_overflow')
        if reasoning_tokens > contract['thinking_tokens']:
          raise ValueError('student_thinking_overflow')
        row.update(instance_id=iid, messages=messages[:index], assistant=message,
                   tools=schemas, reasoning_tokens=reasoning_tokens,
                   teacher_duration_seconds=outcome.get('duration_seconds'),
                   trace_sha256=fingerprints[iid],
                   teacher_metadata=document.get('extra', {}),
                   teacher_step_metadata=[{'step_id': step.get('step_id'),
                                            'source': step.get('source'),
                                            'model_name': step.get('model_name'),
                                            'llm_call_count': step.get('llm_call_count'),
                                            'extra': step.get('extra'),
                                            'metrics': step.get('metrics')}
                                           for step in document.get('steps', [])])
        rows.append(row)
      if not rows:
        raise ValueError('no_assistant_targets')
      if config.data.sequence_mode != 'turn':
        from .sequences import build_sequences
        metadata = {key: value for key, value in rows[0].items() if key in (
            'instance_id', 'teacher_duration_seconds', 'trace_sha256',
            'teacher_metadata', 'teacher_step_metadata', 'tools')}
        total_turns = len(rows)
        rows, omissions = build_sequences(tokenizer, messages, schemas, config)
        for row in rows:
          row.update(metadata)
        coverage[iid] = {'assistant_turns': total_turns,
                         'supervised_turns': sum(len(row['target_turns']) for row in rows),
                         'excluded_targets': omissions,
                         'windows_with_omitted_history': sum(bool(row['omitted_history_turns']) for row in rows)}
      accepted[iid] = rows
    except (ValueError, KeyError, TypeError) as error:
      rejected.append({'instance_id': iid, 'reason': str(error)})
  ids = sorted(accepted)
  if len(ids) < 2:
    report = config.data.prepared.parent / f'{config.data.prepared.name}-excluded.json'
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(rejected, indent=2) + '\n')
    raise ValueError(f'Need at least two valid passing tasks for a task-separated split; exclusions={rejected}')
  random.Random(config.data.seed).shuffle(ids)
  count = max(1, min(len(ids) - 1, round(len(ids) * config.data.validation_fraction)))
  splits = {'validation': sorted(ids[:count]), 'train': sorted(ids[count:])}
  manifest = {
      'schema_version': 1, 'objective': 'hard_distillation_sft',
      'config_fingerprint': config.fingerprint(), 'harness': contract,
      'base_source': config.model.source,
      'requested_base_revision': config.model.revision,
      'base_revision': getattr(tokenizer, 'init_kwargs', {}).get('_commit_hash') or revision,
      'tokenizer_sha256': digest({'vocab': tokenizer.get_vocab(),
                                  'template': tokenizer.chat_template}),
      'splits': splits, 'trace_hashes': fingerprints, 'outcomes_sha256': digest(outcomes),
      'excluded': rejected,
      'counts': {split: sum(len(accepted[i]) for i in task_ids) for split, task_ids in splits.items()},
      'supervised_tokens': {split: sum(sum(v != -100 for v in r['labels'])
                                      for i in task_ids for r in accepted[i])
                            for split, task_ids in splits.items()},
      'teacher_budget_provenance': 'Provided harness contract; historical enforcement is not inferred from ATIF.',
  }
  if config.data.sequence_mode != 'turn':
    from .sequences import sequence_contract
    manifest.update(schema_version=2, objective='offline_teacher_trajectories',
                    sequence=sequence_contract(config), coverage=coverage,
                    trajectory_counts={split: len(task_ids) for split, task_ids in splits.items()},
                    loss_normalization='optimizer_step_supervised_tokens')
  examples = {split: (row for iid in task_ids for row in accepted[iid])
              for split, task_ids in splits.items()}
  return write_prepared(config, manifest, examples)


def write_prepared(config, manifest, examples):
  """Publish either source format through the same immutable artifact layout."""
  destination = config.data.prepared
  destination.parent.mkdir(parents=True, exist_ok=True)
  if destination.exists():
    raise FileExistsError(f'Prepared data already exists: {destination}; choose a new data.prepared path')
  with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
    stage = Path(temporary) / 'prepared'
    stage.mkdir()
    for split, rows in examples.items():
      with (stage / f'{split}.jsonl').open('w') as handle:
        for row in rows:
          handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    manifest['dataset_hashes'] = {split: digest((stage / f'{split}.jsonl').read_text()) for split in examples}
    manifest['fingerprint'] = digest(manifest)
    (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    stage.rename(destination)
  return manifest
