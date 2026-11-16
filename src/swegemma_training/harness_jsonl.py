"""Import verified conversations with recorded prompts and native Gemma spans."""

import bisect
import copy
import json
import re

from tqdm.auto import tqdm

from .artifacts import file_sha256
from .config import digest
from .data import write_prepared
from .sequences import sequence_contract


def preparation_identity(config):
  paths = {'train': config.data.train_file, 'validation': config.data.validation_file}
  return {'format': 'harness_jsonl',
          'files': {split: {'path': str(path), 'sha256': file_sha256(path)} for split, path in paths.items()},
          'prompt_override': ({'path': str(config.data.prompt_override),
                               'sha256': file_sha256(config.data.prompt_override)}
                              if config.data.prompt_override else None),
          'renderer_version': 1}


def validate_messages(messages, tools, schemas):
  """Check identities and arguments while preserving asynchronous source order."""
  if not messages or messages[0].get('role') != 'system' or not isinstance(messages[0].get('content'), str):
    raise ValueError('Conversation requires a recorded initial system prompt')
  if not any(m.get('role') == 'user' for m in messages):
    raise ValueError('Conversation has no user task')
  expected = {t['function']['name']: t['function']['parameters'] for t in schemas}
  supplied = {t['function']['name']: t['function']['parameters'] for t in tools}
  if set(supplied) != set(expected) or len(supplied) != len(tools):
    raise ValueError('Conversation tools differ from the configured harness')
  for name, parameters in supplied.items():
    target = expected[name]
    if (set(parameters.get('required', [])) != set(target.get('required', [])) or
        {k: v.get('type') for k, v in parameters.get('properties', {}).items()} !=
        {k: v.get('type') for k, v in target.get('properties', {}).items()}):
      raise ValueError(f'Tool argument schema differs from the harness: {name}')
  pending, seen, interleaved = {}, set(), 0
  for index, message in enumerate(messages):
    role = message.get('role')
    if role not in {'system', 'user', 'assistant', 'tool'} or (role == 'system' and index != 0):
      raise ValueError(f'Unsupported message role/boundary: {role}')
    if not isinstance(message.get('content', ''), str):
      raise ValueError('Only text conversations are supported by this importer')
    if role == 'assistant' and pending:
      interleaved += 1
    if role == 'user' and pending:
      raise ValueError('User task boundary precedes pending tool results')
    if role != 'assistant' and message.get('tool_calls'):
      raise ValueError('Only assistant messages may issue tool calls')
    for call in message.get('tool_calls', []):
      name, cid = call['function']['name'], call['id']
      arguments = call['function'].get('arguments', {})
      if not isinstance(arguments, dict):
        raise ValueError('Tool arguments must already be decoded JSON objects')
      if name not in expected or cid in seen or not cid:
        raise ValueError(f'Unknown tool or duplicate call ID: {name}/{cid}')
      parameters = expected[name]
      if (set(parameters.get('required', [])) - arguments.keys() or
          arguments.keys() - parameters.get('properties', {}).keys()):
        raise ValueError(f'Tool arguments differ from the harness: {name}')
      pending[cid] = name
      seen.add(cid)
    if role == 'tool':
      cid = message.get('tool_call_id')
      if cid not in pending or message.get('name', pending.get(cid)) != pending[cid]:
        raise ValueError('Unmatched or misnamed tool result')
      del pending[cid]
  if pending:
    raise ValueError('Conversation ends with outstanding tool calls')
  return interleaved


def common_prefix(left, right):
  low, high = 0, min(len(left), len(right))
  while low < high:
    middle = (low + high + 1) // 2
    if left[:middle] == right[:middle]:
      low = middle
    else:
      high = middle - 1
  # A closer and the following call both start with '<'. Keep that complete
  # control marker outside the shared prefix rather than cutting it in half.
  if low and left[low - 1:low] == '<' and left[low:].startswith('turn|>'):
    low -= 1
  return low


def merge_spans(spans):
  merged = []
  for start, end in spans:
    if merged and start <= merged[-1][1]:
      merged[-1][1] = max(end, merged[-1][1])
    else:
      merged.append([start, end])
  return merged


def assistant_char_spans(tokenizer, messages, tools, thinking):
  """Derive ownership from completed prefixes, without generation-only headers."""
  def render(prefix):
    return tokenizer.apply_chat_template(prefix, tools=tools, enable_thinking=thinking,
        tokenize=False, add_generation_prompt=False)

  text = render(messages)
  spans = []
  for index, message in enumerate(messages):
    if message['role'] != 'assistant':
      continue
    before, after = render(messages[:index]), render(messages[:index + 1])
    start, end = common_prefix(before, text), common_prefix(after, text)
    if any(not re.fullmatch(r'(?:<turn\|>|\s)*', suffix)
           for suffix in (before[start:], after[end:])):
      raise ValueError('Template reorders assistant content across observations or drops recorded reasoning')
    if text.startswith('<|turn>model\n', start):
      start += len('<|turn>model\n')
    if text[:end].endswith('<|tool_response>'):
      end -= len('<|tool_response>')
    while end > start and text[end - 1].isspace():
      end -= 1
    if start < end:
      spans.append([start, end])
  return text, merge_spans(spans)


def normalize_recorded_spans(text, spans):
  result, previous = [], 0
  for start, end in spans:
    if not 0 <= previous <= start < end <= len(text):
      raise ValueError('Invalid recorded target character spans')
    previous = end
    if text[:end].endswith('<|tool_response>'):
      end -= len('<|tool_response>')
    while end > start and text[end - 1].isspace():
      end -= 1
    if start < end:
      result.append([start, end])
  return merge_spans(result)


def render_example(document, tokenizer, config, contract, prompt_override):
  messages, tools = document['messages'], document['tools']
  recorded = document.get('rendered') or {}
  thinking = recorded.get('enable_thinking')
  if not isinstance(thinking, bool):
    raise ValueError('Record must declare rendered.enable_thinking for reproducible formatting')
  text, spans = assistant_char_spans(tokenizer, messages, tools, thinking)
  if text != recorded.get('text'):
    raise ValueError('Recorded text differs from the pinned student template; explicit conversion is required')
  if spans != normalize_recorded_spans(text, recorded.get('target_char_spans', [])):
    raise ValueError('Recorded targets disagree with independently derived assistant ownership')
  original_prompt = messages[0]['content']
  if prompt_override is not None:
    messages = copy.deepcopy(messages)
    messages[0]['content'] = prompt_override
    text, spans = assistant_char_spans(tokenizer, messages, tools, thinking)
  encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
  ids, offsets = encoded['input_ids'], encoded['offset_mapping']
  if len(ids) > contract['context_length']:
    raise ValueError(f'student_context_overflow: {len(ids)} tokens')
  starts, ends = [a for a, _ in offsets], [b for _, b in offsets]
  labels, token_spans = [-100] * len(ids), []
  for start, end in spans:
    left, right = bisect.bisect_left(starts, start), bisect.bisect_right(ends, end)
    if left >= right or offsets[left][0] != start or offsets[right - 1][1] != end:
      raise ValueError('Assistant target boundary crosses a tokenizer token')
    if right - left > contract['completion_tokens']:
      raise ValueError('student_completion_overflow')
    labels[left:right] = ids[left:right]
    token_spans.append({'start': left, 'end': right})
  if not any(v != -100 for v in labels[1:]):
    raise ValueError('Conversation has no supervised assistant tokens')
  for message in messages:
    reasoning = message.get('reasoning') or message.get('reasoning_content') or ''
    if len(tokenizer.encode(reasoning, add_special_tokens=False)) > contract['thinking_tokens']:
      raise ValueError('student_thinking_overflow')
  return {'input_ids': ids, 'attention_mask': [1] * len(ids), 'labels': labels,
          'assistant_spans': token_spans, 'target_turns': list(range(len(token_spans))),
          'messages': messages, 'tools': tools, 'window_index': 0, 'omitted_history_turns': [],
          'source_prompt_sha256': digest(original_prompt),
          'effective_prompt_sha256': digest(messages[0]['content']),
          'rendering': {'enable_thinking': thinking, 'invented_reasoning': False}}


def source_passed(document):
  result = document.get('test_results') or {}
  f2p, p2p = result.get('fail_to_pass') or [], result.get('pass_to_pass') or {}
  return (document.get('verification_status') == 'VERIFIED' and
          document.get('is_successful_demonstration') is True and
          result.get('environment_validated') is True and result.get('timed_out') is False and
          bool(f2p) and all(test.get('status') == 'PASSED' for test in f2p) and
          p2p.get('considered') is not None and p2p.get('passed') == p2p['considered'])


def prepare_jsonl(config, tokenizer, revision, schemas, contract):
  if not tokenizer.is_fast:
    raise ValueError('harness_jsonl requires a tokenizer with exact offset mappings')
  identity = preparation_identity(config)
  prompt = config.data.prompt_override.read_text() if config.data.prompt_override else None
  if prompt is not None and not prompt.strip():
    raise ValueError('Prompt override is empty')
  examples, memberships, hashes, rejected, coverage = {}, {}, {}, [], {}
  source_prompts, effective_prompts, repositories = {}, {}, {}
  seen = set()
  for split, path in (('train', config.data.train_file), ('validation', config.data.validation_file)):
    examples[split], memberships[split], repositories[split] = [], [], set()
    with path.open() as handle:
      for line_number, line in enumerate(tqdm(handle, desc=f'Preparing {split} trajectories',
                                             unit='trajectory', disable=not config.progress.enabled), 1):
        if not line.strip():
          continue
        document = json.loads(line)
        iid = document['task_id']
        if iid in seen:
          raise ValueError(f'Duplicate task across source files/splits: {iid}')
        seen.add(iid)
        hashes[iid] = digest(document)
        if not source_passed(document):
          rejected.append({'instance_id': iid, 'reason': 'not_source_verified_passing'})
          continue
        try:
          interleaved = validate_messages(document['messages'], document['tools'], schemas)
          row = render_example(document, tokenizer, config, contract, prompt)
        except (ValueError, KeyError, TypeError) as error:
          rejected.append({'instance_id': iid, 'reason': str(error)})
          continue
        source_prompts[row['source_prompt_sha256']] = document['messages'][0]['content']
        effective_prompts[row['effective_prompt_sha256']] = row['messages'][0]['content']
        row.update(instance_id=iid, trace_sha256=hashes[iid],
                   teacher_duration_seconds=document.get('teacher', {}).get('duration_sec'),
                   teacher_metadata={key: value for key, value in document.items()
                                     if key not in {'messages', 'tools', 'rendered'}},
                   source_location={'path': str(path), 'line': line_number},
                   source_verification='publisher_fail_to_pass_and_pass_to_pass',
                   interleaved_tool_requests=interleaved)
        examples[split].append(row)
        memberships[split].append(iid)
        repositories[split].add(document.get('repository'))
        coverage[iid] = {'assistant_turns': len(row['assistant_spans']),
                         'supervised_turns': len(row['assistant_spans']), 'excluded_targets': [],
                         'windows_with_omitted_history': 0, 'interleaved_tool_requests': interleaved}
  if not all(examples.values()):
    raise ValueError(f'Both source splits need valid passing trajectories; exclusions={rejected[:10]}')
  manifest = {'schema_version': 2, 'objective': 'offline_teacher_trajectories',
      'source_format': 'harness_jsonl', 'source_preparation': identity,
      'config_fingerprint': config.fingerprint(), 'harness': contract,
      'base_source': config.model.source, 'requested_base_revision': config.model.revision,
      'base_revision': getattr(tokenizer, 'init_kwargs', {}).get('_commit_hash') or revision,
      'tokenizer_sha256': digest({'vocab': tokenizer.get_vocab(), 'template': tokenizer.chat_template}),
      'splits': memberships, 'trace_hashes': hashes, 'excluded': rejected, 'coverage': coverage,
      'sequence': sequence_contract(config), 'loss_normalization': 'optimizer_step_supervised_tokens',
      'trajectory_counts': {key: len(rows) for key, rows in examples.items()},
      'counts': {key: len(rows) for key, rows in examples.items()},
      'supervised_tokens': {key: sum(sum(v != -100 for v in row['labels']) for row in rows)
                            for key, rows in examples.items()},
      'training_prompts': {'source': source_prompts, 'effective': effective_prompts,
                           'override': identity['prompt_override']},
      'repository_overlap': len(repositories['train'] & repositories['validation']),
      'verification_policy': 'Source VERIFIED + successful flag + validated environment + passing F2P/P2P; raw exit codes retained.',
      'teacher_budget_provenance': 'Recorded formatting; historical enforcement of target-harness budgets is not inferred.'}
  return write_prepared(config, manifest, examples)
