"""Whole conversations and explicit, turn-aligned reduced-context windows."""

import re

from .data import render_target


def render_conversation(tokenizer, messages, tools, supervised=None):
  """Locate target spans by exact token prefixes, never by text offsets."""
  ids = tokenizer.apply_chat_template(messages, tools=tools, enable_thinking=True,
                                      tokenize=True, return_dict=False, add_generation_prompt=False)
  if hasattr(ids, 'tolist'):
    ids = ids.tolist()
  labels = [-100] * len(ids)
  spans = []
  for index, message in enumerate(messages):
    if message['role'] != 'assistant' or (supervised is not None and index not in supervised):
      continue
    target = render_target(tokenizer, messages[:index], message, tools)
    end = len(target['input_ids'])
    if ids[:end] != target['input_ids']:
      boundary = next((i for i, (a, b) in enumerate(zip(ids, target['input_ids'])) if a != b),
                      min(len(ids), end))
      suffix = tokenizer.decode(target['input_ids'][boundary:], skip_special_tokens=False)
      # Gemma omits the final turn closer when another assistant message
      # continues that model turn. Only that control suffix may disappear.
      if end - boundary > 32 or not re.fullmatch(r'(?:<turn\|>|\s)*', suffix):
        raise ValueError(f'Trajectory template rewrites assistant message {index} '
                         '(content reordered across observations or reasoning dropped)')
      end = boundary
    start = target['prompt_tokens']
    labels[start:end] = ids[start:end]
    spans.append({'message_index': index, 'start': start, 'end': end})
  if not spans or not any(v != -100 for v in labels[1:]):
    raise ValueError('Trajectory has no supervised assistant tokens')
  return {'input_ids': ids, 'attention_mask': [1] * len(ids), 'labels': labels,
          'assistant_spans': spans}


def sequence_contract(config):
  return {'mode': config.data.sequence_mode,
          'window': config.data.window.model_dump() if config.data.sequence_mode == 'windowed' else None,
          'mask_version': 1}


def build_sequences(tokenizer, messages, tools, config):
  """Return rows and coverage; never truncate a response or a tool observation."""
  if config.data.sequence_mode == 'trajectory':
    row = render_conversation(tokenizer, messages, tools)
    if len(row['input_ids']) > config.runtime.context_length:
      raise ValueError(f'trajectory_context_overflow: {len(row["input_ids"])} tokens')
    row.update(messages=messages, target_turns=list(range(len(row['assistant_spans']))),
               omitted_history_turns=[], window_index=0)
    return [row], []

  starts = [i for i, message in enumerate(messages) if message['role'] == 'assistant']
  pinned = messages[:starts[0]]
  ends = starts[1:] + [len(messages)]
  units = [messages[a:b] for a, b in zip(starts, ends)]
  budget = config.data.window.max_tokens
  pinned_ids = tokenizer.apply_chat_template(pinned, tools=tools, enable_thinking=True,
                                            tokenize=True, return_dict=False, add_generation_prompt=True)
  if len(pinned_ids) >= budget:
    raise ValueError('window_pinned_context_overflow')
  rows, excluded = [], []
  cursor = 0
  while cursor < len(units):
    history_start = cursor
    # Retain a contiguous suffix of complete exchanges. The history budget is
    # additional to the pinned instruction/task/schema tokens.
    for begin in range(cursor - 1, -1, -1):
      history = [m for unit in units[begin:cursor] for m in unit]
      tokens = tokenizer.apply_chat_template(pinned + history, tools=tools,
          enable_thinking=True, tokenize=True, return_dict=False, add_generation_prompt=True)
      if len(tokens) - len(pinned_ids) > config.data.window.history_tokens:
        break
      history_start = begin

    def render(begin, end):
      transcript = list(pinned)
      targets = set()
      for turn in range(begin, end):
        if turn >= cursor:
          targets.add(len(transcript))
        transcript.extend(units[turn])
      row = render_conversation(tokenizer, transcript, tools, targets)
      row['messages'] = transcript
      return row

    try:
      row = render(history_start, cursor + 1)
    except ValueError as error:
      excluded.append({'turn': cursor, 'reason': str(error)})
      cursor += 1
      continue
    while len(row['input_ids']) > budget and history_start < cursor:
      history_start += 1
      row = render(history_start, cursor + 1)
    if len(row['input_ids']) > budget:
      excluded.append({'turn': cursor, 'reason': 'window_exchange_overflow',
                       'tokens_with_pinned_context': len(row['input_ids'])})
      cursor += 1
      continue
    end = cursor + 1
    while end < len(units):
      try:
        candidate = render(history_start, end + 1)
      except ValueError:
        break  # Finalize valid targets, then report this exchange separately.
      if len(candidate['input_ids']) > budget:
        break
      row = candidate
      end += 1
    row.update(target_turns=list(range(cursor, end)), window_index=len(rows),
               omitted_history_turns=list(range(history_start)),
               history_turns=list(range(history_start, cursor)))
    rows.append(row)
    cursor = end
  if not rows:
    raise ValueError('no_window_targets_fit')
  return rows, excluded
