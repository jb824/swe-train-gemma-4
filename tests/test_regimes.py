import copy
import json

import pytest
import torch

from swegemma_training.config import Config, Objective, digest, load_config
from swegemma_training.data import atif_messages, prepare
from swegemma_training.harness import read_harness
from swegemma_training.modeling import Collator, attach_adapter, training_model_class
from swegemma_training.objectives import (
    chunked_objective, construct_targets, probability_floor, reference_support, sparse_soft_loss,
)
from swegemma_training.reference import build_cache, prepared_rows, validate_cache
from swegemma_training.runner import load_prepared, trainer_type
from swegemma_training.sequences import build_sequences, render_conversation
from test_training import tiny_config
from conftest import trajectory


def test_legacy_config_identity_and_new_regime_templates(config, tmp_path):
  from typer.testing import CliRunner
  from swegemma_training.cli import app
  legacy = config.model_dump(mode='json', exclude={'publish', 'progress', 'output', 'evaluation', 'objective'})
  legacy['memory'].pop('cpu_offload_pin_memory')
  legacy['data'].pop('sequence_mode')
  legacy['data'].pop('window')
  for key in ('format', 'train_file', 'validation_file', 'prompt_override'):
    legacy['data'].pop(key)
  assert config.fingerprint() == digest(legacy)
  for regime in ('trajectory-sft', 'windowed-sft', 'windowed-soft'):
    path = tmp_path / f'{regime}.yaml'
    result = CliRunner().invoke(app, ['config', 'init', '--regime', regime, '--output', str(path)])
    assert result.exit_code == 0, result.output
    cfg = load_config(path)
    assert cfg.data.sequence_mode == regime.split('-')[0]
    assert cfg.training.save_steps == 1
    assert cfg.objective.name == ('soft' if regime.endswith('-soft') else 'sft')
    assert cfg.objective.soft.cache_dir == tmp_path / 'reference-cache'
  with pytest.raises(ValueError, match='context'):
    Config.model_validate({'runtime': {'context_length': 1024}, 'data': {'sequence_mode': 'windowed'}})


def test_full_trajectory_masks_all_assistants(traces, tokenizer):
  traces.data.sequence_mode = 'trajectory'
  manifest = prepare(traces, tokenizer, schemas=[])
  assert manifest['schema_version'] == 2
  assert sum(manifest['counts'].values()) == 3
  for rows in prepared_rows(traces).values():
    for row in rows:
      assert len(row['assistant_spans']) == 2
      positions = {i for span in row['assistant_spans'] for i in range(span['start'], span['end'])}
      assert positions == {i for i, v in enumerate(row['labels']) if v != -100}
      assert row['labels'][0] == -100
      assert row['target_turns'] == [0, 1]
  assert load_prepared(traces)['fingerprint'] == manifest['fingerprint']
  traces.data.sequence_mode = 'turn'
  with pytest.raises(ValueError, match='Sequence'):
    load_prepared(traces)


def test_windows_preserve_exchanges_and_supervise_each_turn_once(config, tokenizer):
  messages = [{'role': 'system', 'content': 'Instructions'}, {'role': 'user', 'content': 'Issue'}]
  for i in range(8):
    messages += [{'role': 'assistant', 'content': f'Action {i} ' + 'a' * 20,
                  'tool_calls': [{'id': str(i), 'function': {'name': 'run_command', 'arguments': {}}}]},
                 {'role': 'tool', 'content': 'Observation ' + 'x' * 30, 'tool_call_id': str(i)}]
  config.data.sequence_mode = 'windowed'
  config.data.window.max_tokens = 440
  config.data.window.history_tokens = 200
  rows, excluded = build_sequences(tokenizer, messages, [], config)
  assert not excluded and len(rows) > 1
  assert [v for row in rows for v in row['target_turns']] == list(range(8))
  assert any(row['omitted_history_turns'] for row in rows)
  for row in rows:
    assert len(row['input_ids']) <= 440
    assert row['messages'][:2] == messages[:2]
    assert row['messages'][-1]['role'] == 'tool'
    assert len(row['assistant_spans']) == len(row['target_turns'])
    assert sum(m['role'] == 'assistant' for m in row['messages']) == len(row['history_turns']) + len(row['target_turns'])
  oversized = copy.deepcopy(messages)
  oversized[2]['content'] = 'a' * 1000
  rows, exclusions = build_sequences(tokenizer, oversized, [], config)
  assert exclusions[0]['turn'] == 0
  assert 0 not in [v for row in rows for v in row['target_turns']]
  config.data.window.max_tokens = 10
  with pytest.raises(ValueError, match='pinned'):
    build_sequences(tokenizer, messages, [], config)
  config.data.sequence_mode = 'trajectory'
  config.runtime.context_length = 100
  with pytest.raises(ValueError, match='trajectory_context_overflow'):
    build_sequences(tokenizer, messages, [], config)


def test_trajectory_rejects_template_that_rewrites_history(config, tokenizer):
  messages = atif_messages(trajectory('one'), read_harness(config))
  tokenizer.chat_template = '''{% for m in messages %}{{m.role}}: {% if loop.last %}{{m.get('reasoning', '')}}{% endif %}{{m.get('content', '')}}{{eos_token}}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}'''
  with pytest.raises(ValueError, match='rewrites'):
    render_conversation(tokenizer, messages, [])


@pytest.mark.parametrize('rstar', [0, 0.3, 1])
def test_soft_full_support_matches_dense_loss_and_gradients(rstar):
  torch.manual_seed(42)
  ref = torch.randn(7, 11)
  gold = torch.tensor([0, 1, 2, 3, 4, 5, 6])
  support = reference_support(ref, gold, 11)
  floor = probability_floor(support['gold_probs'], rstar)
  q, tail = construct_targets(support, gold, floor)
  assert torch.allclose(q.sum(-1) + tail, torch.ones(7))
  assert torch.equal(tail, torch.zeros_like(tail)) or tail.abs().max() < 1e-7
  assert q[:, -1].eq(0).all()  # Gold is already in top-K.
  p = ref.softmax(-1)
  p_gold = p.gather(1, gold[:, None]).squeeze(1)
  ratio = (floor - p_gold).clamp_min(0).sum() / (1 - p_gold).sum()
  torch.testing.assert_close(ratio, torch.tensor(float(rstar)))
  dense = p * ((1 - torch.maximum(p_gold, torch.full_like(p_gold, floor))) / (1 - p_gold))[:, None]
  dense.scatter_(1, gold[:, None], torch.maximum(p_gold, torch.full_like(p_gold, floor))[:, None])
  logits = torch.randn(7, 11, requires_grad=True)
  expected = -(dense * logits.log_softmax(-1)).sum()
  expected_grad, = torch.autograd.grad(expected, logits, retain_graph=True)
  actual = sparse_soft_loss(logits, support['ids'], q, tail)
  actual_grad, = torch.autograd.grad(actual, logits)
  torch.testing.assert_close(actual, expected)
  torch.testing.assert_close(actual_grad, expected_grad)
  assert torch.isfinite(actual_grad).all()


def test_topk_tail_includes_gold_once():
  ref = torch.tensor([[9., 4., 1., -2.], [-2., 1., 4., 9.]])
  gold = torch.tensor([0, 0])
  support = reference_support(ref, gold, 2)
  q, tail = construct_targets(support, gold, 0.8)
  assert support['base_probs'][0, -1] == 0
  assert support['base_probs'][1, -1] > 0
  assert torch.allclose(q.sum(-1) + tail, torch.ones(2))
  logits = torch.randn(2, 4, requires_grad=True)
  loss = sparse_soft_loss(logits, support['ids'], q, tail)
  loss.backward()
  assert torch.isfinite(logits.grad).all()


def test_chunked_token_weighting_and_softcap():
  hidden = torch.randn(3, 6, 9, requires_grad=True)
  head = torch.randn(11, 9)
  labels = torch.randint(0, 11, (3, 6))
  labels[0, :4] = -100
  labels[1, :2] = -100
  labels[2, :1] = -100
  count = labels[:, 1:].ne(-100).sum()
  combined = chunked_objective(hidden, head, labels, chunk_tokens=2, softcap=3)
  pieces = sum(chunked_objective(hidden[i:i+1], head, labels[i:i+1], denominator=count,
                                 chunk_tokens=2, softcap=3) for i in range(3))
  torch.testing.assert_close(combined, pieces)
  a, = torch.autograd.grad(combined, hidden, retain_graph=True)
  b, = torch.autograd.grad(pieces, hidden)
  torch.testing.assert_close(a, b)


def test_cache_calibration_identity_collation_and_tamper(traces, tokenizer):
  from swegemma_training.runner import launch
  traces.data.sequence_mode = 'windowed'
  traces.objective = Objective(name='soft')
  traces.objective.soft.cache_dir = traces.output.directory.parent / 'cache'
  manifest = prepare(traces, tokenizer, schemas=[])
  with pytest.raises(ValueError, match='cache-reference'):
    launch(traces)
  assert not traces.output.directory.exists()
  def predict(row):
    gold = torch.tensor([v for v in row['labels'][1:] if v != -100])
    return reference_support(torch.zeros(len(gold), 256), gold, 32)
  result = build_cache(traces, manifest, predict)
  assert len(result['files']) == sum(manifest['counts'].values())
  assert all(abs(s['realized_budget'] - 0.5) < 1e-6 for s in result['calibration'].values())
  rows = next(iter(prepared_rows(traces).values()))
  batch = Collator(0, traces.objective.soft.cache_dir)(rows)
  assert len(batch['soft_ids']) == batch['labels'][:, 1:].ne(-100).sum()
  assert torch.allclose(batch['soft_probs'].sum(-1) + batch['soft_tail'], torch.ones(len(batch['soft_tail'])))
  traces.objective.soft.rstar = 0.7
  with pytest.raises(ValueError, match='stale'):
    validate_cache(traces, manifest)
  traces.objective.soft.rstar = 0.5
  shard = traces.objective.soft.cache_dir / next(iter(result['files']))
  shard.write_bytes(b'changed')
  with pytest.raises(ValueError, match='changed'):
    validate_cache(traces, manifest)


def test_failed_reference_build_does_not_publish_partial_cache(traces, tokenizer):
  traces.objective.name = 'soft'
  traces.objective.soft.cache_dir = traces.output.directory.parent / 'failed-cache'
  manifest = prepare(traces, tokenizer, schemas=[])
  calls = 0

  def predict(row):
    nonlocal calls
    calls += 1
    if calls == 2:
      raise RuntimeError('Reference device failed')
    gold = torch.tensor([v for v in row['labels'][1:] if v != -100])
    return reference_support(torch.zeros(len(gold), 256), gold, 32)

  with pytest.raises(RuntimeError, match='device failed'):
    build_cache(traces, manifest, predict)
  assert not traces.objective.soft.cache_dir.exists()


@pytest.mark.parametrize('soft', [False, True])
def test_token_weighted_trl_updates_and_resume(config, tokenizer, tmp_path, soft):
  """Real Trainer accumulation, partial updates, and adapter checkpoint resume."""
  from datasets import Dataset
  from transformers import set_seed
  trl = pytest.importorskip('trl')
  set_seed(19)
  base = training_model_class()(tiny_config())
  initial = copy.deepcopy(base.state_dict())
  rows = [
      {'input_ids': [2, 10, 11, 12, 1], 'attention_mask': [1]*5, 'labels': [-100, -100, 11, 12, 1]},
      {'input_ids': [2, 10, 11, 12, 1], 'attention_mask': [1]*5, 'labels': [-100, -100, -100, -100, 1]},
      {'input_ids': [2, 10, 11, 12, 1], 'attention_mask': [1]*5, 'labels': [-100, -100, -100, 12, 1]},
  ]
  for i, row in enumerate(rows):
    row['example_id'] = str(i)
  cache_dir = tmp_path / 'soft-cache' if soft else None
  if soft:
    from safetensors.torch import save_file
    cache_dir.mkdir()
    base.eval()
    for row in rows:
      ids = torch.tensor([row['input_ids']])
      labels = torch.tensor([row['labels']])
      with torch.no_grad():
        support = base(input_ids=ids, reference_targets=labels, reference_top_k=256)
      gold = labels[:, 1:][labels[:, 1:].ne(-100)]
      q, tail = construct_targets(support, gold, probability_floor(support['gold_probs'], 0.5))
      save_file({'soft_ids': support['ids'], 'soft_probs': q, 'soft_tail': tail}, cache_dir / f'{row["example_id"]}.safetensors')

  def run(microbatch, accum, destination, epochs=1, resume=None):
    set_seed(19)
    model = training_model_class()(tiny_config())
    model.load_state_dict(initial)
    model.requires_grad_(False)
    model.token_weighted = True
    model, _ = attach_adapter(model, config)
    args = trl.SFTConfig(output_dir=str(destination), use_cpu=True, bf16=False,
        per_device_train_batch_size=microbatch, gradient_accumulation_steps=accum,
        num_train_epochs=epochs, optim='sgd', learning_rate=0.01, lr_scheduler_type='constant',
        max_grad_norm=1000, save_steps=1, disable_tqdm=True, report_to=[],
        dataset_kwargs={'skip_prepare_dataset': True}, remove_unused_columns=False,
        average_tokens_across_devices=True, seed=19)
    trainer = trainer_type(trl.SFTTrainer)(model=model, args=args, processing_class=tokenizer,
        train_dataset=Dataset.from_list(rows), eval_dataset=Dataset.from_list(rows),
        data_collator=Collator(0, cache_dir))
    trainer.model_accepts_loss_kwargs = True
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    assert trainer.label_names == ['labels']
    assert torch.isfinite(torch.tensor(trainer.evaluate()['eval_loss']))
    return {name: value.detach().clone() for name, value in model.named_parameters() if value.requires_grad}
  together = run(3, 1, tmp_path / 'together')
  accumulated = run(1, 3, tmp_path / 'accumulated')
  for name in together:
    torch.testing.assert_close(together[name], accumulated[name], rtol=1e-5, atol=1e-7)
  complete = run(1, 2, tmp_path / 'partial', epochs=2)
  resumed = run(1, 2, tmp_path / 'resumed', epochs=2, resume=tmp_path / 'partial/checkpoint-1')
  for name in complete:
    torch.testing.assert_close(complete[name], resumed[name], rtol=1e-5, atol=1e-7)


def test_windows_report_template_lookahead_without_training_on_future_results(config, tokenizer):
  tokenizer.chat_template = '''{% for m in messages %}{{m.role}}: {% if m.get('tool_calls') %}CALL{% if loop.index < messages|length and messages[loop.index].role == 'tool' %}{{messages[loop.index].content}}{% endif %}{% endif %}{{m.get('content','')}}{{eos_token}}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}'''
  messages = [{'role': 'system', 'content': 'Fix'}, {'role': 'user', 'content': 'Issue'},
              {'role': 'assistant', 'content': 'Narration before result', 'tool_calls': [{'id': '1'}]},
              {'role': 'tool', 'content': 'FUTURE_RESULT'}, {'role': 'assistant', 'content': 'Done'}]
  with pytest.raises(ValueError, match='reordered'):
    render_conversation(tokenizer, messages, [])
  config.data.sequence_mode = 'windowed'
  rows, excluded = build_sequences(tokenizer, messages, [], config)
  assert len(excluded) == 1 and excluded[0]['turn'] == 0
  assert [t for row in rows for t in row['target_turns']] == [1]
  assert 'FUTURE_RESULT' not in tokenizer.decode([t for row in rows for t in row['labels'] if t != -100]).replace(' ', '')


def test_reference_forward_works_through_peft_and_accelerate(config, tokenizer, tmp_path):
  from transformers import Trainer
  trl = pytest.importorskip('trl')
  model = training_model_class()(tiny_config())
  model.requires_grad_(False)
  model, _ = attach_adapter(model, config)
  args = trl.SFTConfig(output_dir=str(tmp_path), use_cpu=True, bf16=False, eval_strategy='no', report_to=[])
  trainer = Trainer(model=model, args=args, processing_class=tokenizer, data_collator=Collator(0))
  trainer.create_optimizer()
  model, _ = trainer.accelerator.prepare(model, trainer.optimizer)
  model.eval()
  ids = torch.tensor([[2, 10, 11, 1]])
  labels = ids.clone()
  labels[:, :2] = -100
  with torch.no_grad(), trainer.accelerator.autocast():
    reference = model(input_ids=ids, reference_targets=labels, reference_top_k=32)
  assert reference['ids'].shape == (2, 33)
  assert torch.isfinite(reference['gold_probs']).all()
  assert all(p.grad is None for p in model.parameters())


def test_native_gemma_trajectory_masks_and_template_reordering(config):
  from transformers import AutoTokenizer
  try:
    tokenizer = AutoTokenizer.from_pretrained(config.model.source, local_files_only=True)
  except OSError:
    pytest.skip('Gemma tokenizer not cached locally')
  messages = [{'role': 'system', 'content': 'Fix the issue.'},
              {'role': 'user', 'content': 'Inspect a.py'},
              {'role': 'assistant', 'reasoning': 'Inspect first.', 'content': '',
               'tool_calls': [{'id': 'r1', 'type': 'function', 'function': {
                   'name': 'read_file', 'arguments': {'filepath': 'a.py'}}}]},
              {'role': 'tool', 'tool_call_id': 'r1', 'name': 'read_file', 'content': 'x = 1'},
              {'role': 'assistant', 'reasoning': 'The code is correct.', 'content': 'Done.'}]
  row = render_conversation(tokenizer, messages, [])
  assert len(row['assistant_spans']) == 2
  supervised = tokenizer.decode([v for v in row['labels'] if v != -100])
  assert 'Inspect first.' in supervised and 'The code is correct.' in supervised
  assert 'x = 1' not in supervised and 'read_file' in supervised
  bad = copy.deepcopy(messages)
  bad[2]['content'] = 'Narration emitted before the observation.'
  with pytest.raises(ValueError, match='reordered'):
    render_conversation(tokenizer, bad, [])


def test_harness_metrics_mark_missing_traces_and_invalid_calls(tmp_path):
  from types import SimpleNamespace
  from swegemma_training.evaluation import phase_summary
  path = tmp_path / 'trace.json'
  path.write_text(json.dumps({'steps': [{'tool_calls': [
      {'function_name': 'read_file', 'arguments': {'filepath': 'a.py'}},
      {'function_name': 'invented', 'arguments': {}},
      {'function_name': 'read_file', 'arguments': 'broken json'},
  ]}]}))
  result = SimpleNamespace(trace_json_path=path, resolved=True, test_exit_code=0, tool_calls=3, duration_seconds=2)
  summary = phase_summary([result], {'read_file'})
  assert summary['invalid_tool_calls'] == 2 and summary['resolution_rate'] == 1
  result.trace_json_path = None
  assert phase_summary([result], {'read_file'})['invalid_tool_calls'] is None
