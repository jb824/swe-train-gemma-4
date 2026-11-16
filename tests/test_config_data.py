import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from swegemma_training.cli import app
from swegemma_training.config import load_config
from swegemma_training.data import atif_messages, outcome_index, prepare, render_target, trace_id
from swegemma_training.harness import read_harness
from swegemma_training.runner import load_prepared
from conftest import trajectory


def test_file_config_roundtrip_paths_overrides_and_profiles(tmp_path, harness):
  runner = CliRunner()
  path = tmp_path / 'nested/train.yaml'
  result = runner.invoke(app, ['config', 'init', '--profile', 'l4x4', '--output', str(path)])
  assert result.exit_code == 0, result.output
  cfg = load_config(
      path, [f'harness.submission_dir={harness}', 'optimizer.learning_rate=0.0002'])
  assert cfg.memory.processes == 4 and cfg.memory.offload == 'none'
  assert cfg.data.traces == path.parent / 'teacher-traces'
  assert cfg.optimizer.learning_rate == 0.0002
  assert not cfg.publish.hf.enabled and not cfg.publish.kaggle.enabled
  assert 'max_output_tokens' not in cfg.model_dump()['runtime']
  for override in ('training.unknown=4', 'memory.offload=inference_auto', 'adapter.rank_pattern.q_proj=0'):
    with pytest.raises(ValueError):
      load_config(path, [override])
  assert runner.invoke(app, ['config', 'init', '--output', str(path)]).exit_code != 0


def test_passing_tasks_are_split_before_examples_and_timings_preserved(traces, tokenizer):
  manifest = prepare(traces, tokenizer, schemas=[])
  assert not set(manifest['splits']['train']) & set(manifest['splits']['validation'])
  assert 'failed' not in sum(manifest['splits'].values(), [])
  rows = [json.loads(line) for split in ('train', 'validation')
          for line in (traces.data.prepared / f'{split}.jsonl').read_text().splitlines()]
  assert len(rows) == 6
  assert {r['teacher_duration_seconds'] for r in rows} == {10, 20}
  assert any(r['assistant'].get('reasoning') == 'Inspect it' and r['assistant'].get('tool_calls') for r in rows)
  for row in rows:
    prefix = row['prompt_tokens']
    assert row['labels'][:prefix] == [-100] * prefix
    assert row['labels'][prefix:] == row['input_ids'][prefix:]
  assert load_prepared(traces)['fingerprint'] == manifest['fingerprint']
  with (traces.data.prepared / 'train.jsonl').open('a') as f:
    f.write('{}\n')
  with pytest.raises(ValueError, match='changed'):
    load_prepared(traces)


def test_failed_launcher_reports_actual_checkpoint_availability(traces, tokenizer, monkeypatch, capsys):
  import subprocess
  from swegemma_training import runner
  from torch.distributed.run import get_args_parser
  prepare(traces, tokenizer, schemas=[])
  monkeypatch.setenv('TORCH_SHOW_CPP_STACKTRACES', '0')
  invoked = {}

  def fail(command, check, env):
    invoked['command'] = command
    invoked['env'] = env
    run = Path(command[command.index('--run') + 1])
    logs = run / 'worker-logs' / 'attempt_0' / '0'
    logs.mkdir(parents=True)
    (logs / 'stderr.log').write_text('Native worker failure\n')
    raise subprocess.CalledProcessError(1, command)

  monkeypatch.setattr(runner.subprocess, 'run', fail)
  with pytest.raises(subprocess.CalledProcessError):
    runner.launch(traces)
  run = next(traces.output.directory.iterdir())
  report = json.loads((run / 'launcher-failure.json').read_text())
  assert report['complete_checkpoints'] == [] and report['torchrun_exit_code'] == 1
  assert (run / 'worker-logs/attempt_0/0/stderr.log').read_text() == 'Native worker failure\n'
  assert 'No complete checkpoint was saved' in capsys.readouterr().err
  assert invoked['env']['PYTHONFAULTHANDLER'] == '1'
  assert invoked['env']['PYTHONUNBUFFERED'] == '1'
  assert invoked['env']['TORCH_SHOW_CPP_STACKTRACES'] == '0'
  # Validate the actual torchrun argument parser without launching a worker.
  parsed = get_args_parser().parse_args(invoked['command'][3:])
  assert parsed.tee == '3' and parsed.log_dir == str(run / 'worker-logs')


def test_ambiguous_tool_results_and_missing_prompt_are_rejected(config):
  contract = read_harness(config)
  document = trajectory('one')
  document['steps'][3]['observation']['extra']['tool_name'] = 'write_file'
  with pytest.raises(ValueError, match='Unmatched'):
    atif_messages(document, contract)
  document = trajectory('one')
  document['steps'].pop(0)
  with pytest.raises(ValueError, match='initial user prompt'):
    atif_messages(document, contract)


def test_overflow_is_quarantined_not_truncated(traces, tokenizer):
  doc = trajectory('one')
  doc['steps'][-1]['message'] = 'x' * 33000
  (traces.data.traces / 'one.json').write_text(json.dumps(doc))
  manifest = prepare(traces, tokenizer, schemas=[])
  assert {'instance_id': 'one', 'reason': 'student_context_overflow'} in manifest['excluded']
  assert 'one' not in sum(manifest['splits'].values(), [])


def test_template_that_drops_reasoning_is_rejected(tokenizer):
  tokenizer.chat_template = "{% for m in messages %}{{m.role}}: {{m.get('content', '')}}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
  with pytest.raises(ValueError, match='discarded'):
    render_target(tokenizer, [{'role': 'user', 'content': 'Hi'}],
                  {'role': 'assistant', 'content': 'yes', 'reasoning': 'why'}, [])


def test_harness_include_cannot_escape_submission(config):
  (config.harness.submission_dir / 'agent.yaml').write_text('instruction: !include ../secret.txt')
  with pytest.raises(ValueError, match='Unsafe'):
    read_harness(config)


def test_actual_harness_trace_filenames_and_kaggle_outcome_logs(tmp_path):
  path = tmp_path / 'teacher.log'
  path.write_text(json.dumps([{'stream_name': 'stdout', 'data':
      '[1/2] Evaluating fastapi_1 (fastapi/fastapi)...\n'
      ' -> resolved=True, exit_code=0, duration=23.5s\n'
      '[2/2] Evaluating fastapi_2 (fastapi/fastapi)...\n'
      ' -> resolved=False, exit_code=-1, duration=0.0s\n'}]))
  records = outcome_index(path)
  assert records['fastapi_1']['duration_seconds'] == 23.5
  assert records['fastapi_2']['resolved'] is False
  assert trace_id(Path('trace_fastapi_1.json'), {}) == 'fastapi_1'


def test_changed_system_prompt_is_rejected(config):
  doc = trajectory('one')
  doc['steps'].insert(0, {'source': 'system', 'message': 'A different harness',
                          'extra': {'event_type': 'system_instruction'}})
  with pytest.raises(ValueError, match='system instruction differs'):
    atif_messages(doc, read_harness(config))


def test_gemma_tokenizer_preserves_reasoning_and_masks_tool_boundaries():
  from transformers import AutoTokenizer
  try:
    tokenizer = AutoTokenizer.from_pretrained(
        'google/gemma-4-12B-it-qat-w4a16-ct', local_files_only=True)
  except OSError:
    pytest.skip('Gemma tokenizer not available in local cache; no network in tests')
  history = [{'role': 'system', 'content': 'Fix the repository.'},
             {'role': 'user', 'content': 'Inspect a.py'}]
  call = {'role': 'assistant', 'reasoning': 'Read the file.', 'tool_calls': [
      {'id': 'one', 'type': 'function',
       'function': {'name': 'read_file', 'arguments': {'filepath': 'a.py'}}}]}
  row = render_target(tokenizer, history, call, [])
  completion = tokenizer.decode(row['input_ids'][row['prompt_tokens']:])
  assert 'Read the file.' in completion and 'read_file' in completion
  assert not completion.endswith('<|tool_response>')
  history += [call, {'role': 'tool', 'name': 'read_file', 'tool_call_id': 'one', 'content': 'x = 1'}]
  row = render_target(tokenizer, history, {'role': 'assistant', 'reasoning': 'Modify it.', 'content': 'Done.'}, [])
  completion = tokenizer.decode(row['input_ids'][row['prompt_tokens']:])
  assert 'Modify it.' in completion and 'Done.' in completion
  assert row['labels'][:row['prompt_tokens']] == [-100] * row['prompt_tokens']
  # A model response with text plus a call after a tool response is rendered as
  # a continued Gemma turn. Its generation-only closer must not reject history.
  next_call = dict(call, content='Let me apply the fix:')
  row = render_target(tokenizer, history, next_call, [])
  assert row['completion_tokens'] > 0


def test_rewriting_actual_prompt_content_is_never_silently_masked(tokenizer):
  tokenizer.chat_template = "{% for m in messages %}{{m.role}}: {% if loop.last and m.role == 'user' %}CHANGED{% else %}{{m.get('content','')}}{% endif %}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
  with pytest.raises(ValueError, match='rewrites'):
    render_target(tokenizer, [{'role': 'user', 'content': 'Original history'}],
                  {'role': 'assistant', 'content': 'Answer'}, [])
