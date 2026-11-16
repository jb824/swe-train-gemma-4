import json

import pytest

from swegemma_training.config import Config


@pytest.fixture
def harness(tmp_path):
  root = tmp_path / 'harness'
  (root / 'configs').mkdir(parents=True)
  (root / 'prompts').mkdir()
  (root / 'prompts/system.md').write_text('Fix the repository with the six tools.')
  (root / 'agent.yaml').write_text('''name: swe_baseline_agent
model: teacher
instruction: !include prompts/system.md
tools: [run_command, read_file, edit_file, write_file, get_status, submit_patch]
generate_content_config: !include configs/sampling.yaml
''')
  (root / 'configs/sampling.yaml').write_text('''temperature: 0.2
max_output_tokens: 12288
thinking_config:
  thinking_budget: 5120
  include_thoughts: true
''')
  return root


@pytest.fixture
def config(tmp_path, harness):
  value = Config()
  value.harness.submission_dir = harness
  value.data.traces = tmp_path / 'traces'
  value.data.outcomes = tmp_path / 'task_results.jsonl'
  value.data.prepared = tmp_path / 'prepared'
  value.output.directory = tmp_path / 'runs'
  value.memory.offload = 'none'
  value.progress.enabled = False
  return value


@pytest.fixture
def tokenizer():
  from tokenizers import Tokenizer
  from tokenizers.models import WordLevel
  from tokenizers.pre_tokenizers import Split
  from transformers import PreTrainedTokenizerFast
  vocab = {'[PAD]': 0, '[EOS]': 1, '[UNK]': 2}
  vocab.update({chr(i): i + 3 for i in range(128)})
  backend = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
  backend.pre_tokenizer = Split('', behavior='isolated')
  result = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]',
                                  eos_token='[EOS]', unk_token='[UNK]')
  result.chat_template = '''{% for m in messages %}{{ m.role }}: {% if m.get('reasoning') %}<think>{{ m.reasoning }}</think>{% endif %}{{ m.get('content', '') }}{% if m.get('tool_calls') %}{{ m.tool_calls | tojson }}{% endif %}{{ eos_token }}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}'''
  return result


def trajectory(iid):
  return {'schema_version': 'ATIF-v1.7', 'instance_id': iid, 'steps': [
      {'source': 'user', 'message': 'Fix the issue'},
      {'source': 'agent', 'message': 'Inspect it', 'extra': {'event_type': 'thinking', 'timestamp': 1}},
      {'source': 'agent', 'extra': {'timestamp': 1}, 'tool_calls': [
          {'function_name': 'read_file', 'tool_call_id': 'r1', 'arguments': {'filepath': 'a.py'}}]},
      {'source': 'system', 'observation': {'content': 'x = 1', 'extra': {'tool_name': 'read_file'}}},
      {'source': 'agent', 'message': 'Done', 'extra': {'event_type': 'final', 'timestamp': 2}},
  ]}


@pytest.fixture
def traces(config):
  config.data.traces.mkdir()
  outcomes = []
  for iid, passing in [('one', True), ('two', True), ('three', True), ('failed', False)]:
    (config.data.traces / f'{iid}.json').write_text(json.dumps(trajectory(iid)))
    outcomes.append({'instance_id': iid, 'resolved': passing,
                     'test_exit_code': 0 if passing else 1,
                     'duration_seconds': 10 if iid == 'one' else 20})
  config.data.outcomes.write_text('\n'.join(json.dumps(row) for row in outcomes))
  return config
