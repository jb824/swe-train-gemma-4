import copy
import json
from pathlib import Path

import pytest

from swegemma_training.config import Config, load_config
from swegemma_training.data import prepare
from swegemma_training.harness_jsonl import assistant_char_spans, render_example, source_passed
from swegemma_training.runner import load_prepared


@pytest.fixture
def exported(config, tokenizer, tmp_path):
  tokenizer.chat_template = '''{% set ns = namespace(prev='') %}{% for m in messages %}{% if m.role == 'assistant' %}{% if ns.prev != 'assistant' %}<|turn>model
{% endif %}{{m.get('content','')}}{% if m.get('tool_calls') %}{{m.tool_calls | tojson}}<|tool_response>{% elif loop.last or messages[loop.index].role != 'assistant' %}<turn|>
{% endif %}{% set ns.prev = 'assistant' %}{% elif m.role == 'tool' %}{{m.content}}<tool_response|>{% else %}<|turn>{{m.role}}
{{m.content}}<turn|>
{% set ns.prev = m.role %}{% endif %}{% endfor %}'''
  schemas = [{'type': 'function', 'function': {'name': 'read_file', 'description': 'Read',
              'parameters': {'type': 'object', 'properties': {'filepath': {'type': 'string'}},
                             'required': ['filepath']}}}]
  (config.harness.submission_dir / 'agent.yaml').write_text('''name: swe_baseline_agent
model: teacher
instruction: !include prompts/system.md
tools: [read_file]
generate_content_config: !include configs/sampling.yaml
''')
  config.data.format = 'harness_jsonl'
  config.data.sequence_mode = 'trajectory'
  config.data.train_file = tmp_path / 'source-train.jsonl'
  config.data.validation_file = tmp_path / 'source-validation.jsonl'

  def document(iid, code=0):
    messages = [{'role': 'system', 'content': 'Recorded instruction'},
                {'role': 'user', 'content': 'USER_TASK'},
                {'role': 'assistant', 'content': 'Inspect.'},
                {'role': 'assistant', 'content': '', 'tool_calls': [
                    {'id': 'one', 'type': 'function',
                     'function': {'name': 'read_file', 'arguments': {'filepath': 'a.py'}}}]},
                {'role': 'tool', 'name': 'read_file', 'tool_call_id': 'one', 'content': 'SECRET_OBSERVATION'},
                {'role': 'assistant', 'content': 'Done.'}]
    text = tokenizer.apply_chat_template(messages, tools=schemas, tokenize=False, add_generation_prompt=False)
    spans = [[text.index('Inspect.'), text.index('SECRET_OBSERVATION')],
             [text.index('Done.'), len(text.rstrip())]]
    return {'task_id': iid, 'repository': iid, 'messages': messages, 'tools': schemas,
            'verification_status': 'VERIFIED', 'is_successful_demonstration': True,
            'teacher': {'model': 'teacher', 'duration_sec': 12},
            'test_results': {'environment_validated': True, 'timed_out': False, 'exit_code': code,
                             'fail_to_pass': [{'status': 'PASSED'}], 'pass_to_pass': {'considered': 2, 'passed': 2}},
            'final_patch': 'METADATA_PATCH_MUST_NOT_BE_A_TARGET',
            'rendered': {'enable_thinking': False, 'text': text, 'target_char_spans': spans}}

  documents = {'train': [document('train-one', code=1), document('train-two')],
               'validation': [document('held-out')]}
  def write():
    for split, path in [('train', config.data.train_file), ('validation', config.data.validation_file)]:
      path.write_text('\n'.join(json.dumps(row) for row in documents[split]) + '\n')
  write()
  return config, tokenizer, schemas, documents, write


def test_native_jsonl_preserves_splits_prompts_outcomes_and_masks(exported):
  config, tokenizer, schemas, _, _ = exported
  manifest = prepare(config, tokenizer, schemas)
  assert manifest['counts'] == {'train': 2, 'validation': 1}
  assert manifest['splits'] == {'train': ['train-one', 'train-two'], 'validation': ['held-out']}
  assert manifest['repository_overlap'] == 0 and not manifest['excluded']
  assert load_prepared(config)['fingerprint'] == manifest['fingerprint']
  row = json.loads((config.data.prepared / 'train.jsonl').read_text().splitlines()[0])
  assert row['messages'][0]['content'] == 'Recorded instruction'
  assert row['teacher_metadata']['test_results']['exit_code'] == 1
  assert row['teacher_metadata']['final_patch'].startswith('METADATA_PATCH')
  supervised = tokenizer.decode([v for v in row['labels'] if v != -100]).replace(' ', '')
  assert 'Inspect.' in supervised and 'read_file' in supervised and 'Done.' in supervised
  assert 'SECRET_OBSERVATION' not in supervised and 'USER_TASK' not in supervised
  assert 'Recordedinstruction' not in supervised and 'METADATA_PATCH' not in supervised
  assert '<|tool_response>' not in supervised
  config.data.format = 'atif'
  with pytest.raises(ValueError, match='source format'):
    load_prepared(config)


def test_prompt_override_is_an_ordinary_independent_run_setting(exported, tmp_path):
  config, tokenizer, schemas, documents, _ = exported
  original = copy.deepcopy(documents)
  override = tmp_path / 'alternative.md'
  override.write_text('Alternative instruction')
  config.data.prompt_override = override
  manifest = prepare(config, tokenizer, schemas)
  assert documents == original
  row = json.loads((config.data.prepared / 'train.jsonl').read_text().splitlines()[0])
  assert row['messages'][0]['content'] == 'Alternative instruction'
  assert row['source_prompt_sha256'] != row['effective_prompt_sha256']
  assert 'Recorded instruction' in manifest['training_prompts']['source'].values()
  assert 'Alternative instruction' in manifest['training_prompts']['effective'].values()
  assert load_prepared(config)['fingerprint'] == manifest['fingerprint']
  override.write_text('Edited after preparation')
  with pytest.raises(ValueError, match='prompt override changed'):
    load_prepared(config)


@pytest.mark.parametrize('failure', ['observation_loss', 'rendered_text', 'unmatched_result', 'tool_schema', 'missing_result'])
def test_bad_records_cannot_leak_into_supervision(exported, failure):
  config, tokenizer, schemas, documents, write = exported
  row = documents['train'][0]
  if failure == 'observation_loss':
    text = row['rendered']['text']
    row['rendered']['target_char_spans'] = [[text.index('SECRET_OBSERVATION'), text.index('SECRET_OBSERVATION') + 18]]
  elif failure == 'rendered_text':
    row['rendered']['text'] += 'changed'
  elif failure == 'unmatched_result':
    row['messages'][4]['tool_call_id'] = 'missing'
  elif failure == 'tool_schema':
    row['tools'] = copy.deepcopy(row['tools'])
    row['tools'][0]['function']['parameters']['required'] = []
  else:
    del row['messages'][4]
  write()
  manifest = prepare(config, tokenizer, schemas)
  assert manifest['counts']['train'] == 1
  assert manifest['excluded'][0]['instance_id'] == 'train-one'


def test_split_duplicates_fail_instead_of_reshuffling(exported):
  config, tokenizer, schemas, documents, write = exported
  documents['validation'][0]['task_id'] = documents['train'][0]['task_id']
  write()
  with pytest.raises(ValueError, match='Duplicate task'):
    prepare(config, tokenizer, schemas)
  assert not config.data.prepared.exists()


def test_jsonl_preparation_tracks_source_mutation(exported):
  config, tokenizer, schemas, documents, write = exported
  prepare(config, tokenizer, schemas)
  documents['train'][0]['teacher']['model'] = 'changed-teacher'
  write()
  with pytest.raises(ValueError, match='source files'):
    load_prepared(config)


def test_publisher_verification_is_distinct_from_raw_exit_status(exported):
  _, _, _, documents, _ = exported
  row = documents['train'][0]
  assert row['test_results']['exit_code'] == 1 and source_passed(row)
  row['test_results']['fail_to_pass'][0]['status'] = 'FAILED'
  assert not source_passed(row)
  row['test_results']['fail_to_pass'][0]['status'] = 'PASSED'
  row['verification_status'] = 'PARTIALLY_VERIFIED'
  assert not source_passed(row)


def test_file_config_native_paths_and_mode_validation(tmp_path):
  path = tmp_path / 'native.yaml'
  path.write_text('''data:
  format: harness_jsonl
  sequence_mode: trajectory
  train_file: ./train.jsonl
  validation_file: ./val.jsonl
  prompt_override: ./prompt.md
''')
  config = load_config(path)
  assert config.data.train_file == tmp_path / 'train.jsonl'
  assert config.data.prompt_override == tmp_path / 'prompt.md'
  for data in ({'format': 'harness_jsonl'}, {'prompt_override': './prompt.md'}):
    with pytest.raises(ValueError):
      Config.model_validate({'data': data})


@pytest.mark.parametrize('objective', ['sft', 'soft'])
def test_imported_trajectories_train_through_shared_runner(exported, tmp_path, objective):
  import torch
  from datasets import Dataset
  from trl import SFTConfig, SFTTrainer
  from swegemma_training.modeling import Collator, attach_adapter, training_model_class
  from swegemma_training.objectives import reference_support
  from swegemma_training.reference import build_cache, prepared_rows
  from swegemma_training.runner import trainer_type
  from test_training import tiny_config
  config, tokenizer, schemas, _, _ = exported
  config.objective.name = objective
  config.objective.soft.cache_dir = tmp_path / 'reference'
  manifest = prepare(config, tokenizer, schemas)
  cache = None
  if objective == 'soft':
    def predict(row):
      gold = torch.tensor([v for v in row['labels'][1:] if v != -100])
      return reference_support(torch.zeros(len(gold), 256), gold, 32)
    build_cache(config, manifest, predict)
    cache = config.objective.soft.cache_dir
  tiny = tiny_config()
  tiny.max_position_embeddings = 2048
  model = training_model_class()(tiny)
  model.token_weighted = True
  model.requires_grad_(False)
  model, _ = attach_adapter(model, config)
  data = {key: Dataset.from_list(rows) for key, rows in prepared_rows(config).items()}
  args = SFTConfig(output_dir=str(tmp_path / 'checkpoints'), use_cpu=True, bf16=False,
      max_steps=1, per_device_train_batch_size=1, gradient_accumulation_steps=2,
      save_steps=1, report_to=[], disable_tqdm=True, remove_unused_columns=False,
      dataset_kwargs={'skip_prepare_dataset': True}, average_tokens_across_devices=True)
  trainer = trainer_type(SFTTrainer)(model=model, args=args, processing_class=tokenizer,
      train_dataset=data['train'], eval_dataset=data['validation'], data_collator=Collator(0, cache))
  trainer.model_accepts_loss_kwargs = True
  trainer.train()
  assert trainer.state.global_step == 1
  assert torch.isfinite(torch.tensor(trainer.evaluate()['eval_loss']))
  assert (tmp_path / 'checkpoints/checkpoint-1/adapter_model.safetensors').is_file()


def test_jsonl_spans_with_actual_gemma_and_source_record(config):
  from transformers import AutoTokenizer
  root = Path('/home/josiah/dev/kaggle/gemma-4-developer-agent_nemotron_v3.5')
  source = root / 'datasets/vishalxyzabc7527/gemma4-swe-agent-sft-harness/data/sft_train.jsonl'
  token_path = root / 'runs/20261114T170002Z-c5ddcf43/adapter'
  if not source.is_file() or not token_path.is_dir():
    pytest.skip('Optional local dataset/tokenizer integration fixture unavailable')
  tokenizer = AutoTokenizer.from_pretrained(token_path, local_files_only=True)
  with source.open() as handle:
    document = json.loads(next(handle))
  text, spans = assistant_char_spans(tokenizer, document['messages'], document['tools'], False)
  assert text == document['rendered']['text'] and len(spans) == 6
  from swegemma_training.harness import read_harness
  row = render_example(document, tokenizer, config, read_harness(config), None)
  assert len(row['input_ids']) == document['rendered']['total_tokens']
  supervised = tokenizer.decode([v for v in row['labels'] if v != -100])
  assert 'Checking the validation code.' in supervised and 'call:edit_file' in supervised
  assert 'response:run_command' not in supervised
