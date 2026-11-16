"""Evaluate baseline and exported adapters with the real SWEGemma harness.

Run this optional command in a vLLM-capable environment with the private harness
packages installed. Training does not depend on vLLM or the harness runtime.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

import yaml

from .artifacts import write_json
from .config import Config
from .harness import read_harness, snapshot_harness
from .runner import load_prepared


def phase_summary(results, allowed_tools):
  """Include trace-backed invalid-call counts, explicitly marking missing traces."""
  invalid, inspected = 0, 0
  for result in results:
    path = getattr(result, 'trace_json_path', None)
    if not path or not Path(path).is_file():
      continue
    try:
      trace = json.loads(Path(path).read_text())
    except (OSError, ValueError):
      continue
    if not isinstance(trace, dict) or not isinstance(trace.get('steps'), list):
      continue
    inspected += 1
    for step in trace.get('steps', []):
      for call in step.get('tool_calls') or []:
        arguments = call.get('arguments', {})
        if isinstance(arguments, str):
          try:
            arguments = json.loads(arguments)
          except ValueError:
            arguments = None
        invalid += int(call.get('function_name') not in allowed_tools or not isinstance(arguments, dict))
  passed = sum(bool(r.resolved and r.test_exit_code == 0) for r in results)
  return {'passed': passed, 'tasks': len(results),
          'resolution_rate': passed / len(results) if results else 0.0,
          'tool_calls': sum(r.tool_calls for r in results),
          'duration_seconds': sum(r.duration_seconds for r in results),
          'invalid_tool_calls': invalid if inspected == len(results) else None,
          'traces_inspected': inspected,
          'invalid_call_definition': 'unknown tool name or malformed/non-object JSON arguments; excludes schema/runtime tool errors'}


def evaluate(config: Config, adapter: Path) -> dict:
  from adk_submission import VllmConfig, VllmServer, discover_adapters
  from huggingface_hub import snapshot_download
  from swegemma.config import EvalConfig, build_submission_limits
  from swegemma.evaluate import Evaluator
  from swegemma.models import load_tasks
  from swegemma.results import append_task_result
  from tqdm.auto import tqdm

  contract = read_harness(config)
  prepared = load_prepared(config)
  provenance = json.loads((adapter / 'provenance.json').read_text())
  if provenance['base_source'] != config.model.source or provenance['base_revision'] != prepared['base_revision']:
    raise ValueError('Adapter and configured base checkpoint differ')
  if provenance['harness_fingerprint'] != contract['fingerprint']:
    raise ValueError('Adapter and evaluation harness differ')
  data_root = config.harness.data_root
  if data_root is None:
    raise ValueError('Set harness.data_root for tasks, snapshots, wheels, graphs and embeddings')
  tasks = load_tasks(data_root / 'tasks.jsonl')
  requested = config.evaluation.task_ids or prepared['splits']['validation']
  by_id = {t.instance_id: t for t in tasks}
  if set(requested) - set(by_id):
    raise ValueError('Some evaluation task IDs are absent from tasks.jsonl')
  source = Path(config.model.source)
  if not source.is_dir():
    source = Path(snapshot_download(config.model.source, revision=provenance['base_revision'],
                                    local_files_only=config.model.local_files_only))
  output = config.output.directory / ('evaluation-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
  output.mkdir(parents=True)
  from shutil import copytree
  with tempfile.TemporaryDirectory(prefix='sft-harness-') as directory:
    submission = Path(directory) / 'submission'
    snapshot_harness(contract, submission)
    # The generated inference submission keeps prompts/tools/sampling settings.
    agent = dict(contract['agent'])
    agent['model'] = config.model.source
    agent['adapter'] = config.adapter.name
    (submission / 'agent.yaml').write_text(yaml.safe_dump(agent, sort_keys=False))
    copytree(adapter, submission / 'adapters' / config.adapter.name)
    adapters = discover_adapters(submission)
    maximum_rank = max(config.adapter.rank_pattern.values())
    rank_limit = next((r for r in (8, 16, 32, 64, 128, 256, 512) if r >= maximum_rank), None)
    if rank_limit is None:
      raise ValueError('Rank exceeds supported vLLM adapter rank limits')
    server = VllmServer(VllmConfig(
        model=str(source), port=config.evaluation.port,
        max_model_len=contract['context_length'],
        tensor_parallel_size=config.memory.processes,
        tool_call_parser=config.evaluation.tool_call_parser,
        reasoning_parser=config.evaluation.reasoning_parser,
        enable_lora=True, max_loras=1, max_lora_rank=rank_limit,
        startup_timeout=1200,
    ), adapter_manifest=adapters)

    async def run_phase(label, with_adapter):
      if not with_adapter:
        agent.pop('adapter', None)
      else:
        agent['adapter'] = config.adapter.name
      (submission / 'agent.yaml').write_text(yaml.safe_dump(agent, sort_keys=False))
      models = server.create_model_registry(aliases=[config.model.source])
      limits, constraints = build_submission_limits()
      # Evaluation budgets come from the original eval_config when present.
      eval_file = config.harness.submission_dir / 'eval_config.yaml'
      raw = yaml.safe_load(eval_file.read_text()) if eval_file.is_file() else {}
      budget = raw.get('evaluation', raw) or {}
      allowed = {'timeout_seconds', 'max_tool_calls', 'max_time_minutes', 'max_turns'}
      evaluator = Evaluator(EvalConfig(
          tasks_path=data_root / 'tasks.jsonl', snapshots_dir=data_root / 'snapshots',
          results_dir=output / label, submission_dir=submission, models=models,
          adapter_manifest=adapters, sandbox=config.harness.sandbox,
          wheels_dir=data_root / 'wheels', graph_dir=str(data_root / 'graphs'),
          embeddings_dir=str(data_root / 'embeddings'), limits=limits,
          generation_constraints=constraints,
          **{k: v for k, v in budget.items() if k in allowed},
      ))
      results = []
      try:
        for iid in tqdm(requested, desc=label, disable=not config.progress.enabled):
          result = await evaluator.evaluate_task(by_id[iid])
          results.append(result)
          append_task_result(result, output / label, running_results=results)
          if result.error:
            print(f'{iid}: {result.error}', flush=True)
      finally:
        await asyncio.to_thread(evaluator.sandbox.cleanup_all)
      return phase_summary(results, set(contract['agent']['tools']))

    report = {'runtime': {k: contract[k] for k in ('context_length', 'completion_tokens', 'thinking_tokens')},
              'reasoning_budget_transport': 'Harness-declared; inspect traces for actual token compliance.'}
    try:
      server.start()
      if config.evaluation.compare_base:
        report['baseline'] = asyncio.run(run_phase('baseline', False))
      report['adapter'] = asyncio.run(run_phase('adapter', True))
    finally:
      server.stop()
    write_json(output / 'comparison.json', report)
  return report
