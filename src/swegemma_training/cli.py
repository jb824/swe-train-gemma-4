"""Small command surface; heavy dependencies load only for their stage."""

from __future__ import annotations

from importlib.resources import files
import json
from pathlib import Path
import subprocess
from typing import Annotated

import typer
import yaml

from .artifacts import export_adapter, latest_release, publish as publish_release
from .config import load_config


app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
config_app = typer.Typer(no_args_is_help=True)
app.add_typer(config_app, name='config')
ConfigFile = Annotated[Path, typer.Option('--config', exists=True, dir_okay=False)]
Overrides = Annotated[list[str] | None, typer.Option('--set', help='Override dotted.key=YAML_VALUE; repeatable')]


@config_app.command('init')
def config_init(output: Path = typer.Option(Path('train.yaml')),
                profile: str = typer.Option('rocm32'),
                regime: str = typer.Option('turn-sft', help='turn-sft, trajectory-sft, windowed-sft, windowed-soft')):
  """Write an annotated YAML template for a hardware profile."""
  profiles = {'rocm32': ('cpu', 1), 'nvidia48x2': ('none', 2), 'l4x4': ('none', 4)}
  if profile not in profiles:
    raise typer.BadParameter(f'Profile must be one of {list(profiles)}')
  if output.exists():
    raise typer.BadParameter(f'File already exists: {output}')
  if regime not in {'turn-sft', 'trajectory-sft', 'windowed-sft', 'windowed-soft'}:
    raise typer.BadParameter('Unknown training regime')
  text = files('swegemma_training').joinpath('templates/train.example.yaml').read_text()
  offload, count = profiles[profile]
  text = text.replace('  offload: cpu ', f'  offload: {offload} ')
  text = text.replace('  processes: 1\n', f'  processes: {count}\n')
  mode = regime.split('-')[0]
  text = text.replace('  sequence_mode: turn', f'  sequence_mode: {mode}')
  if regime == 'windowed-soft':
    text = text.replace('  name: sft', '  name: soft')
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(text)
  typer.echo(str(output.resolve()))


@config_app.command('validate')
def config_validate(config: ConfigFile, overrides: Overrides = None):
  """Validate the YAML and resolve its harness budgets."""
  from .harness import read_harness
  cfg = load_config(config, overrides or [])
  runtime = read_harness(cfg)
  typer.echo(f'Valid: {runtime["agent"]["name"]}; {len(runtime["agent"]["tools"])} tools; '
             f'context={runtime["context_length"]}, completion={runtime["completion_tokens"]}, '
             f'thinking={runtime["thinking_tokens"]}')


@config_app.command('show')
def config_show(config: ConfigFile, overrides: Overrides = None):
  """Print the resolved configuration and harness contract."""
  from .harness import read_harness
  cfg = load_config(config, overrides or [])
  value = cfg.model_dump(mode='json')
  contract = read_harness(cfg)
  value['resolved_harness'] = {k: contract[k] for k in (
      'context_length', 'completion_tokens', 'thinking_tokens', 'fingerprint')}
  typer.echo(yaml.safe_dump(value, sort_keys=False))


@app.command()
def doctor(config: ConfigFile, overrides: Overrides = None, smoke: bool = False):
  """Inspect dependencies, devices, and training readiness."""
  from .runner import doctor as inspect_environment
  typer.echo(json.dumps(inspect_environment(load_config(config, overrides or []), smoke=smoke), indent=2))


@app.command()
def prepare(config: ConfigFile, overrides: Overrides = None):
  """Build immutable, task-separated examples from passing traces."""
  from .data import prepare as prepare_data
  manifest = prepare_data(load_config(config, overrides or []))
  typer.echo(json.dumps({k: manifest[k] for k in (
      'fingerprint', 'counts', 'supervised_tokens', 'excluded', 'trajectory_counts', 'coverage') if k in manifest}, indent=2))


@app.command('cache-reference')
def cache_reference(config: ConfigFile, overrides: Overrides = None):
  """Cache frozen initial-student probabilities and calibrate SoFT targets."""
  from .reference import launch_reference
  manifest = launch_reference(load_config(config, overrides or []))
  typer.echo(json.dumps({'fingerprint': manifest['fingerprint'], 'calibration': manifest['calibration']}, indent=2))


@app.command()
def train(config: ConfigFile, overrides: Overrides = None,
          resume: Annotated[str | None, typer.Option(help='latest or checkpoint directory')] = None):
  """Train an adapter locally, with resumable checkpoints."""
  from .runner import launch
  typer.echo(str(launch(load_config(config, overrides or []), resume)))


@app.command()
def evaluate(config: ConfigFile, adapter: Annotated[Path, typer.Option(exists=True)],
             overrides: Overrides = None):
  """Compare a release and its base through the SWEGemma harness."""
  from .evaluation import evaluate as evaluate_harness
  typer.echo(json.dumps(evaluate_harness(load_config(config, overrides or []), adapter.resolve()), indent=2))


def newest_run(root: Path) -> Path:
  paths = list(root.glob('*/completed.json'))
  if not paths:
    raise typer.BadParameter('No completed local SFT runs found')
  return max(paths, key=lambda p: p.parent.name).parent


@app.command('export')
def export_command(config: ConfigFile, overrides: Overrides = None,
                   run: Path | None = None, destination: Path | None = None):
  """Locate or create a portable adapter release."""
  cfg = load_config(config, overrides or [])
  run = run.resolve() if run else newest_run(cfg.output.directory)
  provenance = json.loads((run / 'adapter/provenance.json').read_text())
  if destination is None and (run / 'latest-release.json').is_file():
    from .numerics import check_adapter_file
    release = latest_release(run)
    check_adapter_file(release / 'adapter_model.safetensors')
    typer.echo(str(release))
    return
  target = destination.resolve() if destination else run / 'releases/manual'
  if target.exists():
    # A finished run already exports automatically. Do not overwrite releases.
    if (target / 'provenance.json').is_file() and json.loads((target / 'provenance.json').read_text()) == provenance:
      from .numerics import check_adapter_file
      check_adapter_file(target / 'adapter_model.safetensors')
      typer.echo(str(target))
      return
    raise typer.BadParameter('Destination exists with different or missing provenance')
  typer.echo(str(export_adapter(run / 'adapter', target, provenance)))


@app.command('publish')
def publish_command(config: ConfigFile,
                    target: Annotated[str, typer.Option(help='hf or kaggle')],
                    overrides: Overrides = None, release: Path | None = None,
                    version: bool = False):
  """Upload a local release to an explicitly enabled target."""
  if target not in {'hf', 'kaggle'}:
    raise typer.BadParameter('target must be hf or kaggle')
  cfg = load_config(config, overrides or [])
  release = release.resolve() if release else latest_release(newest_run(cfg.output.directory))
  typer.echo(json.dumps(publish_release(cfg, target, release, version=version), indent=2))


def main():
  try:
    app()
  except (ValueError, OSError, RuntimeError, FloatingPointError) as error:
    typer.secho(f'Error: {error}', fg=typer.colors.RED, err=True)
    raise SystemExit(2) from None
  except subprocess.CalledProcessError as error:
    typer.secho(f'Command failed with exit {error.returncode}; see the command output above.',
                fg=typer.colors.RED, err=True)
    raise SystemExit(error.returncode) from None


if __name__ == '__main__':
  main()
