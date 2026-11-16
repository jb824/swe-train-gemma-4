"""Internal torchrun entrypoint; users invoke swe-train train."""

import argparse
import os
from pathlib import Path

from .config import load_config
from .artifacts import write_json
from .runner import train_worker


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=Path, required=True)
  parser.add_argument('--run', type=Path, required=True)
  parser.add_argument('--resume', type=Path)
  args = parser.parse_args()
  failure = args.run / 'failure.json'
  if int(os.environ.get('RANK', '0')) == 0:
    failure.unlink(missing_ok=True)
  try:
    train_worker(load_config(args.config), args.run, args.resume)
  except Exception as error:
    if int(os.environ.get('RANK', '0')) == 0 and not failure.exists():
      write_json(failure, {'error_type': type(error).__name__, 'message': str(error),
                           'phase': 'initialization', 'context_changed': False})
    raise


if __name__ == '__main__':
  main()
