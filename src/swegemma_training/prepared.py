"""Load token columns from immutable prepared data with bounded conversion memory."""

import json
from pathlib import Path

from .config import digest


def token_rows(source_path: str, split: str):
  with Path(source_path).open() as handle:
    for index, line in enumerate(handle):
      row = json.loads(line)
      yield {key: row[key] for key in ('input_ids', 'attention_mask', 'labels')} | {
          'example_id': f'{split}-{index:06d}'}


def prepared_datasets(config, manifest):
  from datasets import Dataset, Features, List, Value

  features = Features({'input_ids': List(Value('int32')), 'attention_mask': List(Value('int8')),
                       'labels': List(Value('int32')), 'example_id': Value('string')})
  return {split: Dataset.from_generator(
      token_rows, features=features, keep_in_memory=False, writer_batch_size=8,
      cache_dir=str(config.data.prepared / 'arrow-cache'),
      gen_kwargs={'source_path': str(config.data.prepared / f'{split}.jsonl'), 'split': split},
      fingerprint=digest({'token_columns_version': 1, 'split': split, 'data': checksum}),
  ) for split, checksum in manifest['dataset_hashes'].items()}
