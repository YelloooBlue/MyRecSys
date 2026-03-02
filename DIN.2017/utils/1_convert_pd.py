import gzip
import json
import pickle
import ast
from pathlib import Path

import pandas as pd


def to_df(file_path: Path) -> pd.DataFrame:
  """Read line-delimited json objects to DataFrame (supports .json/.json.gz)."""
  open_fn = gzip.open if file_path.suffix == '.gz' else open
  with open_fn(file_path, 'rt', encoding='utf-8') as fin:
    records = []
    for line in fin:
      line = line.strip()
      if not line:
        continue
      try:
        records.append(json.loads(line))
      except json.JSONDecodeError:
        # Some Amazon files use single-quoted python-literal dicts.
        records.append(ast.literal_eval(line))
  return pd.DataFrame(records)


def resolve_input(raw_data_path: Path, stem: str) -> Path:
  json_path = raw_data_path / f'{stem}.json'
  gz_path = raw_data_path / f'{stem}.json.gz'
  if json_path.exists():
    return json_path
  if gz_path.exists():
    return gz_path
  raise FileNotFoundError(f'Missing {stem}.json or {stem}.json.gz in {raw_data_path}')


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT_DIR / 'raw_data'

reviews_path = resolve_input(RAW_DATA_DIR, 'reviews_Electronics_5')
meta_path = resolve_input(RAW_DATA_DIR, 'meta_Electronics')

# 处理Reviews数据
reviews_df = to_df(reviews_path)
with open(RAW_DATA_DIR / 'reviews.pkl', 'wb') as f:
  pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)

# 处理Meta数据
meta_df = to_df(meta_path)
meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())]
meta_df = meta_df.reset_index(drop=True)
with open(RAW_DATA_DIR / 'meta.pkl', 'wb') as f:
  pickle.dump(meta_df, f, pickle.HIGHEST_PROTOCOL)

print(f'Wrote: {RAW_DATA_DIR / "reviews.pkl"}')
print(f'Wrote: {RAW_DATA_DIR / "meta.pkl"}')
