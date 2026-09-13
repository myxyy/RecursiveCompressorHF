"""Paths and immutable controls for the single-GPU self-slot ablation."""
import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROOT = Path('/mnt/raid0/RecursiveCompressor/experiments/logkv-self-slot-20260910')
SOURCE = Path('/mnt/raid0/RecursiveCompressor/experiments/logkv-phase2-rope-20260908/source')
SOURCE_COMMIT = 'afaea2b1d8b0bbc6e91aba25e1b9a3594297875b'
OLD = Path('/mnt/raid0/RecursiveCompressor/experiments/logkv-rope-only-20260909')
OLD_NAME = 'retrieval-rope-no-phase-fixed10-20260909'
NAME = 'retrieval-rope-no-phase-no-self-fixed10-20260910'
BASELINE = OLD / 'exp/copying' / OLD_NAME
RUN = ROOT / 'exp/copying' / NAME
CONTROL = REPO / 'doc/experiments/logkv-rope-only-20260909'

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
