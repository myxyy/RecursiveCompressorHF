"""Main architecture, phase embeddings off, fixed-M10 task comparison."""
import datetime
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROOT = Path('/mnt/raid0/RecursiveCompressor/experiments/logkv-no-level-bias-20261014')
BASE_ROOT = Path('/mnt/raid0/RecursiveCompressor/experiments/logkv-causal-conv-20260914')
SOURCE = ROOT / 'source'
BASE_INITIAL = BASE_ROOT / 'initial_model/model.safetensors'
GPUS = (0, 1)
NUM_LAYERS = 2
NUM_PARAMS = 5792256
INITIAL = ROOT / 'initial_model/model.safetensors'
MODES = ('copying', 'selective-copying')

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n'); tmp.replace(path)
def name(mode):
    assert mode in MODES
    return 'causal-conv4-no-level-bias-fixed10-20261014'
def run_dir(mode): return ROOT / 'exp' / mode / name(mode)
def task_identity(mode=None):
    mode = mode or os.environ['LOGKV_TASK']; assert mode in MODES
    path = SOURCE / 'exp' / mode.replace('-', '_') / 'task.py'
    return dict(task=mode, task_path=str(path), task_sha256=sha(path))
def bind_task(mode):
    import importlib
    module = importlib.import_module('exp.' + mode.replace('-', '_') + '.task')
    assert module.TASK_NAME == mode
    assert Path(module.__file__).resolve() == Path(task_identity(mode)['task_path']).resolve()
    return module
def command(mode, python):
    return [python, str(HERE / 'train_entry.py'), '--run-name', name(mode),
        '--arch', 'logkv', '--no-level-bias', '--gated-attention', '--self-slot', '--phase-levels', '2',
        '--t-dist', 'loguniform', '--max-t', '2028', '--steps', '50000',
        '--batch-size', '64', '--grad-accum', '1', '--lr', '0.0003', '--warmup', '1000',
        '--d-model', '512', '--num-heads', '8', '--d-ff', '1024', '--num-layers', str(NUM_LAYERS), '--conv-kernel-size', '4',
        '--chunk-size', '4', '--loss-positions', 'all', '--seed', '0', '--device', '0']
