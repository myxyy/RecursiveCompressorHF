"""Fixed-M10 Selective Copying: matched local and aligned RoPE models."""
import datetime
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROOT = Path('/mnt/raid0/RecursiveCompressor/experiments/logkv-aligned-selective-20260913')
COPY_ROOT = Path('/mnt/raid0/RecursiveCompressor/experiments/logkv-aligned-rope-20260913')
SOURCE = COPY_ROOT / 'source'
INITIAL = COPY_ROOT / 'initial_model/model.safetensors'
MODES = ('local-control', 'aligned')
TASK_NAME = 'selective-copying'
TASK_SOURCE = SOURCE / 'exp/selective-copying/task.py'
OLD = REPO / 'doc/experiments/logkv-rope-only-20260909/selective-copying/retrieval-rope.json'
OLD_CONFIG = OLD.parent / 'retrieval-rope/run_config.json'


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


def name(mode):
    return f'{mode}-fixed10-20260913'


def run_dir(mode):
    return ROOT / 'exp' / TASK_NAME / name(mode)


def task_identity():
    return dict(task=TASK_NAME, task_path=str(TASK_SOURCE), task_sha256=sha(TASK_SOURCE))


def bind_selective_task():
    """Bind the frozen task before the shared Copying trainer/evaluator imports it."""
    if 'task' not in sys.modules:
        spec = importlib.util.spec_from_file_location('task', TASK_SOURCE)
        module = importlib.util.module_from_spec(spec)
        sys.modules['task'] = module
        spec.loader.exec_module(module)
    module = sys.modules['task']
    assert module.TASK_NAME == TASK_NAME
    assert Path(module.__file__).resolve() == TASK_SOURCE.resolve()
    assert Path(module.make_batch.__code__.co_filename).resolve() == TASK_SOURCE.resolve()
    return module


def command(mode, python):
    assert mode in MODES
    cmd = json.loads(OLD.read_text())['commands'][0]['command'].copy()
    cmd[0] = python
    cmd[1] = str(HERE / 'train_entry.py')
    cmd[cmd.index('--run-name') + 1] = name(mode)
    if mode == 'aligned':
        cmd.remove('--retrieval-rope')
        cmd.extend(['--aligned-rope', '--aligned-rope-scale', '1.0'])
    return cmd
