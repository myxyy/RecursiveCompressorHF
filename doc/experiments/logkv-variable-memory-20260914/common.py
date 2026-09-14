import datetime
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROOT = Path('/mnt/raid0/RecursiveCompressor/experiments/logkv-variable-memory-20260914')
SOURCE = ROOT / 'source'
TASKS = ('copying', 'selective-copying')
GPUS = (0, 1)

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); tmp.replace(path)
def audit_source():
    manifest = json.loads((ROOT/'source_manifest.json').read_text())
    for name, digest in manifest['files'].items(): assert sha(SOURCE/name) == digest, name
    return manifest
def env(gpu):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUNBUFFERED='1',
                PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
def train_command(python, task, folder, steps):
    return [python, str(SOURCE/'exp/variable_memory/train.py'), '--mode','causal-conv4',
            '--task',task,'--run-dir',str(folder),'--steps',str(steps)]
def eval_command(python, task, folder, checkpoint, samples=256):
    return [python, str(SOURCE/'exp/variable_memory/evaluate.py'), '--task',task,
            '--run-dir',str(folder),'--checkpoint',checkpoint,'--samples',str(samples)]
