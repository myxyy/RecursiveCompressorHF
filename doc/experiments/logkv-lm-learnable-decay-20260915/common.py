import hashlib
import json
import os
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROOT = Path('/mnt/raid0/RecursiveCompressor/experiments/logkv-lm-learnable-decay-20260915')
SOURCE = ROOT / 'source'
DATA = ROOT / 'data'
RUN_NAME = 'd1024-h8-l16-conv4-learnable-decay-5000'
PYTHON = REPO / '.venv/bin/python'

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def environment():
    return dict(os.environ, DATA_DIR=str(DATA), CUDA_VISIBLE_DEVICES='0,1,2,3,4,5',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', PYTHONUNBUFFERED='1',
        LOGKV_LM_EXPERIMENT=str(ROOT))

def command(name, steps):
    return [str(PYTHON), '-m','torch.distributed.run','--standalone','--nproc_per_node=6',
        str(HERE/'train_entry.py'), '--run-name',name,'--dataset-type','pretrain',
        '--context-length','2048','--d-model','1024','--num-heads','8','--d-ff','3072',
        '--num-layers','16','--chunk-size','4','--conv-kernel-size','4',
        '--gated-attention','--self-slot','--learnable-decay',
        '--batch-size','4','--grad-accum','1','--lr','0.0002','--warmup','1000',
        '--max-steps',str(steps),'--seed','0','--log-interval','10',
        '--sample-interval','1000','--checkpoint-interval','1000','--max-checkpoints','6',
        '--no-prefault']
