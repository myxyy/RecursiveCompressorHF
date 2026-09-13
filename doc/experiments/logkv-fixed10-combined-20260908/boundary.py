"""Approved fixed-M10 Copying boundary sweep; one GPU, no training."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/mnt/raid0/RecursiveCompressor/experiments/logkv-fixed10-combined-20260908')
SOURCE = Path('/mnt/raid0/RecursiveCompressor/experiments/logkv-position-study-20260907/source')
CHECKPOINT = ROOT / 'exp/copying/combined-no-decay-fixed10-20260908/model'
OUT = ROOT / 'boundary'
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE / 'exp/copying'))
import torch
from logkv_lm import LogKVLM
from task import make_batch, score_logits


def save(name, value):
    path = OUT / name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


@torch.no_grad()
def main():
    OUT.mkdir(exist_ok=False)
    commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=SOURCE, text=True).strip()
    assert commit == '24b360cf85712fde3ee7a2da4d61e2eb45350a51'
    assert not subprocess.check_output(['git','status','--porcelain'], cwd=SOURCE, text=True)
    weight_hash = hashlib.sha256((CHECKPOINT / 'model.safetensors').read_bytes()).hexdigest()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    model = LogKVLM.from_pretrained(CHECKPOINT).to(device='cuda', dtype=torch.bfloat16).eval()
    data = dict(source_commit=commit, checkpoint=str(CHECKPOINT), checkpoint_sha256=weight_hash,
                task='copying', memory_len=10, prefix=0, samples=256, seed=12345,
                sampling='Reset generator for each T: the same 256 memories at every horizon',
                precision='bf16 weights and bf16 autocast', chunk_len=8192, token_budget=2**19,
                gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), torch_version=torch.__version__,
                started=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                complete=False, results=[])
    answers = dict(targets=None, predictions={})
    start = time.monotonic()
    for T in range(4064,4113):
        tick = time.monotonic()
        generator = torch.Generator().manual_seed(12345)
        batch = max(1, min(256, 2**19 // (T+20)))
        predictions, targets = [], []
        total_token = total_string = 0
        digit_correct = torch.zeros(10, dtype=torch.long)
        for offset in range(0,256,batch):
            x,y = make_batch(T, min(batch,256-offset), generator=generator, device='cuda')
            hidden = None
            with torch.autocast('cuda', dtype=torch.bfloat16):
                for i in range(0, x.shape[1], 8192):
                    logits, hidden = model.step(x[:,i:i+8192],hidden)
            tok,st,_,_ = score_logits(logits.float(),y)
            pred = logits[:,-10:].float().argmax(-1).cpu()
            target = y[:,-10:].cpu()
            total_token += tok; total_string += st
            digit_correct += (pred == target).sum(0)
            predictions += [''.join(map(str,row)) for row in pred.tolist()]
            targets += [''.join(map(str,row)) for row in target.tolist()]
        assert total_token == int(digit_correct.sum())
        assert len(targets) == 256
        if answers['targets'] is None:
            answers['targets'] = targets
        assert targets == answers['targets'], 'Memory samples must be identical across horizons'
        answers['predictions'][str(T)] = predictions
        cell = dict(T=T, n=256, token_correct=total_token, string_correct=total_string,
                    token_acc=total_token/2560, string_acc=total_string/256,
                    digit_correct=digit_correct.tolist(), digit_acc=(digit_correct/256).tolist(),
                    answer_start=T+10, answer_end=T+19, elapsed_sec=time.monotonic()-tick)
        data['results'].append(cell)
        save('results.json',data); save('answers.json',answers)
        print(f'T={T} token={cell["token_acc"]:.4f} string={total_string}/256 '
              f'digits={digit_correct.tolist()}',flush=True)
    data.update(complete=True, elapsed_sec=time.monotonic()-start,
                finished=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
    save('results.json',data)


if __name__ == '__main__':
    main()
