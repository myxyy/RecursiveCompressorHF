"""Full M x T sweep, prefix probes and held-out M=128; atomic incremental results."""
import argparse
import hashlib
import numpy as np
import time
from pathlib import Path
import sys

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from exp.variable_memory.common import MEMORIES, evaluate_cell, atomic_json
from logkv_lm import LogKVLM


def grid():
    ts=set(range(1,15)) | {2**k for k in range(4,18)} | {3*2**(k-1) for k in range(4,17)}
    cells=[(m,t,0,"horizon") for m in MEMORIES for t in sorted(ts)]
    cells += [(m,t,p,"prefix") for m in MEMORIES for t in (16,64,256,2048) for p in (7,15,63)]
    cells += [(128,t,p,"unseen-memory") for t in (16,64,256,2048) for p in (0,15)]
    return cells


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--task",choices=["copying","selective-copying"],required=True)
    p.add_argument("--checkpoint",choices=["best","final"],required=True)
    p.add_argument("--samples",type=int,default=256)
    args=p.parse_args()
    torch.set_num_threads(1); torch.set_float32_matmul_precision("high")
    device=torch.device("cuda")
    folder="model_best" if args.checkpoint=="best" else "model"
    model=LogKVLM.from_pretrained(args.run_dir/folder).float().to(device).eval()
    result=dict(task=args.task,checkpoint=args.checkpoint,seed=12345,samples=args.samples,
                precision="fp32 weights/bf16 autocast",complete=False,results=[])
    path=args.run_dir/f"results_{args.checkpoint}.json"
    if path.exists(): raise FileExistsError(path)
    for m,t,prefix,split in grid():
        start=time.time()
        observations=[]
        cell=evaluate_cell(model,args.task,m,t,prefix,args.samples,12345,device,observations=observations)
        folder=args.run_dir/f"observations_{args.checkpoint}"
        folder.mkdir(exist_ok=True)
        observed=folder/f"M{m}-T{t}-P{prefix}.npz"
        arrays={key:np.concatenate([r[key] for r in observations]) for key in observations[0]}
        np.savez_compressed(observed, **arrays)
        correct=arrays['prediction']==arrays['target']
        assert int(correct.sum())==cell['token_correct']
        assert int(correct.all(-1).sum())==cell['string_correct']
        cell.update(digit_errors=(~correct).sum(0).tolist(),
                    observations=str(observed), observations_sha256=hashlib.sha256(observed.read_bytes()).hexdigest())
        cell.update(split=split,elapsed_sec=time.time()-start)
        result["results"].append(cell); atomic_json(path,result)
        print(f"M={m} T={t} P={prefix} token={cell['token_acc']:.4f} string={cell['string_acc']:.4f}",flush=True)
    result["complete"]=True; atomic_json(path,result)


if __name__=="__main__": main()
