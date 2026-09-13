"""Export original/current-source phase2 outputs for a small compatibility comparison."""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
args=p.parse_args();sys.path.insert(0,str(args.source))
from logkv_lm import LogKVLM

torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
checkpoint=Path('/mnt/raid0/RecursiveCompressor/exp/copying/logkv-d512-logu-ph2-gated-self-refined-20260906/model')
data={}
with torch.no_grad():
    for precision in ('bf16','fp32'):
        model=LogKVLM.from_pretrained(checkpoint).to(device='cuda',dtype=torch.bfloat16 if precision=='bf16' else torch.float32).eval()
        digits=torch.randint(1,9,(2,10),generator=torch.Generator().manual_seed(12345)).cuda()
        for N in (16,4096):
            x=torch.zeros(2,N,dtype=torch.long,device='cuda');x[:,:10]=digits
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
                _,hidden=model.step(x)
                logits,_=model.step(torch.full((2,11),9,dtype=torch.long,device='cuda'),hidden)
            data[f'{precision}_{N}']=logits.float().cpu().numpy()
        del model
np.savez_compressed(args.out,**data)
