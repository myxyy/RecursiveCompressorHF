"""Post-hoc content-neutral encoder diagnostic using trained positional transforms.

Run in the experimental commit's environment. This does not test the LM readout
or its content-dependent pooling weights. Only the final summary is bf16-rounded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--task',choices=('copying','selective-copying'),required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
sys.path.insert(0,str(args.source.resolve()))
from logkv_lm import LogKVLM
from exp.position_study.probe import pool

torch.set_num_threads(1)
records=[]
with torch.no_grad():
    for mode in ('binding','combined','combined-no-decay'):
        for checkpoint,folder in (('best','model_best'),('final','model')):
            model_path=args.root/'runs'/f'{mode}-{args.task}'/folder
            digest=hashlib.sha256((model_path/'model.safetensors').read_bytes()).hexdigest()
            model=LogKVLM.from_pretrained(model_path).double().eval()
            for layer,block in enumerate(model.layers):
                comp=block.attention.compressor
                gen=torch.Generator().manual_seed(31415)
                symbols=torch.randn(8,8,64,generator=gen,dtype=torch.float64)
                for M in (4,16,64):
                    basis=torch.zeros(M*8,8,M,64,dtype=torch.float64)
                    for position in range(M):
                        basis[position*8:(position+1)*8,:,position]=symbols
                    code=pool(comp,basis).reshape(M*8,512)
                    singular=torch.linalg.svdvals(code)
                    decoder=torch.linalg.pinv(code,rtol=1e-10)
                    digits=torch.randint(8,(256,M),generator=gen)
                    target=torch.nn.functional.one_hot(digits,8).double().reshape(256,M*8)
                    summary=target@code
                    prediction=(summary@decoder).reshape(256,M,8).argmax(-1)
                    rounded=(summary.to(torch.bfloat16).double()@decoder).reshape(256,M,8).argmax(-1)
                    swap_diff=None
                    if M==16:
                        sample=torch.randn(1,8,M,64,generator=gen,dtype=torch.float64)
                        swapped=sample.clone(); swapped[:,:,1]=sample[:,:,4]; swapped[:,:,4]=sample[:,:,1]
                        swap_diff=float((pool(comp,sample)-pool(comp,swapped)).abs().max())
                    rec=dict(task=args.task,mode=mode,checkpoint=checkpoint,layer=layer,memory_len=M,
                             checkpoint_sha256=digest,features=M*8,
                             rank=int((singular>singular[0]*1e-10).sum()),
                             condition_number=float(singular[0]/singular[-1]),
                             token_acc=float((prediction==digits).double().mean()),
                             string_acc=float((prediction==digits).all(-1).double().mean()),
                             bf16_summary_string_acc=float((rounded==digits).all(-1).double().mean()),
                             swap_1_4_max_difference=swap_diff)
                    records.append(rec)
                    if M==64: print(json.dumps(rec),flush=True)
            del model
args.output.write_text(json.dumps(dict(post_hoc=True,description=__doc__,results=records),indent=2)+'\n')
