"""Content-neutral diagnostic: path collisions and linear decoding after pooling.

This is a synthetic encoder/readout, not a trained LM evaluation.
"""
import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from exp.position_study.common import MODES, initialize, atomic_json


@torch.no_grad()
def pool(comp, values):
    # (batch, head, length, head_dim) -> (batch, head, head_dim)
    B,H,L,D=values.shape
    while L>1:
        N=L//4
        v=values.reshape(B*H*N,4,D)
        zero=torch.zeros_like(v)
        _,_,v=comp(zero,zero,v,num_chunks=N)
        values=v.reshape(B,H,N,D); L=N
    return values[:,:,0]


def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--reflections-only",action="store_true")
    args=p.parse_args()
    torch.set_num_threads(1)
    records=[]
    for mode in ("none", "binding", "relative-kv", "combined"):
        model,digest=initialize(mode,0); model=model.double().eval()
        attn=model.layers[0].attention; comp=attn.compressor
        if args.reflections_only and comp.position_permutations is not None:
            comp.position_permutations.copy_(torch.arange(attn.head_dim).expand_as(comp.position_permutations))
        with torch.no_grad():
            q=k=torch.zeros(1,20,attn.head_dim,dtype=torch.float64).repeat(8,1,1)
            v=torch.ones_like(q)
            out,_=attn._attend(q,k,v,None)
            marker_difference=float((out[:,1]-out[:,0]).abs().max())
            # Construct each symbol at each position separately; the encoder
            # is linear because q/k logits are zero. Use its pseudoinverse
            # to diagnose available order information without LM training.
            gen=torch.Generator().manual_seed(31415)
            symbols=torch.randn(8,8,64,generator=gen,dtype=torch.float64)
            for M in (4,16,64):
                basis=torch.zeros(M*8,8,M,64,dtype=torch.float64)
                for position in range(M):
                    basis[position*8:(position+1)*8,:,position]=symbols
                code=pool(comp,basis).reshape(M*8,512)
                singular=torch.linalg.svdvals(code)
                rank=int((singular>singular[0]*1e-10).sum())
                decoder=torch.linalg.pinv(code,rtol=1e-10)
                digits=torch.randint(8,(256,M),generator=gen)
                target=torch.nn.functional.one_hot(digits,8).double().reshape(256,M*8)
                prediction=((target@code)@decoder).reshape(256,M,8).argmax(-1)
                correct=prediction==digits
                rounded=((target@code).to(torch.bfloat16).double()@decoder).reshape(256,M,8).argmax(-1)
                # Swap local positions 1 and 4 in a two-level tree.
                swap_diff=None
                if M==16:
                    sample=torch.randn(1,8,M,64,generator=gen,dtype=torch.float64)
                    swapped=sample.clone(); swapped[:,:,1]=sample[:,:,4]; swapped[:,:,4]=sample[:,:,1]
                    swap_diff=float((pool(comp,sample)-pool(comp,swapped)).abs().max())
                records.append(dict(mode=mode,memory_len=M,rank=rank,features=M*8,
                                    reflections_only=args.reflections_only,
                                    condition_number=float(singular[0]/singular[-1]),
                                    token_acc=float(correct.double().mean()),
                                    string_acc=float(correct.all(-1).double().mean()),
                                    bf16_summary_string_acc=float((rounded==digits).all(-1).double().mean()),
                                    swap_1_4_max_difference=swap_diff,
                                    identical_value_retrieval_difference=marker_difference,
                                    initial_common_sha256=digest))
    atomic_json(args.output,records)


if __name__=="__main__": main()
