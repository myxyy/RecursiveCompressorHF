"""Worst-length training smoke on one GPU; save finite-gradient and memory evidence."""
import argparse
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from exp.position_study.common import initialize, atomic_json
from exp.position_study.task import make_batch


def main():
    p=argparse.ArgumentParser(); p.add_argument("--mode",required=True)
    p.add_argument("--output",type=Path,required=True); args=p.parse_args()
    torch.set_num_threads(1); torch.set_float32_matmul_precision("high")
    model,digest=initialize(args.mode,0); model.cuda().train()
    params={n:p.detach().clone() for n,p in model.named_parameters()
            if any(k in n for k in ("position_vectors","relative_key","relative_value"))}
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0)
    generator=torch.Generator().manual_seed(1)
    torch.cuda.reset_peak_memory_stats(); start=time.time()
    for _ in range(2):
        x,y=make_batch("copying",64,2028,63,32,generator,"cuda")
        with torch.autocast("cuda",dtype=torch.bfloat16): out=model(x,labels=y)
        out.loss.backward()
        assert torch.isfinite(out.loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    changed={n:float((dict(model.named_parameters())[n]-before).abs().max()) for n,before in params.items()}
    assert all(v>0 for v in changed.values())
    atomic_json(args.output,dict(mode=args.mode,initial_common_sha256=digest,
                loss=float(out.loss),seconds=time.time()-start,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                extra_parameter_updates=changed,all_gradients_finite=True,
                all_master_fp32=all(p.dtype==torch.float32 for p in model.parameters()),
                microbatch=32,length=2219,optimizer_steps=2))


if __name__=="__main__": main()
