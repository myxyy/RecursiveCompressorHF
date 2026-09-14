"""Standard CausalConv with the prior variable-memory training protocol."""
import argparse
import json
import math
import time
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from exp.variable_memory.common import MODES, MEMORIES, initialize, evaluate_cell, atomic_json
from exp.variable_memory.task import make_batch, score


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--task", choices=["copying","selective-copying"], required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-t", type=int, default=2028)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--validation-interval", type=int, default=2000)
    args=p.parse_args()
    if args.run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.run_dir}")
    if args.batch_size % args.grad_accum:
        raise ValueError("batch-size must be divisible by grad-accum")
    args.run_dir.mkdir(parents=True)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    device=torch.device("cuda")
    model,initial_hash=initialize(args.mode,args.seed)
    model.save_pretrained(args.run_dir/"initial_model")
    model.to(device).train()
    optimizer=torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0)
    cfg={**vars(args), "run_dir":str(args.run_dir), "memory_lengths":MEMORIES,
         "prefix_range":[0,63], "lr":3e-4, "warmup":1000, "weight_decay":0,
         "grad_clip":1, "precision":"fp32 weights/bf16 autocast", "t_dist":"loguniform",
         "initial_common_sha256":initial_hash, "data_seed":args.seed+1,
         "metadata_seed":args.seed+2, "validation_seed":54321,
         "num_params":sum(x.numel() for x in model.parameters()),
         "checkpoint_selection":"validation macro string, macro token, negative training EMA",
         "loss":"all-position aligned CE", "deterministic_algorithms":True}
    atomic_json(args.run_dir/"run_config.json",cfg)
    data_gen=torch.Generator().manual_seed(args.seed+1)
    meta_gen=torch.Generator().manual_seed(args.seed+2)
    log=(args.run_dir/"train_log.jsonl").open("w")
    start=time.time(); ema=None; totals=[0,0,0,0]; best=(-1.,-1.,-float("inf"))
    histogram={m:0 for m in MEMORIES}
    for step in range(1,args.steps+1):
        M=MEMORIES[int(torch.randint(len(MEMORIES),(1,),generator=meta_gen))]
        P=int(torch.randint(64,(1,),generator=meta_gen))
        T=max(1,min(args.max_t,int(math.exp(float(torch.rand(1,generator=meta_gen))*math.log(args.max_t+1)))))
        histogram[M]+=1
        lr=3e-4*min(1.,step/1000)
        for group in optimizer.param_groups: group["lr"]=lr
        loss=0.
        for _ in range(args.grad_accum):
            inputs,labels=make_batch(args.task,M,T,P,args.batch_size//args.grad_accum,data_gen,device)
            with torch.autocast("cuda",dtype=torch.bfloat16): out=model(inputs,labels=labels)
            (out.loss/args.grad_accum).backward()
            loss+=float(out.loss)/args.grad_accum
            totals=[a+b for a,b in zip(totals,score(out.logits.detach(),labels,M))]
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not math.isfinite(loss) or not torch.isfinite(norm):
            raise FloatingPointError(f"nonfinite loss/gradient at step {step}")
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        ema=loss if ema is None else .99*ema+.01*loss
        if step%100==0 or step==args.steps:
            rec=dict(step=step,loss=loss,ema_loss=ema,token_acc=totals[0]/totals[2],
                     string_acc=totals[1]/totals[3],elapsed_sec=time.time()-start,lr=lr,
                     memory_histogram=histogram.copy())
            log.write(json.dumps(rec)+"\n"); log.flush()
            print(f"step {step}/{args.steps} EMA={ema:.5f} string={rec['string_acc']:.4f}",flush=True)
            totals=[0,0,0,0]
        if step%args.validation_interval==0 or step==args.steps:
            model.eval()
            cells=[evaluate_cell(model,args.task,m,t,prefix,32,54321,device)
                   for m in MEMORIES for t in (16,64,256,1024) for prefix in (0,7)]
            string=sum(x["string_acc"] for x in cells)/len(cells)
            token=sum(x["token_acc"] for x in cells)/len(cells)
            rec=dict(step=step,validation=cells,macro_string=string,macro_token=token)
            log.write(json.dumps(rec)+"\n"); log.flush()
            print(f"validation {step} string={string:.4f} token={token:.4f}",flush=True)
            if (string,token,-ema)>best:
                best=(string,token,-ema)
                model.save_pretrained(args.run_dir/"model_best")
                atomic_json(args.run_dir/"best.json",dict(step=step,macro_string=string,macro_token=token,ema_loss=ema))
            model.train()
        if step%5000==0 or step==args.steps:
            model.save_pretrained(args.run_dir/"model")
            torch.save(dict(step=step,optimizer=optimizer.state_dict(),data_rng=data_gen.get_state(),
                            metadata_rng=meta_gen.get_state(),ema=ema,best=best),args.run_dir/"optimizer.pt")
    log.close()


if __name__=="__main__": main()
