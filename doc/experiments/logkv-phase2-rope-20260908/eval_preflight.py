"""Check evaluation VRAM at the standard 2**19-token budget, one GPU."""
import json,sys,time
from pathlib import Path
import torch
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE));sys.path.insert(0,str(SOURCE/'exp/copying'))
from configuration_logkv import LogKVConfig
from logkv_lm import LogKVLM
from evaluate import eval_horizon
ROOT=Path('/mnt/raid0/RecursiveCompressor/experiments/logkv-phase2-rope-20260908')
torch.set_num_threads(1);rows=[]
for mode in ('phase2','retrieval-rope','compressor-rope'):
 cfg=LogKVConfig(vocab_size=10,d_model=512,num_heads=8,d_ff=1024,num_layers=2,chunk_size=4,
  phase_emb=True,phase_levels=2,self_slot=True,gated_attention=True,
  retrieval_rope=mode=='retrieval-rope',compressor_rope=mode=='compressor-rope')
 model=LogKVLM(cfg).to(device='cuda',dtype=torch.bfloat16).eval()
 for T in (2048,8192):
  torch.cuda.reset_peak_memory_stats();start=time.monotonic()
  a=eval_horizon(model,T,256,torch.Generator().manual_seed(12345),torch.device('cuda'),True)
  torch.cuda.synchronize()
  r=dict(mode=mode,T=T,samples=256,elapsed_sec=time.monotonic()-start,
   peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,random_model_accuracies=a)
  rows.append(r);print(json.dumps(r),flush=True)
 del model;torch.cuda.empty_cache()
(ROOT/'eval_preflight.json').write_text(json.dumps(rows,indent=2)+'\n')
