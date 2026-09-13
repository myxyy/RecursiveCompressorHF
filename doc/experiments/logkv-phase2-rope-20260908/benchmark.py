"""One-GPU preflight: identical random horizons and maximal-length backward."""
import hashlib,json,math,os,sys,time
from pathlib import Path
import torch
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE));sys.path.insert(0,str(SOURCE/'exp/copying'))
from logkv_lm import LogKVLM
from configuration_logkv import LogKVConfig
from task import make_batch,score_logits
ROOT=Path('/mnt/raid0/RecursiveCompressor/experiments/logkv-phase2-rope-20260908')
torch.set_num_threads(1);torch.set_float32_matmul_precision('high')
rows=[]
for mode in ('phase2','retrieval-rope','compressor-rope'):
 torch.manual_seed(0)
 cfg=LogKVConfig(vocab_size=10,d_model=512,num_heads=8,d_ff=1024,num_layers=2,chunk_size=4,
  phase_emb=True,phase_levels=2,self_slot=True,gated_attention=True,
  retrieval_rope=mode=='retrieval-rope',compressor_rope=mode=='compressor-rope',
  pad_token_id=None,bos_token_id=None,eos_token_id=None)
 model=LogKVLM(cfg).cuda().train()
 digest=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
 optim=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=0)
 def step(T,g):
  ids,labels=make_batch(T,64,generator=g,device='cuda')
  with torch.autocast('cuda',dtype=torch.bfloat16):out=model(ids,labels=labels)
  out.loss.backward();loss=float(out.loss)
  score_logits(out.logits.float(),labels)
  torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
  optim.step();optim.zero_grad()
  assert math.isfinite(loss)
 torch.cuda.reset_peak_memory_stats()
 g=torch.Generator().manual_seed(999)
 for _ in range(2):step(2028,g)
 peak=torch.cuda.max_memory_allocated()/2**30
 g=torch.Generator().manual_seed(1)
 torch.cuda.synchronize();start=time.monotonic();ts=[]
 for _ in range(200):
  T=max(1,min(2028,int(math.exp(torch.rand(1,generator=g).item()*math.log(2029)))))
  ts.append(T);step(T,g)
 torch.cuda.synchronize();elapsed=time.monotonic()-start
 row=dict(mode=mode,steps=200,elapsed_sec=elapsed,sec_per_step=elapsed/200,
  train_50k_hours=elapsed/200*50000/3600,peak_allocated_GiB=peak,
  initial_weights_sha256=digest,horizons=ts)
 rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='horizons'}),flush=True)
 (ROOT/'benchmark.json').write_text(json.dumps(rows,indent=2)+'\n')
 del model,optim;torch.cuda.empty_cache()
assert len({r['initial_weights_sha256'] for r in rows})==1
assert all(r['horizons']==rows[0]['horizons'] for r in rows)
