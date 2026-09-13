"""Verify no-phase configuration, shared initialization, and finite CPU output."""
import hashlib,json,sys
from pathlib import Path
import torch
SOURCE=Path('/mnt/raid0/RecursiveCompressor/experiments/logkv-phase2-rope-20260908/source')
sys.path.insert(0,str(SOURCE))
from configuration_logkv import LogKVConfig
from logkv_lm import LogKVLM
HERE=Path(__file__).resolve().parent
torch.set_num_threads(1)
rows=[];weights={}
for phase in (False,True):
 for mode in ('retrieval-rope','compressor-rope'):
  torch.manual_seed(0)
  cfg=LogKVConfig(vocab_size=10,d_model=512,num_heads=8,d_ff=1024,num_layers=2,chunk_size=4,
   phase_emb=phase,phase_levels=2,gated_attention=True,self_slot=True,
   retrieval_rope=mode=='retrieval-rope',compressor_rope=mode=='compressor-rope',
   pad_token_id=None,bos_token_id=None,eos_token_id=None)
  model=LogKVLM(cfg).eval()
  weights[phase,mode]={k:hashlib.sha256(v.detach().numpy().tobytes()).hexdigest() for k,v in model.state_dict().items()}
  if not phase:assert all(l.attention.phase_emb is None and l.attention.phase_levels==0 for l in model.layers)
  with torch.no_grad():
   ids=torch.zeros(2,40,dtype=torch.long);ids[:,:10]=torch.arange(10)%8+1;ids[:,-11:]=9
   y=model(ids).logits;assert torch.isfinite(y).all()
  rows.append(dict(phase_emb=phase,mode=mode,num_params=sum(p.numel() for p in model.parameters()),
                   parameter_hashes=weights[phase,mode],finite_cpu_forward=True))
assert weights[False,'retrieval-rope']==weights[False,'compressor-rope']
common=set(weights[False,'retrieval-rope'])&set(weights[True,'retrieval-rope'])
different=[k for k in sorted(common) if weights[False,'retrieval-rope'][k]!=weights[True,'retrieval-rope'][k]]
report=dict(runs=rows,no_phase_modes_initial_weights_equal=True,
            common_tensors_different_from_phase_enabled=different,
            caveat='Phase removal changes initialization RNG consumption; historical phase-on comparison is not common-weight matched.')
(HERE/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(no_phase_parameters=rows[0]['num_params'],phase_parameters=rows[2]['num_params'],
 same_initial_weights_between_rope_modes=True,changed_common_tensors_vs_phase=len(different))))
