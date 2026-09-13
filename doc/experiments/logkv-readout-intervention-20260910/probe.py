"""Observations and single-answer interventions, leaving cached prefixes intact.

Head numbers here are zero-based. Patches are applied at a single answer
position; earlier outputs and the input prefix must remain unchanged.
"""
import math,sys,types
import torch
from common import SOURCE
sys.path.insert(0,str(SOURCE))
from logkv import _local_rope
from logkv_lm import LogKVLM


class Probe:
 def __init__(self,model):
  self.model=model;self.active=False;self.patch='none';self.donor=None
  self.target=0;self.level_index=0;self.saved={};self.restore=[];self.handles=[]
  for li,block in enumerate(model.layers):
   att=block.attention;original=att._attend_levels
   self.restore.append((att,'_attend_levels',original))
   def attend(mod,q,n,scale,*flat,_orig=original,_li=li):
    if self.active:self.level_index=0
    out=_orig(q,n,scale,*flat)
    if self.active:
     H=mod.num_heads;B=q.shape[0]//H;t=self.target
     x=out.reshape(B,H,q.shape[1],q.shape[2])
     self.saved[f'l{_li+1}_query']=q.reshape(B,H,q.shape[1],q.shape[2])[:,:,t].detach().clone()
     if (_li==0 and self.patch in ('head8','head1','sham_head8')) or (_li==1 and self.patch in ('l2_raw','l2_raw_gate')):
      x=x.clone()
      if _li==0:
       h=0 if self.patch=='head1' else 7
       x[:,h,t]=self.donor['l1_raw'][:,h]
      else:x[:,:,t]=self.donor['l2_raw']
      out=x.reshape_as(out)
     self.saved[f'l{_li+1}_raw']=x[:,:,t].detach().clone()
    return out
   att._attend_levels=types.MethodType(attend,att)
   original_level=att._level_attention
   self.restore.append((att,'_level_attention',original_level))
   def level(mod,q,k,v,local,invalid,bias,scale,_orig=original_level,_li=li):
    idx=self.level_index
    if self.active:self.level_index+=1
    if not(self.active and _li==0 and idx==7 and self.patch in ('phase','mask','phase_mask','sham_phase')):
     return _orig(q,k,v,local,invalid,bias,scale)
    assert mod.chunk_size==4 and mod.relative_key is None
    assert not mod.relative_position_bias and mod.level_decay is None
    slots=k[:,local,:];values=v[:,local,:]
    j=(~invalid).sum(-1);pos=torch.arange(invalid.shape[1],device=q.device)[None]-j[:,None]
    score_keys=_local_rope(slots,pos)
    rows=torch.arange(7,q.shape[0],mod.num_heads,device=q.device);t=self.target
    assert int(j[t])==3
    if self.patch in ('phase','phase_mask','sham_phase'):
     angle=-3 if self.patch=='sham_phase' else -2
     score_keys=score_keys.clone()
     score_keys[rows,t,0]=_local_rope(slots[rows,t,0],q.new_tensor(angle))
    logits=torch.einsum('bld,blcd->blc',q,score_keys)*scale+bias
    logits=logits.masked_fill(invalid[None],float('-inf'))
    if self.patch in ('mask','phase_mask'):
     logits=logits.clone();logits[rows,t,2]=float('-inf')
    dtype=torch.promote_types(logits.dtype,torch.float32)
    m=logits.max(-1).values.to(dtype);m_safe=torch.where(m==float('-inf'),torch.zeros_like(m),m)
    p=torch.exp(logits.to(dtype)-m_safe[...,None]);den=p.sum(-1)
    acc=torch.einsum('blc,blcd->bld',p.to(values.dtype),values).to(dtype)
    return m,den,acc
   att._level_attention=types.MethodType(level,att)
   def gate_hook(module,args,out,_li=li):
    if not self.active:return out
    if _li==1 and self.patch in ('l2_gate','l2_raw_gate','sham_gate'):
     out=out.clone();out[:,self.target]=self.donor['l2_gate_logit']
    self.saved[f'l{_li+1}_gate_logit']=out[:,self.target].detach().clone()
    return out
   self.handles.append(att.lg.register_forward_hook(gate_hook))
   def lo_pre(module,args,_li=li):
    if self.active:self.saved[f'l{_li+1}_gated_heads']=args[0][:,self.target].detach().clone()
   self.handles.append(att.lo.register_forward_pre_hook(lo_pre))
   def lo_hook(module,args,out,_li=li):
    if self.active:self.saved[f'l{_li+1}_attention_branch']=out[:,self.target].detach().clone()
   self.handles.append(att.lo.register_forward_hook(lo_hook))
   def ffn_pre(module,args,_li=li):
    if self.active:self.saved[f'l{_li+1}_ffn_input']=args[0][:,self.target].detach().clone()
   self.handles.append(block.ffn.register_forward_pre_hook(ffn_pre))
   def ffn_hook(module,args,out,_li=li):
    if not self.active:return out
    if _li==1 and self.patch in ('l2_ffn','sham_ffn'):
     out=out.clone();out[:,self.target]=self.donor['l2_ffn_branch']
    self.saved[f'l{_li+1}_ffn_branch']=out[:,self.target].detach().clone()
    return out
   self.handles.append(block.ffn.register_forward_hook(ffn_hook))
   original_step=block.step;self.restore.append((block,'step',original_step))
   def step(mod,x,hidden=None,_orig=original_step,_li=li):
    if self.active:self.saved[f'l{_li+1}_block_input']=x[:,self.target].detach().clone()
    y,h=_orig(x,hidden)
    if self.active:
     if _li==0 and self.patch in ('l1_block','sham_block'):
      y=y.clone();y[:,self.target]=self.donor['l1_block_output']
     self.saved[f'l{_li+1}_block_output']=y[:,self.target].detach().clone()
    return y,h
   block.step=types.MethodType(step,block)
 def close(self):
  for handle in self.handles:handle.remove()
  for obj,name,orig in self.restore:setattr(obj,name,orig)
 def set(self,target,patch='none',donor=None):
  self.target=target;self.patch=patch;self.donor=donor;self.saved={};self.active=True


def clone_saved(saved):return {k:v.detach().clone() for k,v in saved.items()}


@torch.no_grad()
def prefix_states(model,memory,precision):
 """Aligned states shared by all chosen horizons; prefix contains no marker."""
 device=next(model.parameters()).device;hidden=None;states={0:None}
 with torch.autocast(device.type,dtype=torch.bfloat16,enabled=precision=='bf16' and device.type=='cuda'):
  for pos in range(0,49152,8192):
   ids=torch.zeros(len(memory),8192,dtype=torch.long,device=device)
   if pos==0:ids[:,:10]=memory.to(device)
   _,hidden=model.step(ids,hidden)
   if pos+8192 in (32768,49152):states[pos+8192]=hidden
 return states


@torch.no_grad()
def tail(model,probe,memory,T,states,precision,digit=8,patch='none',donor=None,observe=True):
 device=next(model.parameters()).device;L=T+20;base=((T+10)//8192)*8192
 assert base in states and T+9>=base # no marker belongs to reused blank prefix
 ids=torch.zeros(len(memory),L-base,dtype=torch.long,device=device)
 if base==0:ids[:,:10]=memory.to(device)
 ids[:,T+9-base:]=9
 if observe:probe.set(T+10-base+digit,patch,donor)
 else:probe.active=False
 with torch.autocast(device.type,dtype=torch.bfloat16,enabled=precision=='bf16' and device.type=='cuda'):
  logits,_=model.step(ids,states[base])
 saved=clone_saved(probe.saved) if observe else {}
 probe.active=False
 return logits[:,-10:].float(),saved
