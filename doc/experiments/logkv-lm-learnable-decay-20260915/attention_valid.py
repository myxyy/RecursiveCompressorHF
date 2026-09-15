"""Post-review correction: profile only real prediction positions, never PAD.
Original frozen evaluation/results remain unchanged. One GPU, no training.
"""
import json
import sys
import time
import types
from common import HERE, ROOT, DATA, SOURCE, RUN_NAME, save, sha
sys.path.insert(0,str(SOURCE))
import torch
from logkv_lm import LogKVLM
from evaluate import inputs


@torch.no_grad()
def main():
    start=time.monotonic();torch.set_num_threads(1);torch.set_float32_matmul_precision('high')
    out=HERE/'analysis';out.mkdir(exist_ok=True)
    weights=DATA/'checkpoints_logkv'/RUN_NAME/'checkpoint-5000/model/model.safetensors'
    before=sha(weights);model=LogKVLM.from_pretrained(weights.parent).cuda().eval()
    x,y,tokenizer,manifest=inputs()
    assert manifest==json.loads((HERE/'results/heldout_manifest.json').read_text())
    originals=[];records=[];selected=None;row_index=None
    for layer,block in enumerate(model.layers):
        attention=block.attention;original=attention._attend_levels;originals.append(original)
        def observed(module,q,n,scale,*flat,layer=layer,original=original):
            result=original(q,n,scale,*flat)
            slope=module.level_decay.repeat(q.size(0)//module.num_heads)[:,None,None]
            logits=[]
            for i in range(n):
                k=flat[i][:,flat[2*n+i],:]
                v=torch.einsum('bld,blcd->blc',q,k)*scale-i*slope
                v=v.masked_fill(flat[3*n+i][None,:,:],float('-inf'))
                logits.append(torch.logsumexp(v.float(),dim=-1))
            assert len(flat)>4*n
            logits.append(((q*flat[4*n]).sum(-1)*scale).float())
            probability=torch.stack(logits,-1).softmax(-1)[:,selected,:]
            assert probability.shape[:2]==(8,len(selected))
            assert torch.isfinite(probability).all()
            assert torch.allclose(probability.sum(-1),torch.ones_like(probability[...,0]),atol=1e-6)
            records.append(dict(row=row_index,source=manifest['sources'][row_index],layer=layer+1,
                valid_queries=len(selected),levels=list(range(n))+['self'],mass=probability.mean(1).cpu().tolist()))
            return result
        attention._attend_levels=types.MethodType(observed,attention)
    exact=[];padding=[]
    for index in range(len(x)):
        row_index=index
        valid=(x[index]!=tokenizer.pad_token_id)&(y[index]!=-100)
        selected=valid.nonzero().flatten()[-512:].cuda()
        assert len(selected)>0
        padding.append(dict(row=index,source=manifest['sources'][index],
            original_window_valid_queries=int(valid[-512:].sum()),corrected_queries=len(selected),
            first_query=int(selected[0]),last_query=int(selected[-1])))
        with torch.autocast('cuda',dtype=torch.bfloat16):observed_logits=model(x[index:index+1].cuda()).logits
        if index in [0,32,64,96]:
            for block,original in zip(model.layers,originals):
                block.attention._profile_active=block.attention._attend_levels
                block.attention._attend_levels=original
            with torch.autocast('cuda',dtype=torch.bfloat16):reference=model(x[index:index+1].cuda()).logits
            exact.append(torch.equal(observed_logits,reference))
            for block in model.layers:block.attention._attend_levels=block.attention._profile_active
            del reference
        del observed_logits
    assert all(exact) and sha(weights)==before
    save(out/'attention_valid.json',dict(passed=True,weights_sha256=before,elapsed_seconds=time.monotonic()-start,
        observer_bitexact=exact,records=records,padding_audit=padding,
        selection='Last up to512 queries with both non-PAD input and nonmasked target per row;128 fixed unseen rows.',
        original_issue='Original attention_mass averaged last512 absolute positions, including PAD. Other metrics unaffected.'))
    print('Valid-position attention complete:',len(records),'records',flush=True)


if __name__=='__main__':main()
