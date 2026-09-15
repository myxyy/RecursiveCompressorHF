"""Read-only post-completion audit plus corrected attention/text analysis (CPU)."""
import collections
import datetime
import json
import math
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from safetensors import safe_open
from transformers import AutoTokenizer
from common import HERE, ROOT, DATA, SOURCE, RUN_NAME, save, sha


def read(path):return json.loads(path.read_text())


def main():
    result=HERE/'results';out=HERE/'analysis';out.mkdir(exist_ok=True)
    campaign=read(HERE/'campaign.json');assert campaign==read(ROOT/'campaign.json')
    assert campaign['state']=='complete-awaiting-review' and all(r['returncode']==0 for r in campaign['stages'])
    hashes={}
    def check(path,expected):
        assert sha(path)==expected,str(path)
        hashes[str(path)]=expected
    for group in ['scripts','prerequisite_hashes']:
        for name,expected in campaign[group].items():check(HERE/name,expected)
    for name,expected in campaign['source_manifest']['files'].items():check(SOURCE/name,expected)
    review=read(result/'completion_review.json');assert review['passed']
    for name,expected in review['output_hashes'].items():check(result/name,expected)
    hist=[json.loads(l) for l in (result/'beta_log.jsonl').read_text().splitlines()]
    assert [r['step'] for r in hist]==list(range(0,5001,10))
    beta=np.array([r['beta'] for r in hist]);assert beta.shape==(501,16,8) and np.isfinite(beta).all()
    for step,expected in review['checkpoint_sha256'].items():
        path=DATA/'checkpoints_logkv'/RUN_NAME/f'checkpoint-{step}/model/model.safetensors';check(path,expected)
        with safe_open(path,framework='pt',device='cpu') as f:
            values=np.array([f.get_tensor(f'layers.{i}.attention.level_decay').tolist() for i in range(16)])
        assert np.array_equal(values,beta[int(step)//10])
    coefficients=[]
    for step in range(0,5001,1000):
        b=beta[step//10]
        coefficients.append(dict(step=step,min=float(b.min()),mean=float(b.mean()),max=float(b.max()),negative=int((b<0).sum())))
    save(out/'coefficients_summary.json',dict(checkpoints=coefficients,
        final_layer_means=beta[-1].mean(1).tolist(),last1000_decreasing_heads=int((beta[-1]<beta[400]).sum()),
        last1000_mean_change=float((beta[-1]-beta[400]).mean())))
    manifest=read(result/'heldout_manifest.json')
    loss=read(result/'heldout_loss.json');loss_summary=[]
    for mode in ['learned','reset_logC','zero']:
        rows=[r for r in loss if r['mode']==mode];n=sum(r['tokens'] for r in rows)
        assert len(rows)==4 and n==212943
        for r in rows:assert math.isclose(math.exp(r['loss']),r['perplexity'],rel_tol=1e-12)
        value=sum(r['tokens']*r['loss'] for r in rows)/n
        loss_summary.append(dict(mode=mode,loss=value,perplexity=math.exp(value),tokens=n))
    save(out/'loss_summary.json',loss_summary)
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'tokenizer',local_files_only=True)
    generations=read(result/'generations.json');assert len(generations)==21
    text_rows=[]
    for index,r in enumerate(generations):
        ids=r['token_ids'];text=tokenizer.decode(ids,skip_special_tokens=True)
        assert text==r['text'] and len(ids)==r['generated_tokens'] and len(ids)<=r['max_new_tokens']
        assert r['eos_terminated']==bool(ids[-1]==tokenizer.eos_token_id)
        assert tokenizer.eos_token_id not in ids[:-1]
        if len(ids)<r['max_new_tokens']:assert r['eos_terminated']
        quarter=text[3*len(text)//4:];pairs=list(zip(quarter,quarter[1:]))
        distinct=len(set(pairs))/len(pairs) if pairs else None
        assert distinct==r['q4_char_bigram_distinct']
        grams=list(zip(ids,ids[1:],ids[2:],ids[3:]))
        assert r['token_4gram_repeat_fraction']==(1-len(set(grams))/len(grams) if grams else None)
        longest=run=0;previous=None
        for token in ids:
            run=run+1 if token==previous else 1;longest=max(longest,run);previous=token
        assert longest==r['longest_same_token_run']
        quartiles=[]
        for i in range(4):
            part=text[len(text)*i//4:len(text)*(i+1)//4];bigrams=list(zip(part,part[1:]))
            quartiles.append(len(set(bigrams))/len(bigrams) if bigrams else None)
        text_rows.append(dict(case=index,prompt=r['prompt'],seed=r['seed'],temperature=r['temperature'],
            limit=r['max_new_tokens'],length=len(ids),eos=r['eos_terminated'],quartile_distinct=quartiles,
            token_4gram_repeat_fraction=r['token_4gram_repeat_fraction'],max_same_token=longest))
    for summary in read(result/'generation_summary.json'):
        rows=[r for r in generations if (r['max_new_tokens'],r['temperature'])==(summary['limit'],summary['temperature'])]
        assert len(rows)==summary['n'] and sum(r['eos_terminated'] for r in rows)==summary['eos']
        assert np.mean([r['generated_tokens'] for r in rows])==summary['mean_length']
        assert np.mean([r['q4_char_bigram_distinct'] for r in rows])==summary['mean_q4_distinct']
        assert sum(r['q4_char_bigram_distinct']<.5 for r in rows)==summary['q4_below_half']
    prefixes=[]
    for extended in generations[18:]:
        short=next(r for r in generations[:18] if (r['prompt'],r['seed'],r['temperature'])==(extended['prompt'],extended['seed'],extended['temperature']))
        assert extended['token_ids'][:len(short['token_ids'])]==short['token_ids']
        prefixes.append(dict(prompt=extended['prompt'],short_length=len(short['token_ids']),extended_length=len(extended['token_ids']),prefix_exact=True))
    save(out/'generation_analysis.json',dict(rows=text_rows,extension_prefixes=prefixes,
        note='4096-limit rows extend/repeat the same seed0 trajectories; not three independent extra samples. EOS does not imply coherent output.'))
    valid=read(out/'attention_valid.json');assert valid['passed'] and all(valid['observer_bitexact'])
    assert valid['weights_sha256']==review['checkpoint_sha256']['5000']
    records=valid['records'];assert len(records)==128*16
    sums=np.zeros((16,8,7));counts=np.zeros(16)
    for r in records:
        assert r['levels']==[0,1,2,3,4,5,'self']
        mass=np.array(r['mass']);assert mass.shape==(8,7) and np.isfinite(mass).all()
        assert (mass>=0).all() and np.allclose(mass.sum(1),1,atol=1e-6)
        layer=r['layer']-1;sums[layer]+=mass*r['valid_queries'];counts[layer]+=r['valid_queries']
    average=sums/counts[:,None,None]
    layer_rows=[dict(layer=i+1,self_mass=float(a[:,-1].mean()),level0=float(a[:,0].mean()),
        level1=float(a[:,1].mean()),level2plus=float(a[:,2:6].sum(1).mean())) for i,a in enumerate(average)]
    high=average[:,:,2:6].sum(2)
    top=[]
    for flat in np.argsort(high.ravel())[-8:][::-1]:
        l,h=np.unravel_index(flat,high.shape);top.append(dict(layer=int(l+1),head=int(h+1),mass=float(high[l,h]),beta=float(beta[-1,l,h])))
    save(out/'attention_summary.json',dict(averaging='real-query weighted across128 rows, then equal head/layer means',
        total_queries_per_head=int(counts[0]),overall_mass=average.mean((0,1)).tolist(),layer_rows=layer_rows,
        top_level2plus=top,per_head_mass=average.tolist(),original_four_padding=[valid['padding_audit'][i] for i in [0,32,64,96]]))
    fig,axes=plt.subplots(1,2,figsize=(13,6));bottom=np.zeros(16)
    for i,label in enumerate(['L0','L1','L2','L3','L4','L5','self']):
        values=average[:,:,i].mean(1)*100;axes[0].bar(np.arange(1,17),values,bottom=bottom,label=label);bottom+=values
    axes[0].set(xlabel='Transformer layer',ylabel='Attention mass (%)',title='Real query positions only;128 unseen rows',xticks=range(1,17));axes[0].legend(ncol=4,fontsize=8)
    image=axes[1].imshow(high*100,vmin=0,vmax=100,cmap='viridis',aspect='auto')
    axes[1].set(xlabel='Head',ylabel='Transformer layer',title='Mass on compression levels >=2 (%)',xticks=range(8),xticklabels=range(1,9),yticks=range(16),yticklabels=range(1,17))
    fig.colorbar(image,ax=axes[1]);fig.tight_layout();fig.savefig(out/'attention.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for mode,color in [('learned','tab:blue'),('reset_logC','tab:orange'),('zero','tab:red')]:
        rows=[r for r in loss if r['mode']==mode];offset={'learned':-.24,'reset_logC':0,'zero':.24}[mode]
        ax.bar(np.arange(4)+offset,[r['loss'] for r in rows],width=.24,label=mode,color=color)
    ax.set(xticks=range(4),xticklabels=[r['source'] for r in rows],ylabel='Token-weighted cross entropy',ylim=(3,4.6),title='Inference-only beta interventions on the same checkpoint');ax.legend();fig.tight_layout();fig.savefig(out/'loss.png',dpi=160);plt.close(fig)
    save(out/'review.json',dict(passed=True,checkpoint_count=5,coefficient_snapshots=501,generation_cases=21,
        corrected_attention_records=len(records),hashes=hashes,
        original_attention_padding_issue=True,original_results_preserved=True,
        correction_seconds=valid['elapsed_seconds'],
        analysis_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='review.json'},
        analysis_scripts={name:sha(HERE/name) for name in ['attention_valid.py','analyze_completed.py']},
        checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print(json.dumps(dict(loss=loss_summary,attention=layer_rows,top_heads=top),indent=2))


if __name__=='__main__':main()
