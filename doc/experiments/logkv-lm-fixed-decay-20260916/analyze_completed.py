"""CPU post-completion audit; preserve original campaign and result artifacts."""
import datetime
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from safetensors import safe_open
from common import HERE, ROOT, DATA, SOURCE, RUN_NAME, BASE, BASE_ROOT, save, sha


def read(p):return json.loads(p.read_text())


def main():
    out=HERE/'analysis';out.mkdir(exist_ok=True)
    campaign=read(HERE/'campaign.json');assert campaign==read(ROOT/'campaign.json')
    assert campaign['state']=='complete-awaiting-review' and not campaign['next_campaign_queued']
    assert len(campaign['stages'])==4 and all(s['returncode']==0 for s in campaign['stages'])
    hashes={}
    def check(p,expected):
        assert sha(p)==expected,str(p)
        hashes[str(p)]=expected
    for group in ['scripts','prerequisite_hashes']:
        for name,value in campaign[group].items():check(HERE/name,value)
    for name,value in campaign['source_manifest']['files'].items():check(SOURCE/name,value)
    results=HERE/'results';base=BASE/'results';comparison=read(results/'comparison.json')
    assert read(results/'heldout_manifest.json')==read(base/'heldout_manifest.json')
    config=read(results/'run_config.json');oldconfig=read(base/'run_config.json')
    assert sorted(k for k,v in config['arguments'].items() if v!=oldconfig['arguments'][k])==['learnable_decay','run_name']
    for folder,root,name in [(results,ROOT,RUN_NAME),(base,BASE_ROOT,'d1024-h8-l16-conv4-learnable-decay-5000')]:
        review=read(folder/'completion_review.json');assert review['passed']
        for relative,expected in review['output_hashes'].items():check(folder/relative,expected)
        for step,expected in review['checkpoint_sha256'].items():
            path=root/'data/checkpoints_logkv'/name/f'checkpoint-{step}/model/model.safetensors';check(path,expected)
            if folder==results:
                with safe_open(path,framework='pt',device='cpu') as f:assert not any(k.endswith('.level_decay') for k in f.keys())
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'tokenizer',local_files_only=True)
    generation_rows=[];paired=[]
    for label,folder in [('fixed',results),('learned',base)]:
        generations=read(folder/'generations.json');assert len(generations)==21
        for index,r in enumerate(generations):
            ids=r['token_ids'];text=tokenizer.decode(ids,skip_special_tokens=True)
            assert text==r['text'] and len(ids)==r['generated_tokens']<=r['max_new_tokens']
            assert r['eos_terminated']==(ids[-1]==tokenizer.eos_token_id)
            assert tokenizer.eos_token_id not in ids[:-1]
            if len(ids)<r['max_new_tokens']:assert r['eos_terminated']
            quarter=text[3*len(text)//4:];pairs=list(zip(quarter,quarter[1:]))
            distinct=len(set(pairs))/len(pairs) if pairs else None
            assert distinct==r['q4_char_bigram_distinct']
            grams=list(zip(ids,ids[1:],ids[2:],ids[3:]))
            assert r['token_4gram_repeat_fraction']==(1-len(set(grams))/len(grams) if grams else None)
            longest=run=0;previous=None
            for token in ids:run=run+1 if token==previous else 1;longest=max(longest,run);previous=token
            assert longest==r['longest_same_token_run']
            generation_rows.append(dict(model=label,case=index,prompt=r['prompt'],seed=r['seed'],temperature=r['temperature'],
                limit=r['max_new_tokens'],length=len(ids),eos=r['eos_terminated'],q4_distinct=distinct))
        for r in read(folder/'generation_summary.json'):
            rows=[a for a in generations if (a['max_new_tokens'],a['temperature'])==(r['limit'],r['temperature'])]
            assert len(rows)==r['n'] and sum(a['eos_terminated'] for a in rows)==r['eos']
            assert np.mean([a['generated_tokens'] for a in rows])==r['mean_length']
            assert np.mean([a['q4_char_bigram_distinct'] for a in rows])==r['mean_q4_distinct']
            assert sum(a['q4_char_bigram_distinct']<.5 for a in rows)==r['q4_below_half']
        for a in generations[18:]:
            b=next(b for b in generations[:18] if (a['prompt'],a['seed'],a['temperature'])==(b['prompt'],b['seed'],b['temperature']))
            assert a['token_ids'][:len(b['token_ids'])]==b['token_ids']
        rows=read(folder/'heldout_loss.json');rows=[r for r in rows if r['mode'] in ['fixed','learned']]
        n=sum(r['tokens'] for r in rows);assert n==212943
        loss=sum(r['tokens']*r['loss'] for r in rows)/n
        expected=next(r for r in comparison['loss'] if r['model']==label)
        assert loss==expected['loss'] and math.exp(loss)==expected['perplexity']
    for a,b in zip(generation_rows[:21],generation_rows[21:]):
        for key in ['case','prompt','seed','temperature','limit']:assert a[key]==b[key]
        paired.append(dict(case=a['case'],prompt=a['prompt'],seed=a['seed'],temperature=a['temperature'],limit=a['limit'],
            fixed_length=a['length'],learned_length=b['length'],fixed_q4=a['q4_distinct'],learned_q4=b['q4_distinct'],
            fixed_eos=a['eos'],learned_eos=b['eos']))
    save(out/'paired_generations.json',paired)
    attention_fixed=read(results/'attention_valid.json');attention_learned=read(BASE/'analysis/attention_valid.json')
    check(BASE/'analysis/attention_valid.json',read(BASE/'analysis/review.json')['analysis_hashes']['attention_valid.json'])
    assert attention_fixed['padding_audit']==attention_learned['padding_audit']
    attention={}
    for label,data in [('fixed',attention_fixed),('learned',attention_learned)]:
        assert data['passed'] and all(data['observer_bitexact']) and len(data['records'])==2048
        sums=np.zeros((16,8,7));counts=np.zeros(16)
        for r in data['records']:
            assert r['levels']==[0,1,2,3,4,5,'self'];a=np.array(r['mass'])
            assert a.shape==(8,7) and np.isfinite(a).all() and (a>=0).all()
            assert np.allclose(a.sum(-1),1,atol=1e-6)
            i=r['layer']-1;sums[i]+=a*r['valid_queries'];counts[i]+=r['valid_queries']
        assert np.all(counts==64338)
        attention[label]=sums/counts[:,None,None]
        assert np.array_equal(attention[label],np.array(comparison['attention'][label]))
    fixed,learned=attention['fixed'],attention['learned'];f=fixed[:,:,2:6].sum(-1);l=learned[:,:,2:6].sum(-1)
    layers=[dict(layer=i+1,fixed_self=float(fixed[i,:,6].mean()),learned_self=float(learned[i,:,6].mean()),
        fixed_level2plus=float(f[i].mean()),learned_level2plus=float(l[i].mean())) for i in range(16)]
    stats=dict(fixed_overall=fixed.mean((0,1)).tolist(),learned_overall=learned.mean((0,1)).tolist(),
        increased_level2plus_heads=int((l>f).sum()),same_head_mass_pearson=float(np.corrcoef(f.ravel(),l.ravel())[0,1]),layer_rows=layers)
    save(out/'attention_summary.json',stats)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    axes[0].scatter(f*100,l*100,s=18,alpha=.7);axes[0].plot([0,100],[0,100],'--',color='gray')
    axes[0].set(xlabel='Fixed: mass on levels >=2 (%)',ylabel='Learned: mass on levels >=2 (%)',title='128 corresponding layer/head slots',xlim=(0,85),ylim=(0,85))
    delta=(l-f)*100;lim=max(abs(delta.min()),abs(delta.max()))
    image=axes[1].imshow(delta,vmin=-lim,vmax=lim,cmap='coolwarm',aspect='auto')
    axes[1].set(xlabel='Head',ylabel='Transformer layer',title='Learned minus fixed (percentage points)',xticks=range(8),xticklabels=range(1,9),yticks=range(16),yticklabels=range(1,17));fig.colorbar(image,ax=axes[1])
    fig.tight_layout();fig.savefig(out/'attention_difference.png',dpi=160);plt.close(fig)
    loss=comparison['loss'];fixedloss=next(r for r in loss if r['model']=='fixed');learnedloss=next(r for r in loss if r['model']=='learned')
    save(out/'summary.json',dict(loss=loss,learned_minus_fixed_loss=learnedloss['loss']-fixedloss['loss'],
        learned_relative_ppl_reduction=1-learnedloss['perplexity']/fixedloss['perplexity'],
        generation=comparison['generation'],attention=stats))
    save(out/'review.json',dict(passed=True,checkpoint_count=10,generations_recounted=42,attention_records_recounted=4096,
        hashes=hashes,checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu_work_started=False,
        analysis_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='review.json'},
        script_sha256=sha(HERE/'analyze_completed.py')))
    print('Post-completion audit passed:10 checkpoints,42 generations,4096 attention records.')


if __name__=='__main__':main()
