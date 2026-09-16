"""Audit the fixed training control and compare with the completed learned run."""
import json
import math
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from safetensors import safe_open
from transformers import AutoTokenizer
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from common import HERE, ROOT, DATA, RUN_NAME, BASE, save, sha
from evaluate import repetition


def read(p):return json.loads(p.read_text())


def main():
    out=HERE/'results';out.mkdir(exist_ok=True)
    run=DATA/'checkpoints_logkv'/RUN_NAME
    old=BASE/'results';prior=read(old/'completion_review.json');assert prior['passed']
    for name,expected in prior['output_hashes'].items():assert sha(old/name)==expected
    old_analysis=read(BASE/'analysis/review.json');assert old_analysis['passed']
    for name,expected in old_analysis['analysis_hashes'].items():assert sha(BASE/'analysis'/name)==expected
    init=read(run/'initialization_audit.json');assert init['passed']
    assert init['shared_initial_sha256']==read(HERE/'initialization_audit.json')['shared_initial_sha256']
    current=read(run/'run_config.json');previous=read(old/'run_config.json')
    differences=[k for k,v in current['arguments'].items() if v!=previous['arguments'][k]]
    assert sorted(differences)==['learnable_decay','run_name']
    assert current['world_size']==previous['world_size']==6
    assert current['num_params']==previous['num_params']-128
    history=[json.loads(l) for l in (run/'progress.jsonl').read_text().splitlines()]
    assert [r['step'] for r in history]==list(range(0,5001,10))
    weights={}
    for step in range(1000,5001,1000):
        folder=run/f'checkpoint-{step}/model';config=read(folder/'config.json')
        assert not config['learnable_decay']
        with safe_open(folder/'model.safetensors',framework='pt',device='cpu') as f:
            assert not any(k.endswith('.level_decay') for k in f.keys())
        weights[str(step)]=sha(folder/'model.safetensors');shutil.copy2(folder/'config.json',out/f'config-{step}.json')
    review=read(ROOT/'evaluation/review.json');assert review['passed'] and review['generation_count']==21
    assert review['weights_sha256']==weights['5000']
    for name,expected in review['hashes'].items():assert sha(ROOT/'evaluation'/name)==expected
    assert read(ROOT/'evaluation/heldout_manifest.json')==read(old/'heldout_manifest.json')
    for p in (ROOT/'evaluation').glob('*.json'):shutil.copy2(p,out/p.name)
    for name in ['run_config.json','initialization_audit.json','progress.jsonl','samples.log']:shutil.copy2(run/name,out/name)
    shutil.copy2(ROOT/'train.log',out/'train.log')
    attention=read(ROOT/'attention/attention_valid.json');assert attention['passed'] and all(attention['observer_bitexact'])
    assert attention['weights_sha256']==weights['5000']
    base_attention=read(BASE/'analysis/attention_valid.json')
    assert attention['padding_audit']==base_attention['padding_audit']
    assert len(attention['records'])==len(base_attention['records'])==2048
    shutil.copy2(ROOT/'attention/attention_valid.json',out/'attention_valid.json')
    masses=[]
    for source in [attention,base_attention]:
        sums=np.zeros((16,8,7));counts=np.zeros(16)
        for r in source['records']:
            assert r['levels']==[0,1,2,3,4,5,'self'];a=np.array(r['mass']);assert a.shape==(8,7)
            assert np.isfinite(a).all() and np.allclose(a.sum(-1),1,atol=1e-6)
            i=r['layer']-1;sums[i]+=a*r['valid_queries'];counts[i]+=r['valid_queries']
        masses.append(sums/counts[:,None,None])
    accumulator=EventAccumulator(str(DATA/'tensorboard/logkv-pretrain'/RUN_NAME),size_guidance={'scalars':0}).Reload()
    metrics={tag:[dict(step=r.step,value=r.value) for r in accumulator.Scalars(tag)] for tag in accumulator.Tags()['scalars']}
    for tag in ['train/loss','train/grad_norm','train/lr']:
        assert [r['step'] for r in metrics[tag]]==list(range(1,5001))
        assert all(math.isfinite(r['value']) for r in metrics[tag])
    save(out/'training_metrics.json',metrics)
    old_metrics=read(old/'training_metrics.json')
    assert metrics['train/lr']==old_metrics['train/lr']
    losses=[]
    for label,path,mode in [('fixed',out,'fixed'),('learned',old,'learned')]:
        rows=[r for r in read(path/'heldout_loss.json') if r['mode']==mode]
        n=sum(r['tokens'] for r in rows);assert n==212943
        loss=sum(r['tokens']*r['loss'] for r in rows)/n
        losses.append(dict(model=label,loss=loss,perplexity=math.exp(loss),sources=rows))
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'tokenizer',local_files_only=True)
    samples=read(out/'generations.json');old_samples=read(old/'generations.json');assert len(samples)==len(old_samples)==21
    pairs=[]
    for i,(a,b) in enumerate(zip(samples,old_samples)):
        for key in ['prompt','seed','temperature','top_p','max_new_tokens']:assert a[key]==b[key]
        assert tokenizer.decode(a['token_ids'],skip_special_tokens=True)==a['text']
        assert len(a['token_ids'])==a['generated_tokens']<=a['max_new_tokens']
        assert a['eos_terminated']==(a['token_ids'][-1]==tokenizer.eos_token_id)
        assert tokenizer.eos_token_id not in a['token_ids'][:-1]
        for k,v in repetition(a['token_ids'],a['text']).items():assert a[k]==v
        pairs.append(dict(case=i,prompt=a['prompt'],seed=a['seed'],temperature=a['temperature'],limit=a['max_new_tokens'],
            fixed={k:a[k] for k in ['generated_tokens','eos_terminated','q4_char_bigram_distinct','token_4gram_repeat_fraction']},
            learned={k:b[k] for k in ['generated_tokens','eos_terminated','q4_char_bigram_distinct','token_4gram_repeat_fraction']}))
    for a in samples[18:]:
        b=next(b for b in samples[:18] if (a['prompt'],a['seed'],a['temperature'])==(b['prompt'],b['seed'],b['temperature']))
        assert a['token_ids'][:len(b['token_ids'])]==b['token_ids']
    fixed_gen=read(out/'generation_summary.json');old_gen=read(old/'generation_summary.json')
    for r in fixed_gen:
        rows=[a for a in samples if (a['max_new_tokens'],a['temperature'])==(r['limit'],r['temperature'])]
        assert len(rows)==r['n'] and sum(a['eos_terminated'] for a in rows)==r['eos']
        assert float(np.mean([a['q4_char_bigram_distinct'] for a in rows]))==r['mean_q4_distinct']
    save(out/'comparison.json',dict(loss=losses,generation=dict(fixed=fixed_gen,learned=old_gen),cases=pairs,
        attention=dict(fixed=masses[0].tolist(),learned=masses[1].tolist()),
        config_differences=differences,
        caveat='One seed; option-level comparison includes scalar-vs-tensor autocast bias precision. Same prompts/seeds do not imply identical generated trajectories. Short EOS must not be read as successful long generation.'))
    fig,axes=plt.subplots(1,2,figsize=(12,4));ema_final={}
    for label,m in [('fixed',metrics),('learned',old_metrics)]:
        ema=m['train/loss'][0]['value'];values=[]
        for r in m['train/loss']:ema=.99*ema+.01*r['value'];values.append(ema)
        ema_final[label]=ema;axes[0].plot(range(1,5001),values,label=label)
    axes[0].set(xlabel='Step',ylabel='Rank0 EMA loss');axes[0].legend()
    for index,(label,m) in enumerate(zip(['fixed','learned'],masses)):
        axes[1].plot(range(1,17),m[:,:,2:6].sum(-1).mean(-1)*100,label=label)
    axes[1].set(xlabel='Transformer layer',ylabel='Attention mass on levels >=2 (%)');axes[1].legend()
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    save(out/'completion_review.json',dict(passed=True,steps=5000,checkpoint_sha256=weights,ema_final=ema_final,
        shared_initial_sha256=init['shared_initial_sha256'],paired_evaluation_inputs=True,
        original_baseline_hashes_verified=True,real_query_attention_only=True,
        output_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='completion_review.json'}))
    print('Fixed-vs-learned LM comparison audits passed.',flush=True)


if __name__=='__main__':main()
