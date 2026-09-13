"""Post-campaign CPU review: immutable artifact audit and legacy alpha=1 check."""
import importlib.util
import json
import math
import sys
from pathlib import Path
import torch
from safetensors.torch import load_file
from common import HERE, ROOT, SOURCE, MODES, run_dir, sha, save, now
sys.path.insert(0,str(SOURCE))
from configuration_logkv import LogKVConfig
from logkv_lm import LogKVLM

def read(p): return json.loads(Path(p).read_text())
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def main():
    torch.set_num_threads(1)
    dest=HERE/'results'; campaign=read(ROOT/'campaign.json'); audit=read(dest/'review.json')
    assert campaign['state']=='complete-awaiting-review' and campaign['elapsed_hours']<7.5
    assert audit['passed'] and audit['standard_cells']==246 and audit['paired_cells']==204
    assert read(HERE/'campaign.json')==campaign
    for f,h in campaign['source_manifest']['files'].items(): assert sha(SOURCE/f)==h
    for f,h in campaign['scripts'].items(): assert sha(HERE/f)==h
    for f,h in audit['result_hashes'].items(): assert sha(dest/f)==h
    for key,h in audit['weights'].items():
        mode,cp=key.split('/'); folder='model_best' if cp=='best' else 'model'
        assert sha(run_dir(mode)/folder/'model.safetensors')==h
    rows=read(dest/'metrics.json'); assert len(rows)==450
    for mode in MODES:
        for f in (ROOT/mode).iterdir():
            if f.suffix in ['.json','.log','.png']: assert sha(f)==sha(dest/mode/f.name)
        for f in ['train_log.jsonl','run_config.json','best.json']:
            assert sha(run_dir(mode)/f)==sha(dest/mode/f)
        for cp in ['best','final']:
            for kind in ['standard','paired']:
                data=read(dest/mode/(f'digits_{cp}.json' if kind=='standard' else f'paired_{cp}.json'))
                records=data if kind=='standard' else data['records']
                for r in [r for r in rows if r['mode']==mode and r['checkpoint']==cp and r['evaluation']==kind]:
                    t=torch.tensor([v for b in records[str(r['T'])] for v in b['target']])
                    p=torch.tensor([v for b in records[str(r['T'])] for v in b['prediction']])
                    eq=t==p
                    assert eq.shape==(r['n'],10)
                    assert int(eq.sum())==r['token_correct'] and int(eq.all(1).sum())==r['string_correct']
                    assert (~eq).sum(0).tolist()==r['digit_errors']
    # Full legacy LM class, not a new LM accidentally importing the new block.
    legacy=ROOT.parent/'logkv-phase2-rope-20260908/source'
    oldcore=module('legacy_core',legacy/'logkv.py')
    currentcore=sys.modules['logkv']; sys.modules['logkv']=oldcore
    try: oldlm=module('legacy_lm',legacy/'logkv_lm.py')
    finally: sys.modules['logkv']=currentcore
    oldrun=ROOT.parent/'logkv-rope-only-20260909/exp/copying/retrieval-rope-no-phase-fixed10-20260909'
    cfg=LogKVConfig.from_dict(read(oldrun/'model/config.json'))
    torch.manual_seed(0); old=oldlm.LogKVLM(cfg); rng=torch.get_rng_state()
    torch.manual_seed(0); new=LogKVLM(cfg)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())
    initial=load_file(ROOT/'initial_model/model.safetensors')
    assert all(torch.equal(v,initial[k]) for k,v in old.state_dict().items())
    ids=torch.randint(0,10,(2,85))
    # At the actual model size, compare forward and all parameter gradients.
    a=old(ids,labels=ids); b=new(ids,labels=ids)
    assert torch.equal(a.logits,b.logits) and torch.equal(a.loss,b.loss)
    a.loss.backward();b.loss.backward()
    assert all(torch.equal(x.grad,y.grad) for x,y in zip(old.parameters(),new.parameters()))
    old.zero_grad();new.zero_grad()
    checkpoint_checks={}
    for label,path in [('historical_best',oldrun/'model_best/model.safetensors'),
                       ('current_alpha_one_best',run_dir('alpha-1')/'model_best/model.safetensors')]:
        weights=load_file(path); old.load_state_dict(weights);new.load_state_dict(weights)
        with torch.no_grad():
            a=old(ids).logits;b=new(ids).logits
            assert torch.equal(a,b)
        checkpoint_checks[label]=dict(cpu_fp32_logits_bitexact=True,weight_sha256=sha(path))
    hist=[read_line for line in (oldrun/'train_log.jsonl').read_text().splitlines()
          if 'loss' in (read_line:=json.loads(line))]
    fresh=[read_line for line in (run_dir('alpha-1')/'train_log.jsonl').read_text().splitlines()
           if 'loss' in (read_line:=json.loads(line))]
    first=next(dict(old=a,new=b) for a,b in zip(hist,fresh)
        if {k:v for k,v in a.items() if k!='elapsed_sec'}!={k:v for k,v in b.items() if k!='elapsed_sec'})
    summary=[]
    for mode in MODES:
        for cp in ['best','final']:
            rr=[r for r in rows if r['mode']==mode and r['checkpoint']==cp and r['evaluation']=='standard']
            summary.append(dict(mode=mode,checkpoint=cp,perfect_horizons=sum(r['string_correct']==256 for r in rr),
                worst_horizons=[{'T':r['T'],'exact':r['string_correct']} for r in sorted(rr,key=lambda r:r['string_correct'])[:5]]))
    save(dest/'manual_review.json',dict(passed=True,reviewed=now(),gpu_used=False,
        audited_cells=len(rows),all_original_hashes_passed=True,raw_archive_matches=True,
        legacy_reconstructed_initial_weights_match_saved_initial=True,legacy_initial_rng_matches=True,
        legacy_cpu_fullsize_forward_and_gradients_bitexact=True,checkpoint_checks=checkpoint_checks,
        first_historical_training_difference=first,summary=summary,
        limitation='GPU training trajectories diverge by step100 despite matching configuration and reconstructed initialization; root cause unisolated. CPU equality is not proof of GPU bitwise determinism. One training seed per alpha.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,4))
    for mode in MODES:
        rr=[r for r in rows if r['mode']==mode and r['checkpoint']=='best' and r['evaluation']=='paired']
        for ax,lo,hi in [(axes[0],49144,49160),(axes[1],65535,65537)]:
            sub=[r for r in rr if lo<=r['T']<=hi]
            ax.plot([r['T'] for r in sub],[r['string_acc'] for r in sub],'o-',label=mode)
        train=[v for line in (dest/mode/'train_log.jsonl').read_text().splitlines() if 'loss' in (v:=json.loads(line))]
        axes[2].plot([v['step'] for v in train],[v['string_acc'] for v in train],label=mode,alpha=.7)
    for ax,title in zip(axes,['Paired memories, best: T49152 vicinity','Paired memories, best: T65536 vicinity','Training interval exact accuracy']):
        ax.set(title=title,ylim=(-.02,1.02),ylabel='Exact string accuracy');ax.grid(alpha=.3);ax.legend()
        ax.ticklabel_format(axis='x',style='plain',useOffset=False)
    axes[1].set_xticks([65535,65536,65537])
    axes[2].set_xlabel('Step');axes[0].set_xlabel('T');axes[1].set_xlabel('T')
    fig.tight_layout();fig.savefig(dest/'boundary_training.png',dpi=150);plt.close(fig)
    print(json.dumps(read(dest/'manual_review.json'),indent=2))
if __name__=='__main__':main()
