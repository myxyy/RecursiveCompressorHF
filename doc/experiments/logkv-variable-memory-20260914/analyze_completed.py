"""Post-completion CPU verification and figures; original audit files stay frozen."""
import datetime
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import HERE, ROOT, REPO, TASKS, audit_source, sha, save


def main():
    audit_source()
    campaign=json.loads((ROOT/'campaign.json').read_text())
    assert campaign['state']=='complete-awaiting-review' and not campaign['next_stage_queued']
    assert campaign['whole_batch_hours'] < 7.5
    for name,digest in campaign['scripts'].items(): assert sha(HERE/name)==digest
    assert sha(ROOT/'preflight.json')==campaign['preflight_sha256']
    assert sha(HERE/'evaluation_smoke.json')==campaign['evaluation_smoke_sha256']
    result=HERE/'results'; review=json.loads((result/'review.json').read_text())
    assert review['passed'] and review['cells_recounted']==880
    for name,digest in review['files'].items(): assert sha(result/name)==digest
    for name,digest in review['weights'].items():
        task,cp=name.split('/')
        folder='model_best' if cp=='best' else 'model'
        assert sha(ROOT/'runs'/task/folder/'model.safetensors')==digest
    rows=json.loads((result/'metrics.json').read_text())
    assert len(rows)==880
    sample_hashes={}; same_copying=True
    for r in rows:
        path=Path(r['observations']); digest=sha(path)
        assert digest==r['observations_sha256']==review['observations'][str(path)]
        with np.load(path,allow_pickle=False) as obs:
            good=obs['target']==obs['prediction']
            assert obs['target'].shape==(256,r['memory_len'])
            assert int(good.sum())==r['token_correct']
            assert int(good.all(-1).sum())==r['string_correct']
            assert (~good).sum(0).tolist()==r['digit_errors']
        sample_hashes[(r['task'],r['checkpoint'],r['memory_len'],r['T'],r['prefix'])]=digest
    for r in [r for r in rows if r['task']=='copying' and r['checkpoint']=='best']:
        key=('copying','final',r['memory_len'],r['T'],r['prefix'])
        same_copying &= sample_hashes[key]==r['observations_sha256']
    same_copying &= review['weights']['copying/best']==review['weights']['copying/final']
    assert same_copying
    # Preserve worker completion records alongside the human review.
    out=HERE/'analysis';out.mkdir(exist_ok=True)
    for task in TASKS:
        worker=json.loads((ROOT/f'worker-{task}.json').read_text())
        assert worker['state']=='complete' and worker['first_300_steps_match_benchmark']
        assert all(c['returncode']==0 for c in worker['commands']) and len(worker['commands'])==3
        save(out/f'worker-{task}.json',worker)
    colors={10:'#0072B2',16:'#D55E00',32:'#009E73',64:'#CC79A7'}
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for i,task in enumerate(TASKS):
        for m,color in colors.items():
            for cp,style in [('best','-'),('final','--')]:
                rr=[r for r in rows if r['task']==task and r['checkpoint']==cp and r['memory_len']==m and r['split']=='horizon']
                for j,field in enumerate(['string_acc','token_acc']):
                    axes[i,j].plot([r['T'] for r in rr],[100*r[field] for r in rr],style,color=color,label=f'M{m} {cp}')
        for j in range(2):
            axes[i,j].axvline(2028,color='gray',ls=':',label='Train max T')
            axes[i,j].set(xscale='log',xlabel='T (P=0)',ylabel='Accuracy (%)',ylim=(-2,102),
                          title=f'{task} / '+['exact string','digit'][j])
            axes[i,j].grid(alpha=.2);axes[i,j].legend(fontsize=7,ncol=2)
    fig.savefig(out/'horizons.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,task in zip(axes,TASKS):
        for m,color in colors.items():
            r=next(r for r in rows if r['task']==task and r['checkpoint']=='best' and r['memory_len']==m and r['T']==64 and r['prefix']==0)
            ax.plot(range(1,m+1),[100*(1-e/256) for e in r['digit_errors']],color=color,label=f'M{m}')
        ax.set(title=f'{task}: T64 / best / P0',xlabel='Digit position (1-based)',ylabel='Digit accuracy (%)',ylim=(-2,102))
        ax.grid(alpha=.2);ax.legend()
    fig.savefig(out/'digits.png',dpi=150);plt.close(fig)
    # Compact machine-readable tables for the report; keep best/final explicit.
    selected=[{k:r[k] for k in ['task','checkpoint','memory_len','T','prefix','n','string_correct','token_acc']}
              for r in rows if r['prefix']==0 and r['T'] in (1,16,64,256,1024,2048,8192,131072)]
    save(out/'selected.json',selected)
    save(out/'review.json',dict(passed=True,cpu_only=True,cells_recounted=880,
        campaign_sha256=sha(ROOT/'campaign.json'),original_review_sha256=sha(result/'review.json'),
        copying_best_final_same_weights_and_observations=same_copying,
        whole_batch_hours=campaign['whole_batch_hours'],
        execution_hours=(datetime.datetime.fromisoformat(campaign['finished'])-datetime.datetime.fromisoformat(campaign['started'])).total_seconds()/3600,
        script_sha256=sha(Path(__file__)),
        files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='review.json'}))
    print('Post-completion verification passed:880 cells; Copying best/final identical.')

if __name__=='__main__':main()
