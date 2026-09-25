"""CPU-only post-run audit and comparison; leaves original run artifacts unchanged."""
import argparse
from collections import Counter
from datetime import datetime
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

import torch

REPO = Path(__file__).resolve().parents[3]
ARCHIVE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/mnt/raid0/RecursiveCompressor/experiments/mamba2-copying-20260926-run'))
    args = parser.parse_args()
    root = args.root
    campaign = json.loads((root/'campaign.json').read_text())
    assert campaign['state']=='complete' and campaign['cells']==82 and campaign['independent_audit']
    manifest = campaign['preflight']['source']
    for name, expected in manifest.items():
        assert digest(root/'source'/name)==expected, name
        assert digest(REPO/name)==expected, name
    upstream = importlib.metadata.distribution('mamba-ssm')
    for name, expected in campaign['preflight']['upstream_files'].items():
        assert digest(upstream.locate_file(name))==expected, name
    logs = [json.loads(line) for line in (root/'exp/copying/mamba2/train_log.jsonl').read_text().splitlines()]
    assert [x['step'] for x in logs]==list(range(100,50001,100))
    assert all(torch.isfinite(torch.tensor(x['loss'])) for x in logs)
    best = json.loads((root/'exp/copying/mamba2/best.json').read_text())
    selected = max(logs,key=lambda x:(x['string_acc'],x['token_acc'],-x['ema_loss']))
    assert best['step']==selected['step']
    expected_grid=sorted(set(range(1,15))|{2**k for k in range(4,18)}|{3*2**(k-1) for k in range(4,17)})
    baseline_root=REPO/'doc/experiments/logkv-causal-conv-20260914/results'
    baseline=json.loads((baseline_root/'metrics.json').read_text())
    outputs, summaries, result_hashes = {}, [], {}
    for kind, folder in [('best','model_best'),('final','model')]:
        path=root/'results'/f'{kind}.json'
        result_hashes[kind]=digest(path)
        outputs[kind]=data=json.loads(path.read_text())
        assert [x['T'] for x in data['cells']]==expected_grid
        assert data['seed']==12345
        for name, expected in data['checkpoint_sha256'].items():
            assert digest(root/'exp/copying/mamba2'/folder/name)==expected
        old_digits=json.loads((baseline_root/'copying'/f'digits_{kind}.json').read_text())
        generator=torch.Generator().manual_seed(12345)
        for cell in data['cells']:
            t=cell['T']
            assert cell['samples']==256
            targets=torch.tensor(cell['targets'])
            pred=torch.tensor(cell['predictions'])
            margins=torch.tensor(cell['margins'])
            assert targets.shape==pred.shape==margins.shape==(256,10)
            assert torch.isfinite(margins).all()
            # Copying's only random draws are its 10 symbols per sample.
            assert torch.equal(targets,torch.randint(1,9,(256,10),generator=generator))
            old_targets=[row for batch in old_digits[str(t)] for row in batch['target']]
            assert targets.tolist()==old_targets
            matches=targets==pred
            assert cell['token_correct']==int(matches.sum())
            assert cell['string_correct']==int(matches.all(-1).sum())
            assert cell['digit_errors']==(~matches).sum(0).tolist()
            assert not ((margins>0)&~matches).any()
            assert not ((margins<0)&matches).any()
            control=next(c for c in baseline if c['task']=='copying' and c['checkpoint']==kind and c['T']==t)
            assert control['string_correct']==256 and control['token_correct']==2560
            counts=Counter(tuple(row) for row in cell['predictions'])
            summaries.append(dict(checkpoint=kind,T=t,samples=256,
                string_correct=cell['string_correct'],token_correct=cell['token_correct'],
                token_acc=cell['token_correct']/2560,string_acc=cell['string_correct']/256,
                digit_errors=cell['digit_errors'],margin_min=float(margins.min()),
                margin_median=float(margins.median()),unique_prediction_strings=len(counts),
                most_common_prediction=list(counts.most_common(1)[0][0]),
                most_common_prediction_count=counts.most_common(1)[0][1],
                state_bytes_per_example=cell['state_bytes_per_example'],
                logkv_string_correct=control['string_correct'],logkv_token_correct=control['token_correct']))
    assert all(a['targets']==b['targets'] for a,b in zip(outputs['best']['cells'],outputs['final']['cells']))
    assert {x['state_bytes_per_example'] for x in summaries}=={1063936}
    results=ARCHIVE/'results'
    results.mkdir(exist_ok=True)
    for kind in outputs:
        raw=(root/'results'/f'{kind}.json').read_bytes()
        (results/f'{kind}.json.gz').write_bytes(gzip.compress(raw,mtime=0))
        assert gzip.decompress((results/f'{kind}.json.gz').read_bytes())==raw
    save(results/'metrics.json',summaries)
    for name in ['campaign.json']:
        shutil.copy2(root/name,ARCHIVE/name)
    for src,dest in [('best.json','best-checkpoint.json'),('train_log.jsonl','train_log.jsonl')]:
        shutil.copy2(root/'exp/copying/mamba2'/src,ARCHIVE/dest)
    train_seconds=logs[-1]['elapsed_sec']
    wall=(datetime.fromisoformat(campaign['finished'])-datetime.fromisoformat(campaign['started'])).total_seconds()
    save(results/'review.json',dict(passed=True,cells=82,training_steps=50000,
        best_step=best['step'],final_step=50000,training_seconds=train_seconds,
        execution_seconds=wall,preflight_inclusive_hours=campaign['elapsed_hours'],
        regenerated_targets_match=True,paired_logkv_targets_match=True,
        checkpoint_hashes_verified=True,local_and_upstream_sources_verified=True,
        predictions_and_margins_recounted=True,state_bytes_per_example=1063936,
        source_manifest_sha256=digest(root/'preflight.json'),result_sha256=result_hashes,
        baseline_metrics_sha256=digest(baseline_root/'metrics.json'),
        evaluation_seconds={k:sum(c['seconds'] for c in d['cells']) for k,d in outputs.items()},
        final_training_metrics=logs[-1],new_gpu_work=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4),sharex=True)
    for ax,metric,label in zip(axs,['string_acc','token_acc'],['Exact match (%)','Digit accuracy (%)']):
        for kind in ['best','final']:
            rows=[c for c in summaries if c['checkpoint']==kind]
            ax.plot([c['T'] for c in rows],[c[metric]*100 for c in rows],'.-',label=f'Mamba-2 {kind}')
        ax.axhline(100,color='black',ls='--',label='LogKV best/final')
        ax.axvline(2028,color='grey',ls=':',label='Training max T')
        ax.set(xscale='log',xlabel='Horizon T',ylabel=label,ylim=(-3,104))
        ax.grid(alpha=.2)
    axs[1].axhline(12.5,color='grey',alpha=.5,ls='--',label='Uniform over digits 1–8')
    axs[1].legend(fontsize=8)
    fig.suptitle('Fixed-10 Copying: 256 paired examples per horizon, one training seed')
    fig.tight_layout()
    fig.savefig(results/'comparison.png',dpi=160)
    print('Verified 50,000 steps, 82 cells, paired LogKV targets, weights, sources and fixed-size state.')
    for x in summaries:
        if x['T']==131072: print(json.dumps(x))


if __name__=='__main__':
    main()
