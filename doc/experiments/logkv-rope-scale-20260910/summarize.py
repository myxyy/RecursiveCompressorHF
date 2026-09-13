"""CPU-only independent count audit; archive compact reproducible artifacts."""
import csv
import json
import math
import shutil
from common import HERE, ROOT, SOURCE, MODES, PAIRED_TS, run_dir, save, sha

def main():
    dest=HERE/'results'; dest.mkdir(exist_ok=True)
    grid=sorted(set(range(1,15))|{2**k for k in range(4,18)}|{3*2**(k-1) for k in range(4,17)})
    rows=[]; targets={}; weight_hashes={}; historical={}
    def audit(mode,cp,kind,metrics,records,ts,n):
        assert sorted(map(int,metrics))==ts and sorted(map(int,records))==ts
        for T in ts:
            target=[v for batch in records[str(T)] for v in batch['target']]
            pred=[v for batch in records[str(T)] for v in batch['prediction']]
            margins=[v for batch in records[str(T)] for v in batch['margin']]
            assert len(target)==len(pred)==len(margins)==n
            assert all(len(t)==10 and all(1<=v<=8 for v in t) for t in target)
            assert all(len(t)==10 and all(0<=v<=9 for v in t) for t in pred)
            assert all(len(t)==10 and all(math.isfinite(v) for v in t) for t in margins)
            key=(kind,T); assert targets.setdefault(key,target)==target
            tok=sum(a==b for t,p in zip(target,pred) for a,b in zip(t,p))
            st=sum(t==p for t,p in zip(target,pred)); metric=metrics[str(T)]
            assert metric['n']==n and metric['token_acc']==tok/(n*10) and metric['string_acc']==st/n
            errors=[sum(t[i]!=p[i] for t,p in zip(target,pred)) for i in range(10)]
            rows.append(dict(mode=mode,alpha=MODES[mode],checkpoint=cp,evaluation=kind,T=T,n=n,
                token_correct=tok,string_correct=st,token_acc=tok/(n*10),string_acc=st/n,digit_errors=errors))
    for mode in MODES:
        out=ROOT/mode; local=dest/mode; local.mkdir(exist_ok=True)
        worker=json.loads((out/'worker.json').read_text()); assert worker['state']=='complete'
        assert len(worker['commands'])==3 and all(c['returncode']==0 for c in worker['commands'])
        train=[json.loads(s) for s in (run_dir(mode)/'train_log.jsonl').read_text().splitlines()]
        intervals=[r for r in train if 'loss' in r]
        assert [r['step'] for r in intervals]==list(range(100,50001,100))
        assert all(math.isfinite(r[k]) for r in intervals for k in ['loss','ema_loss','token_acc','string_acc'])
        best=json.loads((run_dir(mode)/'best.json').read_text())
        chosen=max(intervals,key=lambda r:(r['string_acc'],r['token_acc'],-r['ema_loss']))
        assert best['step']==chosen['step']
        for cp,folder in [('best','model_best'),('final','model')]:
            weight_hashes[f'{mode}/{cp}']=sha(run_dir(mode)/folder/'model.safetensors')
            assert weight_hashes[f'{mode}/{cp}']==worker['weights'][cp]
            cfg=json.loads((run_dir(mode)/folder/'config.json').read_text())
            assert cfg['retrieval_rope_scale']==MODES[mode] and cfg['self_slot'] and cfg['retrieval_rope']
            assert not cfg['phase_emb'] and not cfg['compressor_rope']
            standard=json.loads((out/f'results_{cp}.json').read_text())
            assert standard['precision']=='bf16' and standard['samples']==256 and standard['train_max_t']==2028
            audit(mode,cp,'standard',standard['results'],json.loads((out/f'digits_{cp}.json').read_text()),grid,256)
            paired=json.loads((out/f'paired_{cp}.json').read_text()); assert paired['seed']==20260911
            audit(mode,cp,'paired',paired['results'],paired['records'],PAIRED_TS,32)
            memories=[b for batch in paired['records'][str(PAIRED_TS[0])] for b in batch['target']]
            for T in PAIRED_TS:
                assert memories==[b for batch in paired['records'][str(T)] for b in batch['target']]
            shutil.copy2(run_dir(mode)/folder/'config.json',local/f'config_{cp}.json')
        for f in out.iterdir():
            if f.suffix in ['.json','.log','.png']: shutil.copy2(f,local/f.name)
        for f in ['train_log.jsonl','run_config.json','best.json']: shutil.copy2(run_dir(mode)/f,local/f)
    save(dest/'metrics.json',rows)
    with (dest/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    table=['# RoPE angle comparison: automatically audited results','',
        'One training seed; best selected by training intervals. Human interpretation pending.','',
        '| alpha | checkpoint | T | exact / 256 | digit accuracy |','|---|---|---:|---:|---:|']
    for r in rows:
        if r['evaluation']=='standard' and r['T'] in [3,16,8192,32768,49152,65536,98304,131072]:
            table.append(f"| {r['alpha']:.6g} | {r['checkpoint']} | {r['T']} | {r['string_correct']} | {r['token_acc']:.6f} |")
    (dest/'README.md').write_text('\n'.join(table)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for ax,cp in zip(axes,['best','final']):
        for mode in MODES:
            rr=[r for r in rows if r['checkpoint']==cp and r['mode']==mode and r['evaluation']=='standard']
            ax.plot([r['T'] for r in rr],[r['string_acc'] for r in rr],'.-',label=mode)
        ax.set(xscale='log',ylim=(-.02,1.02),xlabel='T',ylabel='Exact string accuracy',title=cp)
        ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(dest/'comparison.png',dpi=150);plt.close(fig)
    save(dest/'review.json',dict(passed=True,standard_cells=3*2*41,paired_cells=3*2*len(PAIRED_TS),
        initial_preflight=json.loads((HERE/'preflight.json').read_text()),weights=weight_hashes,
        result_hashes={str(f.relative_to(dest)):sha(f) for f in dest.rglob('*') if f.is_file() and f.name!='review.json'},
        limitations='Single training seed, 32 paired samples, no 16M test, no main merge.'))
    print('All training, configuration, checkpoint and per-digit count audits passed.',flush=True)
if __name__=='__main__': main()
