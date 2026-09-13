"""CPU-only independent count audit; archive compact reproducible artifacts."""
import csv
import importlib.util
import torch
import json
import math
import shutil
from common import HERE, ROOT, SOURCE, MODES, run_dir, save, sha

def main():
    dest=HERE/'results'; dest.mkdir(exist_ok=True)
    grid=sorted(set(range(1,15))|{2**k for k in range(4,14)}|{3*2**(k-1) for k in range(4,13)})
    rows=[]; targets={}; positions_by_t={}; weight_hashes={}
    # Independently replay the frozen Selective generator with the exact standard
    # evaluator RNG stream and minibatch sizes; do not trust task metadata alone.
    torch.set_num_threads(1)
    spec=importlib.util.spec_from_file_location('audit_selective_task',SOURCE/'exp/selective-copying/task.py')
    task=importlib.util.module_from_spec(spec); spec.loader.exec_module(task)
    assert task.TASK_NAME=='selective-copying'
    generator=torch.Generator().manual_seed(12345)
    expected={}
    for T in grid:
        batch=max(1,min(256,2**19//(T+20))); done=0; expected[T]=[]
        while done<256:
            b=min(batch,256-done)
            ids,labels=task.make_batch(T,b,generator=generator)
            positions=((ids>=1)&(ids<=8)).nonzero()[:,1].reshape(b,10)
            assert torch.equal(ids.gather(1,positions),labels[:,-10:])
            expected[T].append(dict(target=labels[:,-10:].tolist(),positions=positions.tolist()))
            done+=b
    def audit(mode,cp,kind,metrics,records,ts,n):
        assert sorted(map(int,metrics))==ts and sorted(map(int,records))==ts
        for T in ts:
            assert [{k:b[k] for k in ['target','positions']} for b in records[str(T)]]==expected[T]
            assert all(b['memory']==b['target'] for b in records[str(T)])
            target=[v for batch in records[str(T)] for v in batch['target']]
            pred=[v for batch in records[str(T)] for v in batch['prediction']]
            margins=[v for batch in records[str(T)] for v in batch['margin']]
            assert len(target)==len(pred)==len(margins)==n
            assert all(len(t)==10 and all(1<=v<=8 for v in t) for t in target)
            assert all(len(t)==10 and all(0<=v<=9 for v in t) for t in pred)
            assert all(len(t)==10 and all(math.isfinite(v) for v in t) for t in margins)
            positions=[v for batch in records[str(T)] for v in batch['positions']]
            assert len(positions)==n
            assert all(len(p)==10 and all(isinstance(v,int) and 0<=v<T+9 for v in p) and
                all(a<b for a,b in zip(p,p[1:])) for p in positions)
            key=(kind,T); assert targets.setdefault(key,target)==target
            assert positions_by_t.setdefault(key,positions)==positions
            tok=sum(a==b for t,p in zip(target,pred) for a,b in zip(t,p))
            st=sum(t==p for t,p in zip(target,pred)); metric=metrics[str(T)]
            assert metric['n']==n and metric['token_acc']==tok/(n*10) and metric['string_acc']==st/n
            errors=[sum(t[i]!=p[i] for t,p in zip(target,pred)) for i in range(10)]
            rows.append(dict(mode=mode,position_mode=mode,checkpoint=cp,evaluation=kind,T=T,n=n,
                token_correct=tok,string_correct=st,token_acc=tok/(n*10),string_acc=st/n,digit_errors=errors))
    for mode in MODES:
        out=ROOT/mode; local=dest/mode; local.mkdir(exist_ok=True)
        worker=json.loads((out/'worker.json').read_text()); assert worker['state']=='complete'
        init=json.loads((run_dir(mode)/'initialization_audit.json').read_text())
        assert init['passed'] and init['deterministic_algorithms'] and init['task']=='selective-copying'
        assert init['task_sha256']==sha(SOURCE/'exp/selective-copying/task.py')
        assert init['cublas_workspace_config']==':4096:8'
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
            assert cfg['aligned_rope']==(mode=='aligned') and cfg['self_slot']
            assert cfg['retrieval_rope']==(mode=='local-control')
            assert cfg['aligned_rope_scale']==cfg['retrieval_rope_scale']==1.0
            assert not cfg['phase_emb'] and not cfg['compressor_rope']
            standard=json.loads((out/f'results_{cp}.json').read_text())
            assert standard['task']=='selective-copying'
            assert standard['task_sha256']==sha(SOURCE/'exp/selective-copying/task.py')
            assert standard['precision']=='bf16' and standard['samples']==256 and standard['train_max_t']==2028
            audit(mode,cp,'standard',standard['results'],json.loads((out/f'digits_{cp}.json').read_text()),grid,256)
            shutil.copy2(run_dir(mode)/folder/'config.json',local/f'config_{cp}.json')
        for f in out.iterdir():
            if f.suffix in ['.json','.log','.png']: shutil.copy2(f,local/f.name)
        for f in ['train_log.jsonl','run_config.json','best.json','initialization_audit.json']: shutil.copy2(run_dir(mode)/f,local/f)
    save(dest/'metrics.json',rows)
    with (dest/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    table=['# End-anchored RoPE Selective Copying: automatically audited results','',
        'One training seed; best selected by training intervals. Human interpretation pending.','',
        '| mode | checkpoint | T | exact / 256 | digit accuracy |','|---|---|---:|---:|---:|']
    for r in rows:
        if r['evaluation']=='standard' and r['T'] in [3,16,64,128,256,512,1024,2048,4096,8192]:
            table.append(f"| {r['mode']} | {r['checkpoint']} | {r['T']} | {r['string_correct']} | {r['token_acc']:.6f} |")
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
    save(dest/'review.json',dict(passed=True,task='selective-copying',standard_cells=2*2*33,scatter_positions_matched=True,
        initial_preflight=json.loads((HERE/'preflight.json').read_text()),
        model_validation=json.loads((HERE/'validation.json').read_text()),weights=weight_hashes,
        result_hashes={str(f.relative_to(dest)):sha(f) for f in dest.rglob('*') if f.is_file() and f.name!='review.json'},
        limitations='Single training seed; standard horizon memories/scatter positions match across modes and checkpoints, not across T. No additional long horizon sweep or main merge.'))
    print('All training, configuration, checkpoint and per-digit count audits passed.',flush=True)
if __name__=='__main__': main()
