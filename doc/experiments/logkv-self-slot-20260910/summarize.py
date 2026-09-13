"""CPU audit/archive and comparison; never schedules GPU work or commits."""
import csv
import json
import math
import re
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import HERE, ROOT, RUN, BASELINE, CONTROL, save, sha


def main():
    dest=HERE/'results';dest.mkdir(exist_ok=False)
    manifest=json.loads((ROOT/'campaign.json').read_text())
    done=[c for c in manifest['commands'] if c['stage']!='summarize']
    assert len(done)==4 and all(c['returncode']==0 for c in done)
    cfg=json.loads((RUN/'run_config.json').read_text());old=json.loads((BASELINE/'run_config.json').read_text())
    assert not cfg['self_slot'] and old['self_slot']
    assert {k:v for k,v in cfg.items() if k not in ('run_name','self_slot')}=={k:v for k,v in old.items() if k not in ('run_name','self_slot')}
    records=[json.loads(s) for s in (RUN/'train_log.jsonl').read_text().splitlines()]
    train=[r for r in records if 'loss' in r]
    assert [r['step'] for r in train]==list(range(100,50001,100))
    assert all(math.isfinite(r['loss']) and math.isfinite(r['ema_loss']) for r in train)
    best=json.loads((RUN/'best.json').read_text())
    assert best['step']==max(train,key=lambda r:(r['string_acc'],r['token_acc'],-r['ema_loss']))['step']
    for name in ('run_config.json','train_log.jsonl','best.json'):shutil.copy2(RUN/name,dest/name)
    shutil.copy2(BASELINE/'run_config.json',dest/'control_run_config.json')
    for name in ('train.log','best.log','final.log','diagnose.log','preflight.json','diagnostic_preflight.json','gpu_smoke.json','baseline_replay_smoke.json'):
        src=ROOT/name if (ROOT/name).exists() else HERE/name
        shutil.copy2(src,dest/name)
    rows=[];hashes={}
    expected=sorted(set(range(1,15))|{2**k for k in range(4,18)}|{3*2**(k-1) for k in range(4,17)})
    for cp,sub in [('best','model_best'),('final','model')]:
        hashes[cp]=sha(RUN/sub/'model.safetensors')
        shutil.copy2(RUN/sub/'config.json',dest/f'config_{cp}.json')
        for name in (f'results_{cp}.json',f'plot_{cp}.png',f'digits_{cp}.json'):shutil.copy2(ROOT/name,dest/name)
        digits=json.loads((ROOT/f'digits_{cp}.json').read_text())['records']
        for mode,path in [('self-off',ROOT/f'results_{cp}.json'),('self-on',CONTROL/f'extension-131072/results_{cp}.json')]:
            data=json.loads(path.read_text());assert data['samples']==256 and data['precision']=='bf16'
            assert sorted(map(int,data['results']))==expected
            if mode=='self-on':shutil.copy2(path,dest/f'control_results_{cp}.json')
            logs=dict((int(t),(tok,st)) for t,tok,st in re.findall(r'T=\s*(\d+) \| token ([0-9.]+) \| string ([0-9.]+)',(ROOT/f'{cp}.log').read_text()))
            for T in expected:
                cell=data['results'][str(T)];nt=round(cell['token_acc']*2560);ns=round(cell['string_acc']*256)
                assert cell['n']==256 and 0<=ns<=256 and 10*ns<=nt<=2304+ns
                assert abs(nt-cell['token_acc']*2560)<1e-8 and abs(ns-cell['string_acc']*256)<1e-8
                if mode=='self-off':
                    assert logs[T]==(f"{cell['token_acc']:.4f}",f"{cell['string_acc']:.4f}")
                    target=np.concatenate([x['target'] for x in digits[str(T)]])
                    pred=np.concatenate([x['prediction'] for x in digits[str(T)]])
                    margin=np.concatenate([x['margin'] for x in digits[str(T)]])
                    assert target.shape==pred.shape==margin.shape==(256,10) and np.isfinite(margin).all()
                    assert int((target==pred).sum())==nt and int((target==pred).all(-1).sum())==ns
                rows.append(dict(mode=mode,checkpoint=cp,T=T,string_correct=ns,token_correct=nt,**cell))
    with (dest/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    diag=json.loads((ROOT/'diagnostics/summary.json').read_text())
    assert diag['baseline_T131072_reproduced'] and diag['checkpoint_hashes_unchanged']
    assert len(diag['cases'])==44
    for p in (ROOT/'diagnostics').glob('*.npz'):
        with np.load(p) as a:
            data=json.loads(p.with_suffix('.json').read_text())
            logits=a['logits'];target=a['target'];correct=logits.argmax(-1)==target
            assert int(correct.sum())==data['token_correct'] and int(correct.all(-1).sum())==data['string_correct']
            for key in a.files:
                assert np.isfinite(a[key]).all()
                if key.endswith('level_mass'):np.testing.assert_allclose(a[key].sum(-1),1,atol=2e-6)
    shutil.copytree(ROOT/'diagnostics',dest/'diagnostics')
    summary=dict(evaluation_cells=82,comparison_cells=164,all_checks_passed=True,best_step=best['step'],
         train_elapsed_minutes=train[-1]['elapsed_sec']/60,weight_sha256=hashes,diagnostic_cases=44,
         baseline_standard_replay_cases=2,baseline_fp32_error_replay_cases=2,
         scope='Single training seed, fixed ten digits; paired n16 diagnosis; no Selective or 16M')
    save(dest/'summary.json',summary)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for mode,color in [('self-on','tab:blue'),('self-off','tab:orange')]:
        for cp,style in [('best','--'),('final','-')]:
            rs=[r for r in rows if r['mode']==mode and r['checkpoint']==cp]
            for ax,metric in zip(axes,['string_acc','token_acc']):
                ax.plot([r['T'] for r in rs],[r[metric] for r in rs],style+'o',color=color,ms=3,label=f'{mode} {cp}')
    for ax,ylabel in zip(axes,['Exact string accuracy','Digit accuracy']):
        ax.set(xscale='log',xlabel='T',ylabel=ylabel,ylim=(-.02,1.02));ax.axvline(2028,color='gray',ls=':');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(dest/'comparison.png',dpi=160);plt.close(fig)
    lines=['# 自己スロット除去 Copying：CPU検証済み結果','',
       '固定10桁・各T256例。各欄は完全一致例数（自己スロットあり / なし）。',
       '既存対照と新規学習は同じ凍結モデルコード・再構成初期重み・データseed。各条件1学習seed。','',
       '| T | best | final |','|---:|---:|---:|']
    for T in (3,16,8192,32768,49152,65536,131072):
        cells=[' / '.join(str(next(r for r in rows if r['mode']==m and r['checkpoint']==cp and r['T']==T)['string_correct']) for m in ('self-on','self-off')) for cp in ('best','final')]
        lines.append('| '+str(T)+' | '+' | '.join(cells)+' |')
    lines+=['','![比較](comparison.png)','',
       '[全評価](metrics.csv)、[検証記録](summary.json)、[診断一覧](diagnostics/summary.json)。',
       '診断JSONに誤る桁・正解と最大競合logitの差・注意量、NPZに全headの注意量と出力logitを保存。',
       'memory_overlap_massは元の10桁区間に重なるスロットへの注意量であり、情報保持量や因果的重要度ではない。',
       'fp32誤答再評価はbf16で誤った例に限定した診断で、無作為標本の精度ではない。','']
    (dest/'README.md').write_text('\n'.join(lines))
    (dest/'manifest.sha256').write_text(''.join(f'{sha(p)}  {p.relative_to(dest)}\n' for p in sorted(dest.rglob('*')) if p.is_file() and p.name not in ('manifest.sha256','campaign.json')))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
