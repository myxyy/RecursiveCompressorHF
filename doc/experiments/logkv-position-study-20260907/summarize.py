"""Regenerate and validate this archive without importing the experimental model."""
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
MODES=('none','phase2','binding','relative-kv','combined','combined-no-decay')
TASKS=('copying','selective-copying')
MEMORIES=(10,16,32,64)
COLORS=dict(zip(MODES,('#999999','#0072B2','#E69F00','#009E73','#CC79A7','#D55E00')))
TS=sorted(set(range(1,15)) | {2**k for k in range(4,18)} | {3*2**(k-1) for k in range(4,17)})
GRID={(m,t,0,'horizon') for m in MEMORIES for t in TS}
GRID |= {(m,t,p,'prefix') for m in MEMORIES for t in (16,64,256,2048) for p in (7,15,63)}
GRID |= {(128,t,p,'unseen-memory') for t in (16,64,256,2048) for p in (0,15)}
assert len(GRID)==220
records={}; scores={}; rows=[]; summary=[]; common=None; histogram=None
excluded={'mode','task','run_dir','num_params'}
for task in TASKS:
    for mode in MODES:
        manifest=json.loads((ROOT/f'{mode}-{task}.json').read_text())
        assert manifest['state']=='complete' and len(manifest['commands'])==3
        assert all(c['returncode']==0 for c in manifest['commands'])
        run=ROOT/'runs'/f'{mode}-{task}'
        cfg=json.loads((run/'run_config.json').read_text())
        comparable={k:v for k,v in cfg.items() if k not in excluded}
        if common is None: common=comparable
        assert comparable==common
        assert cfg['mode']==mode and cfg['task']==task
        assert len(cfg['initial_common_sha256'])==64
        assert cfg['memory_lengths']==list(MEMORIES) and cfg['prefix_range']==[0,63]
        assert cfg['steps']==50000 and cfg['max_t']==2028 and cfg['batch_size']==64 and cfg['grad_accum']==2
        data=[json.loads(line) for line in (run/'train_log.jsonl').read_text().splitlines()]
        train=[r for r in data if 'loss' in r]; valid=[r for r in data if 'validation' in r]
        assert [r['step'] for r in train]==list(range(100,50001,100))
        assert [r['step'] for r in valid]==list(range(2000,50001,2000))
        assert all(math.isfinite(r[k]) for r in train for k in ('loss','ema_loss'))
        assert all(sum(r['memory_histogram'].values())==r['step'] for r in train)
        if histogram is None: histogram=[r['memory_histogram'] for r in train]
        assert [r['memory_histogram'] for r in train]==histogram
        best=json.loads((run/'best.json').read_text())
        record=next(r for r in valid if r['step']==best['step'])
        assert record['macro_string']==best['macro_string'] and record['macro_token']==best['macro_token']
        ema_by_step={r['step']:r['ema_loss'] for r in train}
        winner=max(valid,key=lambda r:(r['macro_string'],r['macro_token'],-ema_by_step[r['step']]))
        assert winner['step']==best['step'] and ema_by_step[best['step']]==best['ema_loss']
        for r in valid:
            assert len(r['validation'])==32 and all(c['n']==32 for c in r['validation'])
            assert math.isclose(r['macro_string'],sum(c['string_acc'] for c in r['validation'])/32)
            assert math.isclose(r['macro_token'],sum(c['token_acc'] for c in r['validation'])/32)
        entry=dict(task=task,mode=mode,best_step=best['step'],best_validation=best,
                   num_params=cfg['num_params'],commit=manifest['commit'],final_train=train[-1])
        records[task,mode]=(train,valid)
        for checkpoint in ('best','final'):
            model_cfg=json.loads((run/f'model_config_{checkpoint}.json').read_text())
            assert model_cfg['phase_emb']==(mode=='phase2') and model_cfg['phase_levels']==2
            assert model_cfg['compressor_position_transform']==(mode in ('binding','combined','combined-no-decay'))
            assert model_cfg['relative_position_kv']==(mode in ('relative-kv','combined','combined-no-decay'))
            assert model_cfg['level_decay_scale']==(0 if mode=='combined-no-decay' else 1)
            assert not model_cfg['relative_position_bias'] and model_cfg['compressor_decay']==0
            assert not model_cfg['learnable_decay'] and not model_cfg['kv_norm'] and not model_cfg['v_norm_only']
            assert model_cfg['gated_attention'] and model_cfg['self_slot']
            result=json.loads((run/f'results_{checkpoint}.json').read_text())
            assert result['task']==task and result['checkpoint']==checkpoint
            assert result['complete'] and result['samples']==256 and result['seed']==12345
            assert result['precision']=='fp32 weights/bf16 autocast'
            assert len(result['results'])==220
            assert {(x['memory_len'],x['T'],x['prefix'],x['split']) for x in result['results']}==GRID
            for cell in result['results']:
                assert cell['n']==256 and 0<=cell['string_acc']<=cell['token_acc']<=1
                assert cell['string_acc']==cell['string_correct']/256
                assert cell['token_acc']==cell['token_correct']/(256*cell['memory_len'])
                key=(task,mode,checkpoint,cell['memory_len'],cell['T'],cell['prefix'])
                scores[key]=cell
                rows.append(dict(task=task,mode=mode,checkpoint=checkpoint,**cell))
            entry[checkpoint+'_all_horizons']={}
            for m in MEMORIES:
                cs=[scores[task,mode,checkpoint,m,t,0] for t in TS]
                entry[checkpoint+'_all_horizons'][m]=dict(min_string=min(c['string_acc'] for c in cs),
                    min_token=min(c['token_acc'] for c in cs),perfect_horizons=sum(c['string_acc']==1 for c in cs),
                    macro_string=sum(c['string_acc'] for c in cs)/len(cs),
                    macro_token=sum(c['token_acc'] for c in cs)/len(cs),total_horizons=41)
                for label,subset in [('trained_T_range',[c for c in cs if c['T']<=2028]),
                                     ('extrapolated_T',[c for c in cs if c['T']>2028])]:
                    entry[checkpoint+'_all_horizons'][m][label]=dict(cells=len(subset),
                        macro_string=sum(c['string_acc'] for c in subset)/len(subset),
                        macro_token=sum(c['token_acc'] for c in subset)/len(subset),
                        perfect_horizons=sum(c['string_acc']==1 for c in subset))
            for split in ('prefix','unseen-memory'):
                subset=[c for c in result['results'] if c['split']==split]
                entry[checkpoint+'_'+split]=dict(cells=len(subset),
                    macro_string=sum(c['string_acc'] for c in subset)/len(subset),
                    macro_token=sum(c['token_acc'] for c in subset)/len(subset))
        summary.append(entry)
assert len(rows)==5280
diagnostic_tables=['# 学習済み位置変換の事後診断','','均一な圧縮重み・固定乱数の8記号ベクトル・外付け線形復号器、各256例。LMの読み出し精度ではない。bf16化は最終要約だけ。','']
for task in TASKS:
    diagnostic=json.loads((ROOT/f'trained-encoder-{task}.json').read_text())
    assert diagnostic['post_hoc'] and len(diagnostic['results'])==36
    diagnostic_tables += [f'## {task}','','| mode | checkpoint | layer | M | rank / features | condition number | fp64 string (%) | bf16 summary string (%) |','|---|---|---:|---:|---:|---:|---:|---:|']
    expected={(mode,cp,layer,m) for mode in ('binding','combined','combined-no-decay')
              for cp in ('best','final') for layer in (0,1) for m in (4,16,64)}
    assert {(r['mode'],r['checkpoint'],r['layer'],r['memory_len']) for r in diagnostic['results']}==expected
    for r in diagnostic['results']:
        assert r['task']==task and r['features']==r['memory_len']*8 and 0<=r['rank']<=r['features']
        stats=json.loads((ROOT/'runs'/f"{r['mode']}-{task}"/f"position_stats_{r['checkpoint']}.json").read_text())
        assert stats[r['layer']]['checkpoint_sha256']==r['checkpoint_sha256']
        assert 0<=r['string_acc']<=r['token_acc']<=1 and 0<=r['bf16_summary_string_acc']<=1
        assert (r['string_acc']*256).is_integer() and (r['bf16_summary_string_acc']*256).is_integer()
        diagnostic_tables.append(f"| {r['mode']} | {r['checkpoint']} | {r['layer']} | {r['memory_len']} | {r['rank']} / {r['features']} | {r['condition_number']:.2e} | {100*r['string_acc']:.2f} | {100*r['bf16_summary_string_acc']:.2f} |")
    diagnostic_tables.append('')
(ROOT/'trained-encoder-diagnostics.md').write_text('\n'.join(diagnostic_tables))
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (ROOT/'comparison.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n'); writer.writeheader();writer.writerows(rows)
tables=['# Variable-memory study tables','','Cells: token / string accuracy (%), n=256.','']
for task in TASKS:
    for checkpoint in ('best','final'):
        tables += [f'## {task}: {checkpoint}','','| M | T | '+' | '.join(MODES)+' |','|---:|---:|'+'---:|'*6]
        for m in MEMORIES:
            for t in (16,64,256,2048,131072):
                cells=[scores[task,mode,checkpoint,m,t,0] for mode in MODES]
                tables.append(f'| {m} | {t:,} | '+' | '.join(f"{100*c['token_acc']:.2f} / {100*c['string_acc']:.2f}" for c in cells)+' |')
        tables.append('')
        for metric in ('token_acc','string_acc'):
            fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
            for ax,m in zip(axes.flat,MEMORIES):
                for mode in MODES:
                    ax.plot(TS,[scores[task,mode,checkpoint,m,t,0][metric] for t in TS],label=mode,color=COLORS[mode],lw=1.5)
                ax.axvline(2028,color='#bbbbbb',ls=':')
                if metric=='token_acc': ax.axhline(1/8,color='#555555',ls='--',lw=.8,label='8-symbol guess')
                ax.set_xscale('log');ax.set_ylim(-.02,1.02)
                ax.set_title(f'Memory M={m}');ax.set_xlabel('Horizon T');ax.set_ylabel(metric.replace('_',' '));ax.grid(alpha=.25)
            axes[0,0].legend(fontsize=8,ncol=2)
            fig.suptitle(f'{task}: {checkpoint}, {metric}; P=0, n=256, train seed 0')
            fig.tight_layout();fig.savefig(ROOT/f'{task}-{checkpoint}-{metric}.png',dpi=160);plt.close(fig)
(ROOT/'tables.md').write_text('\n'.join(tables))

extra=['# Prefix and unseen-memory probes','','Cells: token / string accuracy (%), n=256.','']
for task in TASKS:
    for checkpoint in ('best','final'):
        extra += [f'## {task}: {checkpoint}','','| M | T | P | '+' | '.join(MODES)+' |','|---:|---:|---:|'+'---:|'*6]
        for m,t,p,split in sorted(GRID):
            if split=='horizon': continue
            cells=[scores[task,mode,checkpoint,m,t,p] for mode in MODES]
            extra.append(f'| {m} | {t} | {p} | '+' | '.join(f"{100*c['token_acc']:.2f} / {100*c['string_acc']:.2f}" for c in cells)+' |')
        extra.append('')
(ROOT/'prefix-and-unseen-memory.md').write_text('\n'.join(extra))

fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
for r,task in enumerate(TASKS):
    for mode in MODES:
        train,valid=records[task,mode]
        axes[r,0].plot([x['step'] for x in train],[x['ema_loss'] for x in train],label=mode,color=COLORS[mode])
        axes[r,1].plot([x['step'] for x in valid],[x['macro_string'] for x in valid],label=mode,color=COLORS[mode])
    axes[r,0].set_yscale('log'); axes[r,0].set_ylabel('Training EMA loss')
    axes[r,1].set_ylabel('Validation macro string accuracy');axes[r,1].set_ylim(-.02,1.02)
    for ax in axes[r]: ax.set_title(task);ax.set_xlabel('Step');ax.grid(alpha=.25);ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(ROOT/'learning.png',dpi=160);plt.close(fig)
print('Validated 12 completed runs, matched initial/data settings, 5,280 evaluation rows and 72 post-hoc encoder diagnostics.')
