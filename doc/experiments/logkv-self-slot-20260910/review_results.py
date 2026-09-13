"""CPU review of completed artifacts and paired digit/attention diagnosis."""
import csv
import json
import re
from datetime import datetime,timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import HERE, ROOT, RUN, BASELINE, CONTROL, save, sha


def main():
    dest=HERE/'results';diag=dest/'diagnostics'
    for line in (dest/'manifest.sha256').read_text().splitlines():
        digest,name=line.split('  ',1);assert sha(dest/name)==digest,name
    campaign=json.loads((ROOT/'campaign.json').read_text())
    assert campaign['state']=='complete-awaiting-review'
    assert len(campaign['commands'])==5 and all(c['returncode']==0 for c in campaign['commands'])
    assert campaign['elapsed_hours']<campaign['time_limit_hours']
    assert not campaign['next_stage_queued']
    assert (dest/'campaign.json').read_bytes()==(ROOT/'campaign.json').read_bytes()
    for name,h in campaign['script_sha256'].items():assert sha(HERE/name)==h
    for name in ('run_config.json','best.json','train_log.jsonl'):
        assert (dest/name).read_bytes()==(RUN/name).read_bytes()
    for name in ('train.log','best.log','final.log','diagnose.log','results_best.json','results_final.json','digits_best.json','digits_final.json'):
        assert (dest/name).read_bytes()==(ROOT/name).read_bytes()
    summary=json.loads((dest/'summary.json').read_text())
    rows=list(csv.DictReader((dest/'metrics.csv').open()))
    assert len(rows)==164
    for cp,sub in [('best','model_best'),('final','model')]:
        assert sha(RUN/sub/'model.safetensors')==summary['weight_sha256'][cp]
        assert sha(BASELINE/sub/'model.safetensors')==campaign['baseline_weights'][cp]
        assert (dest/f'control_results_{cp}.json').read_bytes()==(CONTROL/f'extension-131072/results_{cp}.json').read_bytes()
        logs={int(t):(tok,st) for t,tok,st in re.findall(r'T=\s*(\d+) \| token ([0-9.]+) \| string ([0-9.]+)',(dest/f'{cp}.log').read_text())}
        assert len(logs)==41
        digits=json.loads((dest/f'digits_{cp}.json').read_text())['records']
        for r in rows:
            if r['checkpoint']!=cp or r['mode']!='self-off':continue
            t=r['T'];batches=digits[t]
            pred=np.concatenate([x['prediction'] for x in batches]);target=np.concatenate([x['target'] for x in batches])
            assert pred.shape==target.shape==(256,10)
            correct=pred==target
            assert int(correct.sum())==int(r['token_correct']) and int(correct.all(-1).sum())==int(r['string_correct'])
            assert logs[int(t)]==(f"{float(r['token_acc']):.4f}",f"{float(r['string_acc']):.4f}")
    npz_files=list(diag.glob('*.npz'));assert len(npz_files)==48
    paired_memory=None
    for p in npz_files:
        assert p.read_bytes()==(ROOT/'diagnostics'/p.name).read_bytes()
        jp=p.with_suffix('.json');assert jp.read_bytes()==(ROOT/'diagnostics'/jp.name).read_bytes()
        x=np.load(p);j=json.loads(jp.read_text());target=x['target'];pred=x['logits'].argmax(-1)
        assert np.array_equal(target,j['target']) and np.array_equal(pred,j['prediction'])
        correct=pred==target
        assert int(correct.sum())==j['token_correct'] and int(correct.all(-1).sum())==j['string_correct']
        assert np.array_equal((~correct).sum(0),j['errors_per_digit'])
        logits=x['logits'];right=np.take_along_axis(logits,target[...,None],axis=-1)[...,0]
        rivals=logits.copy();np.put_along_axis(rivals,target[...,None],-np.inf,axis=-1)
        np.testing.assert_allclose(right-rivals.max(-1),j['margin'],atol=0,rtol=0)
        for key in x.files:
            assert np.isfinite(x[key]).all()
            if key.endswith('level_mass'):np.testing.assert_allclose(x[key].sum(-1),1,atol=2e-6)
        if '-paired-' in p.name:
            if paired_memory is None:paired_memory=target
            assert np.array_equal(paired_memory,target)
    replay={};wrong_indices={}
    for cp in ('best','final'):
        bf=np.load(diag/f'self-on-{cp}-bf16-standard-T131072.npz')
        fp=np.load(diag/f'self-on-{cp}-fp32-bf16-errors-T131072.npz')
        pred=bf['logits'].argmax(-1);target=bf['target'];bad=pred!=target;selected=bad.any(-1)
        assert selected.sum()==9 and bad.sum()==9
        assert np.array_equal(target[selected],fp['target'])
        fpbad=fp['logits'].argmax(-1)!=fp['target']
        wrong_indices[cp]=np.where(selected)[0]
        attention=[]
        for digit in np.where(bad.any(0))[0]:
            mask=bad[:,digit]
            for layer in (0,1):
                item=dict(digit=int(digit+1),layer=layer+1,error_examples=int(mask.sum()))
                for metric in ('self_mass','memory_overlap_mass'):
                    val=bf[f'layer{layer}_{metric}'].mean(1)[:,digit]
                    item[metric+'_incorrect']=float(val[mask].mean());item[metric+'_correct']=float(val[~mask].mean())
                attention.append(item)
        replay[cp]=dict(bf16_error_count_by_digit=bad.sum(0).tolist(),fp32_error_count_by_digit=fpbad.sum(0).tolist(),
            fp32_fixed_strings=int((~fpbad.any(-1)).sum()),selected_strings=9,
            bf16_wrong_margins=np.array(json.loads((diag/f'self-on-{cp}-bf16-standard-T131072.json').read_text())['margin'])[bad].tolist(),
            attention_by_digit=attention)
    shifts=[]
    for precision,T,digit in [('bf16',49152,8),('bf16',49153,7),('fp32',49152,8)]:
        x=np.load(diag/f'self-on-best-{precision}-paired-T{T}.npz');pred=x['logits'].argmax(-1);target=x['target'];bad=pred[:,digit]!=target[:,digit]
        shifts.append(dict(precision=precision,T=T,digit=digit+1,absolute_position_zero_based=T+10+digit,
             errors=int(bad.sum()),errors_matching_next_digit=int((pred[bad,digit]==target[bad,digit+1]).sum())))
    regular_ts=[16,8192,32768,65536,131072]
    regular=[]
    for T in regular_ts:
        x=np.load(diag/f'self-on-best-bf16-paired-T{T}.npz');pred=x['logits'].argmax(-1);target=x['target']
        errs=[dict(sample=int(i),digit=int(j+1),target=int(target[i,j]),prediction=int(pred[i,j])) for i,j in np.argwhere(pred!=target)]
        regular.append(dict(T=T,errors=errs))
    assert all(r['errors']==regular[0]['errors'] for r in regular)
    train=[json.loads(s) for s in (dest/'train_log.jsonl').read_text().splitlines()];train=[r for r in train if 'loss' in r]
    tail=[{k:r[k] for k in ('step','string_acc','ema_loss')} for r in train if r['step']>=49000]
    analysis=dict(reviewed=datetime.now(timezone.utc).isoformat(),replay=replay,
       best_final_failed_sample_overlap=int(len(np.intersect1d(wrong_indices['best'],wrong_indices['final']))),
       next_digit_errors=shifts,repeated_paired_error_across_lengths=regular,training_tail=tail,
       interpretation='Self removal worsens this seed; numerical precision explains some but not all failures; observed next-digit substitutions at a fixed absolute position do not identify the causal layer')
    save(dest/'diagnostic_analysis.json',analysis)
    save(dest/'review.json',dict(reviewed=analysis['reviewed'],all_checks_passed=True,successful_commands=5,
       new_evaluation_cells=82,comparison_cells=164,diagnostic_cases=48,archive_checksums_passed=True,
       source_files_match=True,checkpoint_hashes_verified=True,paired_memories_identical=True,
       digit_predictions_and_margins_verified=True,baseline_error_replay_pairing_verified=True,
       new_gpu_experiments_launched=False))
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for cp,color in [('best','tab:blue'),('final','tab:orange')]:
        axes[0].plot(range(1,11),replay[cp]['bf16_error_count_by_digit'],'o-',color=color,label=cp)
    axes[0].set(xlabel='Answer digit (1-based)',ylabel='Wrong examples / 256',title='Self-on, T131072');axes[0].legend()
    for mode,color in [('self-on','tab:blue'),('self-off','tab:orange')]:
        for cp,style in [('best','--'),('final','-')]:
            ts=[49151,49152,49153];vals=[]
            for T in ts:vals.append(json.loads((diag/f'{mode}-{cp}-bf16-paired-T{T}.json').read_text())['string_correct'])
            axes[1].plot(ts,vals,style+'o',color=color,label=f'{mode} {cp}')
    axes[1].set(xticks=[49151,49152,49153],xlabel='T (same 16 memories)',ylabel='Exact strings / 16',title='Boundary neighborhood');axes[1].ticklabel_format(style='plain',useOffset=False);axes[1].legend(fontsize=7)
    for name,color in [('self-on','tab:blue'),('self-off','tab:orange')]:
        path=(CONTROL/'copying/retrieval-rope/train_log.jsonl') if name=='self-on' else dest/'train_log.jsonl'
        records=[json.loads(s) for s in path.read_text().splitlines()];records=[r for r in records if 'loss' in r]
        axes[2].plot([r['step'] for r in records],[r['string_acc'] for r in records],color=color,lw=.8,label=name)
    axes[2].set(xlabel='Training step',ylabel='Interval exact string accuracy',title='Training stability');axes[2].legend()
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(dest/'diagnostic_overview.png',dpi=160);plt.close(fig)
    print(json.dumps({'audit':'passed','next_digit_errors':shifts,'replay_fp32_fixed':{cp:r['fp32_fixed_strings'] for cp,r in replay.items()},'failed_sample_overlap':analysis['best_final_failed_sample_overlap']},indent=2))

if __name__=='__main__':main()
