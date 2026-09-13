"""Audit all trials and archive compact per-example outputs and causal counts."""
import csv,json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import HERE,ROOT,JOBS,save,sha


def main():
 dest=HERE/'results';dest.mkdir(exist_ok=False)
 rows=[];archives=[];trace_comparisons=[]
 for job in JOBS:
  src=ROOT/job['name'];out=dest/job['name'];out.mkdir()
  status=json.loads((src/'status.json').read_text());summary=json.loads((src/'summary.json').read_text())
  assert status['state']=='complete' and status['prefix_immutable'] and status['weight_hash_unchanged']
  for name in ('status.json','summary.json'):shutil.copy2(src/name,out/name)
  shutil.copy2(ROOT/f"{job['name']}.log",dest/f"{job['name']}.log")
  cases={r['name']:r for r in summary['cases']}
  expected=126 if job['precision']=='bf16' else 98
  assert len(cases)==expected,(job['name'],len(cases))
  for name,meta in cases.items():
   path=src/f'{name}.npz';arr=np.load(path);data=json.loads((src/f'{name}.json').read_text())
   logits=arr['logits'];target=arr['target'];pred=logits.argmax(-1);digit=meta['digit']-1
   assert np.isfinite(logits).all() and np.array_equal(target,data['target']) and np.array_equal(pred,data['prediction'])
   assert logits.shape==(meta['n'],10,10) and target.shape==(meta['n'],10)
   assert int((pred==target).all(-1).sum())==meta['string_correct']
   assert int((pred==target).sum())==meta['token_correct']
   assert int((pred[:,digit]==target[:,digit]).sum())==meta['digit_correct']
   rivals=logits.copy();np.put_along_axis(rivals,target[...,None],-np.inf,-1)
   margin=np.take_along_axis(logits,target[...,None],-1)[...,0]-rivals.max(-1)
   np.testing.assert_allclose(margin,data['margin'],atol=0,rtol=0)
   baseline_name=f"{meta['cohort']}__T{meta['T']}-d{meta['digit']}-baseline"
   baseline=np.load(src/f'{baseline_name}.npz');basepred=baseline['logits'].argmax(-1)
   assert np.array_equal(target,baseline['target'])
   basecorrect=basepred[:,digit]==target[:,digit];correct=pred[:,digit]==target[:,digit]
   assert np.array_equal(logits[:,:digit],baseline['logits'][:,:digit])
   if meta['patch'].startswith('sham'):assert np.array_equal(logits,baseline['logits'])
   row={**meta,'job':job['name'],'baseline_digit_correct':int(basecorrect.sum()),
        'rescued':int((~basecorrect&correct).sum()),'regressed':int((basecorrect&~correct).sum()),
        'mean_margin':float(margin[:,digit].mean()),'next_digit_errors':0}
   if digit<9:row['next_digit_errors']=int((~correct&(pred[:,digit]==target[:,digit+1])).sum())
   if meta['donor_T'] is not None:
    donor_name=f"{meta['cohort']}__T{meta['donor_T']}-d{meta['digit']}-baseline"
    donor=np.load(src/f'{donor_name}.npz');assert np.array_equal(donor['target'],target)
    dc=donor['logits'].argmax(-1)[:,digit]==target[:,digit]
    row.update(donor_correct=int(dc.sum()),donor_correct_baseline_wrong=int((dc&~basecorrect).sum()),
               rescued_with_correct_donor=int((dc&~basecorrect&correct).sum()))
    if meta['patch']=='head8':np.testing.assert_array_equal(arr['l1_raw'][:,7],donor['l1_raw'][:,7])
    if meta['patch']=='head1':np.testing.assert_array_equal(arr['l1_raw'][:,0],donor['l1_raw'][:,0])
    if meta['patch']=='l2_gate':np.testing.assert_array_equal(arr['l2_gate_logit'],donor['l2_gate_logit'])
    # Norm differences are descriptive; no assumption of a semantic clock.
    changes={}
    for key in ('l2_query','l2_gate_logit','l2_gated_heads','l2_attention_branch','l2_ffn_branch','l2_block_output'):
     x=arr[key].reshape(meta['n'],-1);b=baseline[key].reshape(meta['n'],-1);d=donor[key].reshape(meta['n'],-1)
     changes[key]=dict(mean_l2_to_baseline=float(np.linalg.norm(x-b,axis=1).mean()),
                       mean_l2_to_donor=float(np.linalg.norm(x-d,axis=1).mean()))
    trace_comparisons.append(dict(job=job['name'],case=name,changes=changes))
   rows.append(row)
   shutil.copy2(src/f'{name}.json',out/f'{name}.json')
   # All intervention vectors remain on RAID; compact logits/targets and
   # discovery/validation baseline vectors are archived in the repository.
   if meta['patch']=='none':shutil.copy2(path,out/path.name)
   else:np.savez_compressed(out/path.name,logits=logits,target=target)
   archives.append(dict(path=str(path),sha256=sha(path),archived_full_vectors=meta['patch']=='none'))
 save(dest/'raw_trace_manifest.json',archives)
 save(dest/'trace_comparisons.json',trace_comparisons)
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with (dest/'metrics.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
 save(dest/'summary.json',dict(all_checks_passed=True,cases=len(rows),jobs=3,discovery_samples=16,validation_samples=32,
      weights_unchanged=True,prefix_states_unchanged=True,sham_and_observation_identity=True,
      scope='Counterfactual single-answer interventions, not a deployable positional encoding or a new trained model'))
 fig,axes=plt.subplots(1,3,figsize=(16,4),layout='constrained')
 patches=['none','head8','head1','phase','mask','phase_mask','l1_block','l2_gate','l2_raw','l2_raw_gate','l2_ffn']
 for ax,job in zip(axes,JOBS):
  for cohort,color in [('discovery','tab:blue'),('validation','tab:orange')]:
   rs=[next(r for r in rows if r['job']==job['name'] and r['cohort']==cohort and r['T']==49152 and r['patch']==p) for p in patches]
   ax.plot(range(len(patches)),[r['digit_correct']/r['n'] for r in rs],'o-',label=cohort,color=color)
  ax.set(title=job['name']+' / answer digit 9',xticks=range(len(patches)),xticklabels=patches,ylim=(-.03,1.03),ylabel='Digit accuracy')
  ax.tick_params(axis='x',rotation=70);ax.grid(alpha=.2);ax.legend(fontsize=7)
 fig.savefig(dest/'interventions.png',dpi=160);plt.close(fig)
 lines=['# 読み出し介入実験：CPU検証済み結果','',
 '既存checkpointでの単一回答位置への反実仮想介入。改善しても、そのまま実装可能な修正の評価ではない。',
 'discoveryは既存seed4321の16例、validationは独立seed20260911の32例。',
 '下表はT49152の9桁目の正答数。best/finalとも自己スロットあり・phase2なし。','',
 '| 介入 | best bf16 発見/検証 | final bf16 発見/検証 | best fp32 発見/検証 |',
 '|---|---:|---:|---:|']
 for patch in patches:
  cells=[]
  for job in JOBS:
   cells.append(' / '.join(str(next(r for r in rows if r['job']==job['name'] and r['cohort']==c and r['T']==49152 and r['patch']==patch)['digit_correct']) for c in ('discovery','validation')))
  lines.append('| '+patch+' | '+' | '.join(cells)+' |')
 lines+=['','![介入比較](interventions.png)','',
 '[全350条件](metrics.csv)、[検証記録](summary.json)、[中間表現の変化](trace_comparisons.json)。',
 '全条件のlogit/targetとJSONを保存。baselineの中間表現はNPZにも保存。',
 '介入後の全中間表現はRAIDに保持し、[生データの場所とSHA256](raw_trace_manifest.json)を保存した。','']
 (dest/'README.md').write_text('\n'.join(lines))
 (dest/'manifest.sha256').write_text(''.join(f'{sha(p)}  {p.relative_to(dest)}\n' for p in sorted(dest.rglob('*')) if p.is_file() and p.name not in ('manifest.sha256','campaign.json')))
 print(json.dumps(dict(cases=len(rows),all_checks_passed=True)))

if __name__=='__main__':main()
