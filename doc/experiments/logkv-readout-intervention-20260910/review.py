"""Independent CPU review of archives, intervention effects and live weights."""
import csv,json
import numpy as np
from common import HERE,ROOT,BASELINE,JOBS,save,sha,now


def main():
 dest=HERE/'results'
 for line in (dest/'manifest.sha256').read_text().splitlines():
  h,p=line.split('  ',1);assert sha(dest/p)==h,p
 campaign=json.loads((ROOT/'campaign.json').read_text())
 assert campaign['state']=='complete-awaiting-review' and campaign['summary_returncode']==0
 assert all(j['returncode']==0 for j in campaign['jobs'])
 assert all(sha(HERE/name)==h for name,h in campaign['script_sha256'].items())
 raw=json.loads((dest/'raw_trace_manifest.json').read_text())
 assert len(raw)==350
 for x in raw:assert sha(x['path'])==x['sha256']
 rows=list(csv.DictReader((dest/'metrics.csv').open()));assert len(rows)==350
 for job in JOBS:
  status=json.loads((dest/job['name']/'status.json').read_text())
  model=BASELINE/('model_best' if job['checkpoint']=='best' else 'model')
  assert sha(model/'model.safetensors')==status['weight_sha256']
  for check in status['checks']:assert check['max_logit_difference']==0
  for r in (r for r in rows if r['job']==job['name']):
   name=r['name'];p=dest/job['name']/f'{name}.npz';a=np.load(p)
   orig=np.load(ROOT/job['name']/f'{name}.npz')
   assert np.array_equal(a['logits'],orig['logits']) and np.array_equal(a['target'],orig['target'])
   target=a['target'];pred=a['logits'].argmax(-1);digit=int(r['digit'])-1
   assert int((pred==target).all(-1).sum())==int(r['string_correct'])
   assert int((pred[:,digit]==target[:,digit]).sum())==int(r['digit_correct'])
   base=np.load(dest/job['name']/f"{r['cohort']}__T{r['T']}-d{r['digit']}-baseline.npz")
   assert np.array_equal(base['target'],target)
   b=base['logits'].argmax(-1)[:,digit]==target[:,digit];c=pred[:,digit]==target[:,digit]
   assert int((~b&c).sum())==int(r['rescued']) and int((b&~c).sum())==int(r['regressed'])
   assert np.array_equal(a['logits'][:,:digit],base['logits'][:,:digit])
   if r['patch'].startswith('sham'):assert np.array_equal(a['logits'],base['logits'])
 # Quantify whether interventions reproduce another intervention's complete
 # answer logits; this is observed evidence, not an expected test invariant.
 identities=[]
 for job in JOBS:
  for cohort in ('discovery','validation'):
   for T,digit in ((49152,9),(49153,8)):
    prefix=dest/job['name']/f'{cohort}__T{T}-d{digit}'
    def logits(patch):return np.load(str(prefix)+f'-{patch}.npz')['logits']
    baseline=logits('baseline');head=logits('head8');phase=logits('phase');mask=logits('mask')
    identities.append(dict(job=job['name'],cohort=cohort,T=T,
      phase_matches_head8_logits=bool(np.array_equal(phase,head)),mask_matches_baseline_logits=bool(np.array_equal(mask,baseline)),
      phase_head8_max_logit_difference=float(np.abs(phase-head).max()),mask_baseline_max_logit_difference=float(np.abs(mask-baseline).max())))
 save(dest/'review.json',dict(reviewed=now(),all_checks_passed=True,cases=350,
      full_raw_hashes_verified=True,archive_logits_match_raw=True,live_checkpoint_hashes_verified=True,
      discovery_reference_logits_bit_exact=True,sham_identity=True,prefix_and_earlier_answers_unchanged=True,
      new_gpu_work_during_review=False))
 save(dest/'intervention_identities.json',identities)
 print('All 350 cases and raw SHA256 verified')
 print(json.dumps(identities,indent=2))

if __name__=='__main__':main()
