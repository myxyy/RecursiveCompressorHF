"""Explicitly authorized GPU-2 Copying extension; both frozen checkpoints."""
import datetime,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=Path('/mnt/raid0/RecursiveCompressor/experiments/logkv-rope-only-20260909')
ROOT=BASE/'extension-131072'
SOURCE=Path('/mnt/raid0/RecursiveCompressor/experiments/logkv-phase2-rope-20260908/source')
NAME='retrieval-rope-no-phase-fixed10-20260909'
COMMIT='afaea2b1d8b0bbc6e91aba25e1b9a3594297875b'

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path,x):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(path)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
 ROOT.mkdir(exist_ok=False)
 original=BASE/'exp/copying'/NAME
 run=ROOT/'exp/copying'/NAME;run.mkdir(parents=True)
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()==COMMIT
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=SOURCE,text=True)
 cfg=json.loads((original/'run_config.json').read_text())
 assert not cfg['phase_emb'] and cfg['retrieval_rope'] and not cfg['compressor_rope']
 assert cfg['level_decay_scale']==1 and cfg['steps']==50000
 shutil.copy2(original/'run_config.json',run/'run_config.json')
 hashes={}
 for cp,folder in (('best','model_best'),('final','model')):
  (run/folder).symlink_to(original/folder,target_is_directory=True)
  hashes[cp]=sha(original/folder/'model.safetensors')
  recorded=json.loads((HERE/'copying/summary.json').read_text())['runs'][0]
  assert recorded['mode']=='retrieval-rope' and hashes[cp]==recorded['weight_sha256'][cp]
 manifest=dict(state='running',started=now(),pid=os.getpid(),gpu=2,source=str(SOURCE),source_commit=COMMIT,
  task='copying',mode='retrieval-rope',phase_emb=False,samples=256,max_t_exp=17,
  checkpoints_sha256=hashes,commands=[],next_stage_queued=False,
  authorization='User explicitly authorized one extra GPU for this T131072 evaluation',
  time_limit_hours=1,original_results_untouched=True)
 save(ROOT/'manifest.json',manifest);deadline=time.monotonic()+3600
 env={**os.environ,'CUDA_VISIBLE_DEVICES':'2','DATA_DIR':str(ROOT),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
      'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}
 try:
  for cp in ('best','final'):
   cmd=[sys.executable,'exp/copying/evaluate.py','--run-name',NAME,'--samples','256','--max-t-exp','17',
        '--seed','12345','--precision','bf16','--checkpoint',cp,'--device','0']
   record=dict(stage=cp,command=cmd,started=now(),returncode=None)
   manifest['commands'].append(record);save(ROOT/'manifest.json',manifest)
   remaining=deadline-time.monotonic()
   if remaining<=0:raise TimeoutError('One-hour extension limit')
   with (ROOT/f'{cp}.log').open('w') as log:
    result=subprocess.run(cmd,cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=remaining)
   record.update(returncode=result.returncode,finished=now());save(ROOT/'manifest.json',manifest)
   if result.returncode:raise RuntimeError(f'{cp} failed: {result.returncode}')
   for name in ('results.json','plot.png'):
    p=Path(name);shutil.copy2(run/p,ROOT/f'{p.stem}_{cp}{p.suffix}')
  for cp,folder in (('best','model_best'),('final','model')):assert sha(original/folder/'model.safetensors')==hashes[cp]
  manifest['state']='complete-awaiting-review'
 except BaseException as exc:
  manifest.update(state='stopped',error=str(exc));raise
 finally:
  manifest['finished']=now();save(ROOT/'manifest.json',manifest)

if __name__=='__main__':main()
