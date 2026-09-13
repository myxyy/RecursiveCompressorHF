"""Four no-phase experiments, independent tasks, <=2 GPUs and <=8h total."""
import concurrent.futures,datetime,fcntl,json,os,shutil,subprocess,sys,threading,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/raid0/RecursiveCompressor/experiments/logkv-rope-only-20260909')
SOURCE=Path('/mnt/raid0/RecursiveCompressor/experiments/logkv-phase2-rope-20260908/source')
COMMIT='afaea2b1d8b0bbc6e91aba25e1b9a3594297875b'
MODES=('retrieval-rope','compressor-rope')

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def train_command(task,mode):
 return [sys.executable,f'exp/{task}/train.py','--run-name',f'{mode}-no-phase-fixed10-20260909',
  '--arch','logkv','--phase-levels','2','--gated-attention','--self-slot',f'--{mode}',
  '--t-dist','loguniform','--max-t','2028','--steps','50000','--batch-size','64','--grad-accum','1',
  '--lr','0.0003','--warmup','1000','--d-model','512','--num-heads','8','--d-ff','1024',
  '--num-layers','2','--chunk-size','4','--loss-positions','all','--seed','0','--device','0']

def main():
 lock=(ROOT/'campaign.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (ROOT/'campaign.json').exists():raise FileExistsError('Campaign already exists; no automatic restart')
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()==COMMIT
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=SOURCE,text=True)
 started=time.monotonic();deadline=started+8*3600;stop=threading.Event()
 campaign=dict(state='running-copying',started=now(),pid=os.getpid(),gpu_limit=2,total_limit_hours=8,
  source=str(SOURCE),source_commit=COMMIT,initial_estimate_hours=7.75,
  selective_started=False,next_stage_queued=True,phase_emb=False)
 save(ROOT/'campaign.json',campaign)
 def command_run(command,env,logpath):
  if stop.is_set():raise RuntimeError('Another worker failed')
  remaining=deadline-time.monotonic()
  if remaining<=0:raise TimeoutError('Eight-hour campaign limit')
  with logpath.open('w') as log:
   p=subprocess.Popen(command,cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT)
   try:
    while True:
     try:code=p.wait(timeout=min(5,max(.01,deadline-time.monotonic())));break
     except subprocess.TimeoutExpired:
      if stop.is_set():raise RuntimeError('Another worker failed')
      if time.monotonic()>=deadline:raise TimeoutError('Eight-hour campaign limit')
    if code:raise RuntimeError(f'Command failed: {code}; see {logpath}')
    return code
   except BaseException:
    stop.set();p.terminate()
    try:p.wait(timeout=20)
    except subprocess.TimeoutExpired:p.kill();p.wait()
    raise
 def task_run(task):
  dest=ROOT/task;dest.mkdir(exist_ok=False)
  stage=dict(task=task,source=str(SOURCE),commit=COMMIT,started=now(),gpu_limit=2,steps=50000,
   max_t_exp=13,samples=256,state='running',phase_emb=False,next_stage_queued=False)
  save(dest/'stage.json',stage)
  def worker(gpu,mode):
   name=f'{mode}-no-phase-fixed10-20260909';run_dir=ROOT/'exp'/task/name
   if run_dir.exists():raise FileExistsError(run_dir)
   env={**os.environ,'DATA_DIR':str(ROOT),'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'1',
        'MKL_NUM_THREADS':'1','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}
   record=dict(mode=mode,task=task,gpu=gpu,source_commit=COMMIT,started=now(),commands=[],state='starting')
   commands=[('train',train_command(task,mode))]
   for cp in ('best','final'):
    commands.append((cp,[sys.executable,f'exp/{task}/evaluate.py','--run-name',name,'--samples','256',
     '--max-t-exp','13','--seed','12345','--precision','bf16','--checkpoint',cp,'--device','0']))
   try:
    for label,cmd in commands:
     entry=dict(stage=label,command=cmd,started=now(),returncode=None)
     record['state']=label;record['commands'].append(entry);save(dest/f'{mode}.json',record)
     code=command_run(cmd,env,dest/f'{mode}-{label}.log')
     entry.update(returncode=code,finished=now());save(dest/f'{mode}.json',record)
     if label=='train':
      cfg=json.loads((run_dir/'run_config.json').read_text())
      assert not cfg['phase_emb'] and cfg['retrieval_rope']==(mode=='retrieval-rope')
      assert cfg['compressor_rope']==(mode=='compressor-rope')
     else:
      for filename in ('results.json','plot.png'):
       path=Path(filename);shutil.copy2(run_dir/path,run_dir/f'{path.stem}_{label}{path.suffix}')
    record['state']='complete'
   except BaseException as exc:
    stop.set();record.update(state='stopped',error=str(exc));raise
   finally:
    record['finished']=now();save(dest/f'{mode}.json',record)
  try:
   with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    jobs=[pool.submit(worker,1,'retrieval-rope'),pool.submit(worker,0,'compressor-rope')]
    for job in concurrent.futures.as_completed(jobs):job.result()
   stage['state']='complete-awaiting-review'
  except BaseException as exc:
   stage.update(state='stopped',error=str(exc));raise
  finally:
   stage['finished']=now();save(dest/'stage.json',stage)
  with (ROOT/f'finish-{task}.log').open('w') as log:
   subprocess.run([sys.executable,str(HERE/'finish_task.py'),'--task',task],stdout=log,stderr=subprocess.STDOUT,check=True)
  with (ROOT/f'compare-{task}.log').open('w') as log:
   subprocess.run([sys.executable,str(HERE/'compare_phase.py'),'--task',task],stdout=log,stderr=subprocess.STDOUT,check=True)
 try:
  task_run('copying')
  elapsed=(time.monotonic()-started)/3600
  # Prior complete stage times: Selective retrieval / Copying retrieval.
  estimate=elapsed*(3.8228/3.8875)+.10
  campaign.update(copying_audit_passed=True,copying_elapsed_hours=elapsed,selective_estimate_hours=estimate,
                  projected_total_hours=elapsed+estimate)
  if elapsed+estimate>=8:
   campaign.update(state='awaiting-duration-confirmation',next_stage_queued=False);return
  campaign.update(state='running-selective',selective_started=True,next_stage_queued=False)
  save(ROOT/'campaign.json',campaign)
  task_run('selective-copying')
  campaign.update(state='complete-awaiting-review',selective_audit_passed=True)
 except BaseException as exc:
  campaign.update(state='stopped',error=str(exc),next_stage_queued=False);raise
 finally:
  campaign.update(updated=now(),elapsed_hours=(time.monotonic()-started)/3600)
  save(ROOT/'campaign.json',campaign);save(HERE/'campaign.json',campaign)
  save(ROOT/'AGENT_STATUS.json',campaign)
  lock.close()

if __name__=='__main__':main()
