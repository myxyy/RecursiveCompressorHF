"""Three authorized inference workers; two-hour hard cap, no retries."""
import fcntl,json,os,signal,subprocess,sys,time
from common import HERE,ROOT,SOURCE,COMMIT,JOBS,save,sha,now


def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 lock=(ROOT/'campaign.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (ROOT/'campaign.json').exists():raise FileExistsError('Campaign exists; no automatic restart')
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()==COMMIT
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=SOURCE,text=True)
 for device in ('cpu','cuda'):
  x=json.loads((HERE/f'preflight-{device}.json').read_text());assert x['sham_patches_identity'] and x['score_interventions_independent_oracle']
 started=time.monotonic();deadline=started+7200;processes=[]
 record=dict(state='running',started=now(),pid=os.getpid(),gpu_limit=3,gpus=[0,1,2],time_limit_hours=2,
       authorization='User explicitly authorized up to three parallel GPUs for this verification',
       source_commit=COMMIT,experiment_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
       script_sha256={p.name:sha(p) for p in HERE.glob('*.py')},jobs=[],next_stage_queued=False)
 def stop(signum,frame):raise InterruptedError(f'Signal {signum}')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 try:
  for job in JOBS:
   env={**os.environ,'CUDA_VISIBLE_DEVICES':str(job['gpu']),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
        'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}
   cmd=[sys.executable,str(HERE/'worker.py'),'--job',job['name']]
   log=(ROOT/f"{job['name']}.log").open('w')
   proc=subprocess.Popen(cmd,cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT)
   processes.append((proc,log,job))
   record['jobs'].append(dict(**job,pid=proc.pid,command=cmd,returncode=None))
  save(ROOT/'campaign.json',record)
  while True:
   states=[proc.poll() for proc,_,_ in processes]
   for entry,code in zip(record['jobs'],states):entry['returncode']=code
   save(ROOT/'campaign.json',record)
   if any(s not in (None,0) for s in states):raise RuntimeError(f'Worker failure: {states}')
   if all(s==0 for s in states):break
   if time.monotonic()>=deadline:raise TimeoutError('Two-hour total limit')
   time.sleep(2)
  record.update(state='gpu-complete',gpu_finished=now());save(ROOT/'campaign.json',record)
  assert all(sha(HERE/name)==h for name,h in record['script_sha256'].items())
  with (ROOT/'summarize.log').open('w') as log:
   result=subprocess.run([sys.executable,str(HERE/'summarize.py')],cwd=HERE,stdout=log,stderr=subprocess.STDOUT,
                         timeout=max(.01,deadline-time.monotonic()))
  record['summary_returncode']=result.returncode
  if result.returncode:raise RuntimeError('CPU audit failed')
  record['state']='complete-awaiting-review'
 except BaseException as exc:
  record.update(state='stopped',error=str(exc));raise
 finally:
  for proc,log,job in processes:
   if proc.poll() is None:
    proc.terminate()
    try:proc.wait(timeout=20)
    except subprocess.TimeoutExpired:proc.kill();proc.wait()
   log.close()
  record.update(finished=now(),elapsed_seconds=time.monotonic()-started)
  save(ROOT/'campaign.json',record);save(ROOT/'AGENT_STATUS.json',record);save(HERE/'campaign.json',record)
  if (HERE/'results').exists():save(HERE/'results/campaign.json',record)
  lock.close()

if __name__=='__main__':main()
