"""User-authorized Copying -> Selective continuation, at most two GPUs.

Wait for ALL Copying workers and audit them, then check the duration budget
before starting the already-planned Selective stage. No other GPU work.
"""
import datetime,fcntl,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/raid0/RecursiveCompressor/experiments/logkv-phase2-rope-20260908')

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path,x):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(path)
def read(path):return json.loads(path.read_text())
def task_duration_hours(mode):
 r=read(ROOT/'copying'/f'{mode}.json')
 return (datetime.datetime.fromisoformat(r['finished'])-datetime.datetime.fromisoformat(r['started'])).total_seconds()/3600

def main():
 lock=(ROOT/'continuation.lock').open('w')
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 path=ROOT/'continuation.json'
 state=dict(state='waiting-for-copying',started=now(),pid=os.getpid(),gpu_limit=2,
  authorization='2026-09-08 user: continuous experiments permitted at up to 2 GPUs',
  estimate_rule='1.25 * observed Copying two-GPU makespan + 0.25 hours',
  total_initial_estimate_hours='12-14',selective_started=False,next_stage_queued=True)
 save(path,state)
 try:
  while True:
   copying=read(ROOT/'copying/stage.json')
   if copying['state']!='running':break
   time.sleep(55)
  if copying['state']!='complete-awaiting-review':raise RuntimeError('Copying did not complete')
  with (ROOT/'finish-copying.log').open('w') as log:
   subprocess.run([sys.executable,str(HERE/'finish_task.py'),'--task','copying'],stdout=log,stderr=subprocess.STDOUT,check=True)
  observed=max(task_duration_hours('phase2')+task_duration_hours('compressor-rope'),task_duration_hours('retrieval-rope'))
  estimate=observed*1.25+.25
  state.update(copying_audit_passed=True,copying_observed_hours=observed,selective_estimate_hours=estimate)
  save(path,state)
  if estimate>=8:
   state.update(state='awaiting-duration-confirmation',next_stage_queued=False)
   return
  if (ROOT/'selective-copying').exists():raise FileExistsError('Selective stage already exists; no duplicate launch')
  state.update(state='running-selective',selective_started=True,next_stage_queued=False,selective_started_at=now())
  save(path,state)
  with (ROOT/'stage-selective.log').open('w') as log:
   subprocess.run([sys.executable,str(HERE/'run_stage.py'),'--task','selective-copying'],stdout=log,stderr=subprocess.STDOUT,check=True)
  with (ROOT/'finish-selective.log').open('w') as log:
   subprocess.run([sys.executable,str(HERE/'finish_task.py'),'--task','selective-copying'],stdout=log,stderr=subprocess.STDOUT,check=True)
  state.update(state='complete-awaiting-review',selective_audit_passed=True)
 except Exception as exc:
  state.update(state='stopped',error=str(exc),next_stage_queued=False)
  raise
 finally:
  state['updated']=now();save(path,state)
  status=read(ROOT/'AGENT_STATUS.json')
  status.update(status=state['state'],selective_started=state['selective_started'],next_stage_queued=False,
                continuation_manifest=str(path),updated=now())
  save(ROOT/'AGENT_STATUS.json',status)
  save(HERE/'continuation.json',state)
  lock.close()

if __name__=='__main__':main()
