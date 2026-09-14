"""Two independent tasks, <=2 GPUs, 7.5h from preflight, no queued next campaign."""
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from common import ROOT, HERE, SOURCE, TASKS, GPUS, env, save, now, audit_source, sha

def main():
    lock=(ROOT/'campaign.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'campaign.json').exists() or (ROOT/'runs').exists():
        raise FileExistsError('No automatic overwrite or restart')
    pre=json.loads((ROOT/'preflight.json').read_text())
    assert pre['passed'] and pre['estimated_campaign_hours']<7.5
    assert json.loads((HERE/'evaluation_smoke.json').read_text())['passed']
    deadline=pre['gpu_started_unix']+7.5*3600
    assert max(b['projected_seconds'] for b in pre['benchmarks'].values()) < deadline-time.time()
    manifest=audit_source()
    scripts={p.name:sha(p) for p in HERE.glob('*.py')}
    record=dict(state='starting',started=now(),pid=os.getpid(),tasks=list(TASKS),gpus=list(GPUS),
        source_manifest=manifest,scripts=scripts,preflight_sha256=sha(ROOT/'preflight.json'),
        evaluation_smoke_sha256=sha(HERE/'evaluation_smoke.json'),
        projected_hours=pre['estimated_campaign_hours'],deadline_unix=deadline,
        whole_batch_limit_hours=7.5,next_stage_queued=False)
    save(ROOT/'campaign.json',record)
    children=[]; logs=[]; audit=None
    def interrupted(sig,frame): raise InterruptedError(f'Signal {sig}')
    signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGINT,interrupted)
    try:
        for task,gpu in zip(TASKS,GPUS):
            log=(ROOT/f'{task}.log').open('x'); logs.append(log)
            children.append(subprocess.Popen([sys.executable,str(HERE/'worker.py'),task],cwd=SOURCE,
                env=env(gpu),stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        record.update(state='running',workers={t:p.pid for t,p in zip(TASKS,children)})
        save(ROOT/'campaign.json',record)
        while True:
            codes=[p.poll() for p in children]
            if any(c is not None and c!=0 for c in codes): raise RuntimeError(f'Worker failed: {codes}')
            if time.time()>=deadline: raise TimeoutError('7.5h whole-batch limit')
            if all(c==0 for c in codes): break
            time.sleep(5)
        record['state']='cpu-audit';save(ROOT/'campaign.json',record)
        with (ROOT/'summary.log').open('w') as log:
            audit=subprocess.Popen([sys.executable,str(HERE/'summarize.py')],cwd=SOURCE,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            code=audit.wait(timeout=max(1,deadline-time.time()))
            if code: raise RuntimeError(f'CPU audit failed: {code}')
        audit_source()
        for name,digest in scripts.items(): assert sha(HERE/name)==digest
        assert sha(ROOT/'preflight.json')==record['preflight_sha256']
        assert sha(HERE/'evaluation_smoke.json')==record['evaluation_smoke_sha256']
        record['state']='complete-awaiting-review'
    except BaseException as exc:
        record.update(state='stopped',error=str(exc));raise
    finally:
        for p in children+([audit] if audit else []):
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        for p in children+([audit] if audit else []):
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        for f in logs:f.close()
        record.update(finished=now(),whole_batch_hours=(time.time()-pre['gpu_started_unix'])/3600)
        save(ROOT/'campaign.json',record); save(HERE/'campaign.json',record)

if __name__=='__main__':main()
