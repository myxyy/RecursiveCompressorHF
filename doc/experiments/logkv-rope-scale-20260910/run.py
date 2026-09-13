"""Three independent GPU workers, 7.5h total cap, fail-stop, no retries."""
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from common import HERE, ROOT, SOURCE, MODES, now, save, sha, run_dir

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'campaign.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'campaign.json').exists() or any(run_dir(m).exists() for m in MODES):
        raise FileExistsError('No automatic restart or overwrite')
    preflight=json.loads((HERE/'preflight.json').read_text()); assert preflight['passed']
    assert json.loads((HERE/'evaluation_smoke.json').read_text())['passed']
    manifest=json.loads((ROOT/'source_manifest.json').read_text())
    for f,h in manifest['files'].items(): assert sha(SOURCE/f)==h
    record=dict(state='starting',started=now(),pid=os.getpid(),gpu_limit=3,gpus=[0,1,2],
        estimated_hours='5-6',time_limit_hours=7.5,source_manifest=manifest,
        scripts={p.name:sha(p) for p in HERE.glob('*.py')},next_stage_queued=False)
    started=time.monotonic(); children=[]; logs=[]; audit=None
    def interrupted(sig,frame): raise InterruptedError(f'Signal {sig}')
    signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGINT,interrupted)
    def stop(proc):
        # Group may still contain a training child after its worker exits.
        try: os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError: return
    try:
        for gpu,mode in enumerate(MODES):
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'DATA_DIR':str(ROOT),
                 'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONUNBUFFERED':'1',
                 'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}
            log=(ROOT/f'{mode}.log').open('w'); logs.append(log)
            proc=subprocess.Popen([sys.executable,str(HERE/'worker.py'),'--mode',mode],
                cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(proc)
        record.update(state='running',workers={m:p.pid for m,p in zip(MODES,children)})
        save(ROOT/'campaign.json',record); save(ROOT/'AGENT_STATUS.json',record)
        while True:
            codes=[p.poll() for p in children]
            if any(c is not None and c!=0 for c in codes): raise RuntimeError(f'Worker failure: {codes}')
            if time.monotonic()-started>=7.5*3600: raise TimeoutError('7.5 hour total limit')
            if all(c==0 for c in codes): break
            time.sleep(5)
        record['state']='cpu-audit'; save(ROOT/'campaign.json',record)
        with (ROOT/'summarize.log').open('w') as log:
            audit=subprocess.Popen([sys.executable,str(HERE/'summarize.py')],cwd=HERE,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            code=audit.wait(timeout=max(1,7.5*3600-(time.monotonic()-started)))
            if code: raise RuntimeError(f'Audit failed: {code}')
        for f,h in manifest['files'].items(): assert sha(SOURCE/f)==h
        for f,h in record['scripts'].items(): assert sha(HERE/f)==h
        record['state']='complete-awaiting-review'
    except BaseException as exc:
        record.update(state='stopped',error=str(exc)); raise
    finally:
        for p in children+([audit] if audit else []): stop(p)
        for p in children+([audit] if audit else []):
            try: p.wait(timeout=15)
            except subprocess.TimeoutExpired: pass
            try: os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError: pass
        for log in logs: log.close()
        record.update(finished=now(),elapsed_hours=(time.monotonic()-started)/3600)
        save(ROOT/'campaign.json',record); save(ROOT/'AGENT_STATUS.json',record); save(HERE/'campaign.json',record)
        lock.close()
if __name__=='__main__': main()
