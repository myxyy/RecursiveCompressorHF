"""Two independent GPU workers, 7.5h total cap, fail-stop, no retries."""
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from common import HERE, ROOT, SOURCE, INITIAL, BASE_INITIAL, GPUS, NUM_LAYERS, MODES, now, save, sha, run_dir

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'campaign.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'campaign.json').exists() or any(run_dir(m).exists() for m in MODES):
        raise FileExistsError('No automatic restart or overwrite')
    preflight=json.loads((HERE/'preflight.json').read_text()); assert preflight['passed']
    assert preflight['tasks']==list(MODES)
    assert preflight['initial_weights_sha256']==sha(INITIAL)
    assert preflight['estimated_campaign_hours'] < 7.5
    assert preflight['repeat_training_bitexact']
    assert json.loads((HERE/'validation.json').read_text())['passed']
    assert json.loads((HERE/'evaluation_smoke.json').read_text())['passed']
    assert json.loads((HERE/'coefficient_smoke.json').read_text())['passed']
    manifest=json.loads((ROOT/'source_manifest.json').read_text())
    for f,h in manifest['files'].items(): assert sha(SOURCE/f)==h
    record=dict(tasks=list(MODES),state='starting',started=now(),pid=os.getpid(),gpu_limit=2,gpus=list(GPUS),num_layers=NUM_LAYERS, conv_kernel_size=4, learnable_decay=True,
        estimated_hours=preflight['estimated_campaign_hours'],time_limit_hours=7.5,source_manifest=manifest,source_root=str(SOURCE),
        scripts={p.name:sha(p) for p in HERE.glob('*.py')},
        prerequisite_hashes={f:sha(HERE/f) for f in ['preflight.json','validation.json','evaluation_smoke.json','coefficient_smoke.json']},
        initial_weights_sha256=sha(INITIAL),base_initial_sha256=sha(BASE_INITIAL),next_stage_queued=False)
    started=time.monotonic()
    remaining=preflight['gpu_started_unix']+7.5*3600-time.time()
    if remaining <= 0: raise TimeoutError('Whole GPU batch budget already exhausted')
    estimated_work=preflight['estimated_campaign_hours']-preflight['gpu_preflight_hours']
    if estimated_work*3600 > remaining:
        raise TimeoutError('Projected work exceeds remaining whole GPU batch budget; obtain duration confirmation')
    deadline=started+remaining
    record.update(gpu_batch_started=preflight['gpu_started'],remaining_hours_at_launch=remaining/3600)
    children=[]; logs=[]; audit=None
    def interrupted(sig,frame): raise InterruptedError(f'Signal {sig}')
    signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGINT,interrupted)
    def stop(proc):
        # Group may still contain a training child after its worker exits.
        try: os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError: return
    try:
        for gpu,mode in zip(GPUS,MODES):
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'DATA_DIR':str(ROOT),'LOGKV_TASK':mode,
                 'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONUNBUFFERED':'1',
                 'CUBLAS_WORKSPACE_CONFIG':':4096:8',
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
            if time.monotonic()>=deadline: raise TimeoutError('7.5 hour total GPU batch limit including preflight')
            if all(c==0 for c in codes): break
            time.sleep(5)
        record['state']='cpu-audit'; save(ROOT/'campaign.json',record)
        with (ROOT/'summarize.log').open('w') as log:
            audit=subprocess.Popen([sys.executable,str(HERE/'summarize.py')],cwd=HERE,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            code=audit.wait(timeout=max(1,deadline-time.monotonic()))
            if code: raise RuntimeError(f'Audit failed: {code}')
        for f,h in manifest['files'].items(): assert sha(SOURCE/f)==h
        for f,h in record['scripts'].items(): assert sha(HERE/f)==h
        for f,h in record['prerequisite_hashes'].items(): assert sha(HERE/f)==h
        assert sha(INITIAL)==record['initial_weights_sha256']
        assert sha(BASE_INITIAL)==record['base_initial_sha256']
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
        record.update(finished=now(),elapsed_hours=(time.monotonic()-started)/3600,
            whole_gpu_batch_hours=(time.time()-preflight['gpu_started_unix'])/3600)
        save(ROOT/'campaign.json',record); save(ROOT/'AGENT_STATUS.json',record); save(HERE/'campaign.json',record)
        lock.close()
if __name__=='__main__': main()
