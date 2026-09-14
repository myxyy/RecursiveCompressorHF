"""Two-GPU bounded benchmark; does not automatically start the real campaign."""
import json
import os
import signal
import subprocess
import sys
import time
from common import ROOT, HERE, SOURCE, TASKS, GPUS, env, save, now, audit_source, sha

def main():
    manifest=audit_source()
    assert not (ROOT/'preflight.json').exists()
    started=time.time(); children=[]; logs=[]
    record=dict(passed=False,started=now(),gpu_started_unix=started,gpus=list(GPUS),source_manifest=manifest)
    save(ROOT/'preflight-start.json',record)
    try:
        for task,gpu in zip(TASKS,GPUS):
            log=(ROOT/f'benchmark-{task}.log').open('x'); logs.append(log)
            children.append(subprocess.Popen([sys.executable,str(HERE/'benchmark.py'),task],cwd=SOURCE,
                env=env(gpu),stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        while True:
            codes=[p.poll() for p in children]
            if any(c is not None and c!=0 for c in codes): raise RuntimeError(f'Benchmark failure: {codes}')
            if time.time()-started>1200: raise TimeoutError('20-minute benchmark limit')
            if all(c==0 for c in codes): break
            time.sleep(2)
        benches={task:json.loads((ROOT/f'benchmark-{task}.json').read_text()) for task in TASKS}
        assert all(b['passed'] for b in benches.values())
        assert len({b['initial_weights_sha256'] for b in benches.values()})==1
        record.update(passed=True,benchmarks=benches,
            estimated_campaign_hours=(max(b['projected_seconds'] for b in benches.values())+time.time()-started)/3600,
            initial_weights_sha256=benches[TASKS[0]]['initial_weights_sha256'])
        audit_source()
    except BaseException as exc:
        record['error']=str(exc); raise
    finally:
        for p in children:
            try: os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError: pass
        for p in children:
            try: p.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(p.pid,signal.SIGKILL); p.wait()
        for f in logs:f.close()
        record.update(finished=now(),elapsed_seconds=time.time()-started)
        save(ROOT/'preflight.json',record); save(HERE/'preflight.json',record)
    print(json.dumps(record,indent=2),flush=True)

if __name__=='__main__':main()
