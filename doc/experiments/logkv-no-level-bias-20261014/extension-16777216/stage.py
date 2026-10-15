"""Supervise a parallel best/final evaluation stage, fail-stop without retry."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import ROOT, SOURCE, save, sha, now


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','full']);args=p.parse_args()
    out=ROOT/'extension-16777216';out.mkdir(exist_ok=True)
    record=dict(stage=args.stage,started=now(),pid=os.getpid(),gpus=[0,1],commands=[],state='running')
    assert not (out/(args.stage+'-stage.json')).exists()
    timeout=300
    if args.stage=='full':
        pre=json.loads((out/'preflight-stage.json').read_text());assert pre['state']=='complete'
        payloads=[json.loads((out/cp/'preflight.json').read_text()) for cp in ['best','final']]
        assert all(x['passed'] for x in payloads)
        estimate=max(x['projected_full_seconds_with_margin'] for x in payloads)
        assert estimate + pre['elapsed_seconds'] < 8*3600, 'Duration confirmation required'
        timeout=max(1800, int(estimate*2))
        record.update(projected_full_seconds_with_margin=estimate,
            preflight_hashes={cp:sha(out/cp/'preflight.json') for cp in ['best','final']})
    record.update(timeout_seconds=timeout,scripts={f.name:sha(f) for f in HERE.glob('*.py')})
    processes=[];logs=[];started=time.monotonic()
    def interrupt(sig,frame):raise InterruptedError(f'signal {sig}')
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        for gpu,cp in enumerate(['best','final']):
            dest=out/cp;dest.mkdir(exist_ok=True)
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'}
            cmd=[sys.executable,str(HERE/'evaluate.py'),args.stage,'--checkpoint',cp]
            log=(dest/(args.stage+'.log')).open('x');logs.append(log)
            proc=subprocess.Popen(cmd,cwd=SOURCE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            processes.append(proc);record['commands'].append(dict(checkpoint=cp,gpu=gpu,pid=proc.pid,command=cmd))
        save(out/(args.stage+'-stage.json'),record)
        while True:
            codes=[p.poll() for p in processes]
            if any(c is not None and c!=0 for c in codes):raise RuntimeError(f'Worker failed: {codes}')
            if all(c==0 for c in codes):break
            if time.monotonic()-started>timeout:raise TimeoutError('Stage deadline exceeded')
            time.sleep(2)
        record['state']='complete'
    except BaseException as exc:
        record.update(state='failed',error=str(exc));raise
    finally:
        for proc in processes:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        for log in logs:log.close()
        record.update(completed=now(),elapsed_seconds=time.monotonic()-started,returncodes=[p.returncode for p in processes])
        save(out/(args.stage+'-stage.json'),record)
        save(HERE/(args.stage+'-stage.json'),record)


if __name__=='__main__':main()
