"""Finite, fail-stop campaign: 6GPU training -> 1GPU evaluation -> CPU audit."""
import datetime
import fcntl
import json
import os
import signal
import subprocess
import time
from common import HERE, ROOT, DATA, SOURCE, RUN_NAME, PYTHON, command, environment, save, sha


def main():
    lock=(ROOT/'campaign.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'campaign.json').exists(), 'Campaign already started'
    preflight=json.loads((HERE/'preflight.json').read_text())
    smoke=json.loads((ROOT/'evaluation_smoke/review.json').read_text())
    assert json.loads((ROOT/'attention_smoke/attention_valid.json').read_text())['passed']
    assert preflight['passed'] and preflight['estimated_total_hours']<8 and smoke['passed']
    manifest=json.loads((HERE/'source_manifest.json').read_text())
    for name,expected in manifest['files'].items():assert sha(SOURCE/name)==expected
    assert not (DATA/'checkpoints_logkv'/RUN_NAME).exists()
    start=time.time();preflight_start=datetime.datetime.fromisoformat(json.loads((HERE/'benchmark_launch.json').read_text())['started']).timestamp()
    deadline=preflight_start+7.5*3600
    state=dict(state='running',started=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        pid=os.getpid(),gpu_limit=6,training_command=command(RUN_NAME,5000),steps=5000,
        hard_deadline=datetime.datetime.fromtimestamp(deadline,datetime.timezone.utc).isoformat(),
        next_campaign_queued=False,source_manifest=manifest,
        scripts={p.name:sha(p) for p in HERE.glob('*.py')},
        prerequisite_hashes={name:sha(HERE/name) for name in ['preflight.json','source_manifest.json','benchmark_launch.json','initialization_audit.json']},stages=[])
    def update():save(ROOT/'campaign.json',state);save(HERE/'campaign.json',state)
    update()
    def stage(name,cmd,env,max_seconds):
        entry=dict(stage=name,command=cmd,started=datetime.datetime.now(datetime.timezone.utc).isoformat());state['stages'].append(entry);update()
        with (ROOT/f'{name}.log').open('w') as log:
            child=subprocess.Popen(cmd,cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            entry['pid']=child.pid;update()
            try:
                rc=child.wait(timeout=max(1,min(max_seconds,deadline-time.time())))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                entry['timeout']=True;raise RuntimeError(f'{name}: time limit')
        entry['returncode']=rc;entry['finished']=datetime.datetime.now(datetime.timezone.utc).isoformat();update()
        if rc:raise RuntimeError(f'{name}: exit {rc}')
    try:
        stage('train',command(RUN_NAME,5000),environment(),7*3600)
        env=environment();env['CUDA_VISIBLE_DEVICES']='0'
        stage('evaluate',[str(PYTHON),str(HERE/'evaluate.py')],env,3600)
        stage('attention',[str(PYTHON),str(HERE/'attention_valid.py')],env,300)
        env['CUDA_VISIBLE_DEVICES']=''
        stage('summarize',[str(PYTHON),str(HERE/'summarize.py')],env,600)
        state['state']='complete-awaiting-review'
        for name,expected in state['scripts'].items():assert sha(HERE/name)==expected
        for name,expected in manifest['files'].items():assert sha(SOURCE/name)==expected
    except Exception as e:
        state['state']='failed';state['error']=str(e)
    finally:
        state['finished']=datetime.datetime.now(datetime.timezone.utc).isoformat();state['elapsed_hours']=(time.time()-start)/3600;update()


if __name__=='__main__':main()
