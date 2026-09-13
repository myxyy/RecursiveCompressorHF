"""One training run, standard evaluation and bounded diagnosis on GPU 0."""
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from common import HERE, ROOT, SOURCE, SOURCE_COMMIT, OLD, BASELINE, RUN, NAME, now, save, sha


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'campaign.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'campaign.json').exists() or RUN.exists():raise FileExistsError('No automatic restart/overwrite')
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()==SOURCE_COMMIT
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=SOURCE,text=True)
    preflight=json.loads((HERE/'preflight.json').read_text())
    assert preflight['initial_weights_identical'] and preflight['no_self_reference_gradients_streaming_passed']
    assert json.loads((ROOT/'diagnostic_preflight.json').read_text())['instrumentation_preserves_logits']
    assert json.loads((ROOT/'gpu_smoke.json').read_text())['bf16_instrumentation_output_identical']
    assert json.loads((ROOT/'baseline_replay_smoke.json').read_text())['matches_archived_baseline']
    source_record=json.loads((OLD/'copying/retrieval-rope.json').read_text())
    cmd=list(source_record['commands'][0]['command']);cmd[0]=sys.executable
    cmd[cmd.index('--run-name')+1]=NAME;cmd.remove('--self-slot')
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','DATA_DIR':str(ROOT),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
         'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}
    started=time.monotonic();deadline=started+7.5*3600
    record=dict(state='starting',started=now(),pid=os.getpid(),gpu=0,gpu_limit=1,time_limit_hours=7.5,
       estimated_hours='5-6',task='copying',source=str(SOURCE),source_commit=SOURCE_COMMIT,
       experiment_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
       script_sha256={p.name:sha(p) for p in HERE.glob('*.py')},
       baseline_weights={cp:sha(BASELINE/sub/'model.safetensors') for cp,sub in [('best','model_best'),('final','model')]},
       commands=[],next_stage_queued=True,scope='One no-self training run; T131072 eval; paired/precision/attention diagnosis; no Selective or 16M')
    child=None
    def interrupted(signum,frame):
        if child is not None and child.poll() is None:child.terminate()
        raise InterruptedError(f'Signal {signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def execute(label,command,cwd):
        nonlocal child
        if time.monotonic()>=deadline:raise TimeoutError('7.5-hour campaign limit')
        entry=dict(stage=label,command=command,started=now(),returncode=None)
        record['state']=label;record['commands'].append(entry);save(ROOT/'campaign.json',record)
        with (ROOT/f'{label}.log').open('w') as log:
            child=subprocess.Popen(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT)
            try:code=child.wait(timeout=max(.01,deadline-time.monotonic()))
            except BaseException:
                child.terminate()
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                raise
        entry.update(returncode=code,finished=now());save(ROOT/'campaign.json',record)
        if code:raise RuntimeError(f'{label} failed: {code}')
    try:
        execute('train',cmd,SOURCE)
        cfg=json.loads((RUN/'run_config.json').read_text());old=json.loads((BASELINE/'run_config.json').read_text())
        assert cfg['self_slot'] is False and old['self_slot'] is True
        assert {k:v for k,v in cfg.items() if k not in ('run_name','self_slot')}=={k:v for k,v in old.items() if k not in ('run_name','self_slot')}
        for cp in ('best','final'):
            execute(cp,[sys.executable,str(HERE/'evaluate_detail.py'),'--checkpoint',cp],SOURCE)
            for filename in ('results.json','plot.png'):
                p=RUN/filename;shutil.copy2(p,ROOT/f'{p.stem}_{cp}{p.suffix}')
        execute('diagnose',[sys.executable,str(HERE/'diagnose.py')],SOURCE)
        for cp,sub in [('best','model_best'),('final','model')]:assert sha(BASELINE/sub/'model.safetensors')==record['baseline_weights'][cp]
        assert all(sha(HERE/name)==h for name,h in record['script_sha256'].items())
        record.update(state='gpu-complete',next_stage_queued=False);save(ROOT/'campaign.json',record)
        execute('summarize',[sys.executable,str(HERE/'summarize.py')],HERE)
        record.update(state='complete-awaiting-review',next_stage_queued=False)
    except BaseException as exc:
        record.update(state='stopped',error=str(exc),next_stage_queued=False)
        raise
    finally:
        record.update(finished=now(),elapsed_hours=(time.monotonic()-started)/3600)
        save(ROOT/'campaign.json',record);save(ROOT/'AGENT_STATUS.json',record)
        save(HERE/'campaign.json',record)
        if (HERE/'results').exists():save(HERE/'results/campaign.json',record)
        lock.close()

if __name__=='__main__':main()
