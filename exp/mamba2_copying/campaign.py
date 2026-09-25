"""One Copying run, followed by best/final evaluation; no automatic extra runs.

Preflight measures the actual 300-step task trajectory. A launch requires its
source hashes to match and a total estimate below eight hours. A subprocess
deadline also caps the complete preflight/training/evaluation campaign at 7.5h.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]


def write(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2)+'\n')
    tmp.replace(path)


def hashes(root):
    paths = [p for folder in ('models','exp') for p in (root/folder).rglob('*.py')]
    paths += [root/'pyproject.toml',root/'uv.lock']
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def environment(root):
    env = dict(os.environ, DATA_DIR=str(root), CUDA_VISIBLE_DEVICES='0',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1',
               TRITON_CACHE_DIR='/mnt/raid0/RecursiveCompressor/cache/triton-mamba2',
               TMPDIR='/mnt/raid0/RecursiveCompressor/tmp')
    return env


def execute(command, log, cwd, env, deadline):
    remaining = deadline-time.time()
    if remaining <= 0:
        raise TimeoutError('Campaign deadline reached')
    with log.open('w') as out:
        child = subprocess.Popen(command,cwd=cwd,env=env,stdout=out,stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            code = child.wait(timeout=remaining)
        except BaseException:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
                child.wait()
            raise
    if code:
        raise RuntimeError(f'Child exited {code}; inspect {log}')


def train_command(name, steps):
    return [sys.executable,'-m','exp.copying.train','--arch','mamba2','--run-name',name,
            '--d-model','512','--num-layers','2','--mamba-d-state','128',
            '--mamba-headdim','64','--mamba-d-conv','4','--mamba-expand','2',
            '--mamba-scan-chunk-size','256','--max-t','2028','--t-dist','loguniform',
            '--steps',str(steps),'--batch-size','64','--grad-accum','1','--lr','0.0003',
            '--warmup','1000','--loss-positions','all','--seed','0','--device','cuda:0',
            '--log-interval','100','--eval-interval','0','--save-interval','10000']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['preflight','run'])
    p.add_argument('--root',type=Path,required=True)
    args=p.parse_args()
    root=args.root.resolve()
    root.mkdir(parents=True,exist_ok=True)
    env=environment(root)
    record=root/'preflight.json'
    if args.mode=='preflight':
        if record.exists() or (root/'exp/copying/preflight').exists():
            raise FileExistsError('Use a fresh campaign root for a new preflight')
        start=time.time()
        info=dict(started=stamp(),started_unix=start,deadline_unix=start+7.5*3600,
                  source=hashes(REPO),base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
        info['packages'] = {name:importlib.metadata.version(name) for name in
                            ['torch','transformers','mamba-ssm','triton','einops','apache-tvm-ffi']}
        upstream = importlib.metadata.distribution('mamba-ssm')
        info['mamba_source'] = json.loads(upstream.read_text('direct_url.json'))
        from models.mamba2.configuration import UPSTREAM_REVISION
        assert info['mamba_source']['vcs_info']['commit_id'] == UPSTREAM_REVISION
        info['upstream_files'] = {
            str(path):hashlib.sha256(upstream.locate_file(path).read_bytes()).hexdigest()
            for path in upstream.files if str(path).endswith('.py')
        }
        command=train_command('preflight',300)
        execute(command,root/'preflight-train.log',REPO,env,info['deadline_unix'])
        records=[json.loads(line) for line in (root/'exp/copying/preflight/train_log.jsonl').read_text().splitlines()]
        seconds_per_step=(records[-1]['elapsed_sec']-records[0]['elapsed_sec'])/200
        # Includes compilation/preflight, 30% training margin and 30 min evaluation reserve.
        info.update(seconds_per_step=seconds_per_step,
                    estimated_hours=(time.time()-start+seconds_per_step*50000*1.3+1800)/3600,
                    preflight_finished=stamp(),command=command)
        write(record,info)
        print(json.dumps(info,indent=2),flush=True)
        return
    info=json.loads(record.read_text())
    if info['estimated_hours']>=8:
        raise RuntimeError('Estimate >=8h: user confirmation is required before launch')
    if hashes(REPO)!=info['source']:
        raise RuntimeError('Sources changed since preflight; revalidate before launch')
    status_path=root/'campaign.json'
    if status_path.exists():
        raise FileExistsError('A campaign has already been launched in this directory')
    snapshot=root/'source'
    for name in info['source']:
        dest=snapshot/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(REPO/name,dest)
    status=dict(state='training',started=stamp(),pid=os.getpid(),gpus=[0],
                preflight=info,train_command=train_command('mamba2',50000))
    write(status_path,status)
    try:
        execute(status['train_command'],root/'train.log',snapshot,env,info['deadline_unix'])
        log=root/'exp/copying/mamba2/train_log.jsonl'
        records=[json.loads(line) for line in log.read_text().splitlines()]
        if len(records)!=500 or records[-1]['step']!=50000:
            raise RuntimeError('Incomplete training log')
        previous=[json.loads(line) for line in (root/'exp/copying/preflight/train_log.jsonl').read_text().splitlines()]
        keys=['step','loss','ema_loss','token_acc','string_acc','lr']
        status['preflight_replay_exact']=all(all(a[k]==b[k] for k in keys) for a,b in zip(previous,records))
        # Report nondeterministic GPU kernel differences; never silently assert bit parity.
        for kind,subdir in [('best','model_best'),('final','model')]:
            status['state']='evaluating-'+kind
            write(status_path,status)
            command=[sys.executable,'-m','exp.mamba2_copying.evaluate',
                     '--model-dir',str(root/'exp/copying/mamba2'/subdir),
                     '--output',str(root/'results'/f'{kind}.json')]
            execute(command,root/f'evaluate-{kind}.log',snapshot,env,info['deadline_unix'])
        # Independently recount every prediction, and check paired samples and bounded cache.
        best=json.loads((root/'results/best.json').read_text())
        final=json.loads((root/'results/final.json').read_text())
        assert len(best['cells'])==len(final['cells'])==41
        for a,b in zip(best['cells'],final['cells']):
            assert a['T']==b['T'] and a['targets']==b['targets']
            for cell in (a,b):
                flags=[[x==y for x,y in zip(t,p)] for t,p in zip(cell['targets'],cell['predictions'])]
                assert cell['token_correct']==sum(sum(row) for row in flags)
                assert cell['string_correct']==sum(all(row) for row in flags)
        assert len({c['state_bytes_per_example'] for r in (best,final) for c in r['cells']})==1
        assert hashes(snapshot)==info['source']
        upstream=importlib.metadata.distribution('mamba-ssm')
        assert all(hashlib.sha256(upstream.locate_file(name).read_bytes()).hexdigest()==digest
                   for name,digest in info['upstream_files'].items())
        status.update(state='complete',finished=stamp(),cells=82,independent_audit=True,
                      elapsed_hours=(time.time()-info['started_unix'])/3600)
    except BaseException as exc:
        status.update(state='failed',finished=stamp(),error=repr(exc))
        raise
    finally:
        write(status_path,status)


if __name__=='__main__':
    main()
