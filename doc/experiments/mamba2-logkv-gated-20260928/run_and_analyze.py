"""Start two preflight-approved tasks on GPU0/1, stop on failure, then CPU audit."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ARCHIVE = Path(__file__).resolve().parent
REPO = ARCHIVE.parents[2]
ROOT = Path('/mnt/raid0/RecursiveCompressor/experiments/mamba2-logkv-gated-20260928-run')


def write(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def main():
    def stop(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    status_path = ROOT / 'supervisor.json'
    if status_path.exists():
        raise FileExistsError('This campaign has already been launched')
    # Validate both campaigns before either starts training.
    from exp.mamba2_logkv_gated.campaign import hashes
    manifest = hashes(REPO)
    for task in ['copying', 'selective-copying']:
        info = json.loads((ROOT / task / 'preflight.json').read_text())
        assert info['source'] == manifest
        assert info['estimated_hours'] < 8
        reserve = info['seconds_per_step']*50000*1.3 + info['evaluation_reserve_seconds']
        assert time.time() + reserve < info['deadline_unix']
    state = dict(state='running', started=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), gpus=[0, 1])
    write(status_path, state)
    children, logs = [], []
    try:
        for gpu, task in enumerate(['copying', 'selective-copying']):
            log = (ROOT / task / 'campaign.log').open('w')
            logs.append(log)
            command = [sys.executable, '-m', 'exp.mamba2_logkv_gated.campaign', 'run',
                       '--task', task, '--gpu', str(gpu), '--root', str(ROOT / task)]
            children.append(subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=True))
        state['worker_pids'] = [p.pid for p in children]
        write(status_path, state)
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError('A task failed; stopping the other task')
            time.sleep(5)
        if any(p.returncode != 0 for p in children):
            raise RuntimeError('A task failed')
        state['state'] = 'cpu-analysis'
        write(status_path, state)
        subprocess.run([sys.executable, str(ARCHIVE / 'analyze_completed.py'), '--root', str(ROOT)],
                       cwd=REPO, check=True, timeout=600)
        state['state'] = 'complete'
    except BaseException as exc:
        state.update(state='failed', error=repr(exc))
        raise
    finally:
        for child in children:
            if child.poll() is None:
                # Campaign handler unwinds execute() and terminates its worker group.
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=30)
        for log in logs:
            log.close()
        state['finished'] = datetime.now(timezone.utc).isoformat()
        write(status_path, state)


if __name__ == '__main__':
    sys.path.insert(0, str(REPO))
    main()
