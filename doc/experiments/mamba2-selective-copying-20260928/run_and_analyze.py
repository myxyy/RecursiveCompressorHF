"""Run one preflight-approved campaign, then CPU analysis; never retry."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
ARCHIVE = Path(__file__).resolve().parent
ROOT = Path('/mnt/raid0/RecursiveCompressor/experiments/mamba2-selective-copying-20260928-run')


def main():
    status = {'started': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
              'state': 'campaign', 'gpus': [0]}
    path = ROOT / 'supervisor.json'
    if path.exists():
        raise FileExistsError('Supervisor already launched')
    path.write_text(json.dumps(status, indent=2) + '\n')
    try:
        subprocess.run([sys.executable, '-m', 'exp.mamba2_copying.campaign', 'run',
                        '--root', str(ROOT), '--task', 'selective-copying'], cwd=REPO, check=True)
        status['state'] = 'cpu-analysis'
        path.write_text(json.dumps(status, indent=2) + '\n')
        subprocess.run([sys.executable, str(ARCHIVE / 'analyze_completed.py'), '--root', str(ROOT)],
                       cwd=REPO, check=True, timeout=600)
        status['state'] = 'complete'
    except BaseException as exc:
        status.update(state='failed', error=repr(exc))
        raise
    finally:
        status['finished'] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(status, indent=2) + '\n')


if __name__ == '__main__':
    main()
