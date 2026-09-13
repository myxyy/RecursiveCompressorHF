"""Fixed-M10 Copying: end-anchored global RoPE versus local retrieval RoPE."""
import datetime
import hashlib
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROOT = Path('/mnt/raid0/RecursiveCompressor/experiments/logkv-aligned-rope-20260913')
SOURCE = ROOT/'source'
MODES = ('local-control','aligned')
OLD = REPO/'doc/experiments/logkv-rope-only-20260909/copying/retrieval-rope.json'
OLD_CONFIG = OLD.parent/'retrieval-rope/run_config.json'
PAIRED_TS = sorted(set([3,16,8192,131071,131072]+list(range(49144,49161))+
    [v+d for v in [16384,32768,65536,98304] for d in [-1,0,1]]))
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def name(mode): return f'{mode}-fixed10-20260913'
def run_dir(mode): return ROOT/'exp/copying'/name(mode)
def command(mode,python):
    cmd=json.loads(OLD.read_text())['commands'][0]['command'].copy()
    cmd[0]=python;cmd[1]=str(HERE/'train_entry.py')
    cmd[cmd.index('--run-name')+1]=name(mode)
    if mode=='aligned':
        cmd.remove('--retrieval-rope');cmd.extend(['--aligned-rope','--aligned-rope-scale','1.0'])
    return cmd
