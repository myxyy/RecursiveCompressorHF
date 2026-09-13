"""Immutable paths for inference-only, <=3 GPU intervention study."""
import datetime, hashlib, json
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
ROOT=Path('/mnt/raid0/RecursiveCompressor/experiments/logkv-readout-intervention-20260910')
SOURCE=Path('/mnt/raid0/RecursiveCompressor/experiments/logkv-phase2-rope-20260908/source')
COMMIT='afaea2b1d8b0bbc6e91aba25e1b9a3594297875b'
BASELINE=Path('/mnt/raid0/RecursiveCompressor/experiments/logkv-rope-only-20260909/exp/copying/retrieval-rope-no-phase-fixed10-20260909')
OLD=REPO/'doc/experiments/logkv-self-slot-20260910/results/diagnostics'
JOBS=[dict(name='best-bf16',checkpoint='best',precision='bf16',gpu=0),
      dict(name='final-bf16',checkpoint='final',precision='bf16',gpu=1),
      dict(name='best-fp32',checkpoint='best',precision='fp32',gpu=2)]

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
