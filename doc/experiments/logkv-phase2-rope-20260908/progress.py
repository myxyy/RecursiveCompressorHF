"""Read-only, CPU-only status snapshot; does not launch or resume experiments."""
import datetime,json
from pathlib import Path
ROOT=Path('/mnt/raid0/RecursiveCompressor/experiments/logkv-phase2-rope-20260908')
stage=json.loads((ROOT/'copying/stage.json').read_text())
out=dict(stage=stage['state'],elapsed_hours=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(stage['started'])).total_seconds()/3600,runs=[])
for mode in ('phase2','retrieval-rope','compressor-rope'):
 path=ROOT/'copying'/f'{mode}.json'
 if not path.exists():out['runs'].append(dict(mode=mode,state='queued in current Copying stage'));continue
 m=json.loads(path.read_text());row=dict(mode=mode,state=m['state'])
 log=ROOT/'exp/copying'/f'{mode}-fixed10-20260908'/'train_log.jsonl'
 if log.exists():
  records=[]
  for line in log.read_text().splitlines():
   try:r=json.loads(line)
   except json.JSONDecodeError:continue
   if 'loss' in r:records.append(r)
  if records:
   r=records[-1];row.update({k:r[k] for k in ('step','ema_loss','string_acc','elapsed_sec')})
   row['train_eta_hours']=(50000-r['step'])*r['elapsed_sec']/r['step']/3600
 out['runs'].append(row)
print(json.dumps(out,ensure_ascii=False))
