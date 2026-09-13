"""Finish CPU-only auditing after the already-started Copying stage.

Never launches GPU jobs or another task, and never merges or commits.
"""
import datetime,json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/raid0/RecursiveCompressor/experiments/logkv-phase2-rope-20260908')
while True:
 stage=json.loads((ROOT/'copying/stage.json').read_text())
 if stage['state']!='running':break
 time.sleep(55)
if stage['state']!='complete-awaiting-review':
 raise SystemExit('Copying stage stopped; manual review required. No next stage started.')
subprocess.run([sys.executable,str(HERE/'summarize_stage.py'),'--task','copying'],check=True)
import csv
rows=list(csv.DictReader((HERE/'copying/metrics.csv').open()))
summary=json.loads((HERE/'copying/summary.json').read_text())
lines=['# Copying段階の評価結果','',
 'CPU自動集計。Selective Copying・16M評価は開始していない。解釈と次段階はユーザーとの確認後に進める。','',
 '[条件と実装](../../../logkv-phase2-rope.md)。各方式は固定M=10、50,000 steps、各T256例、1学習seed。',
 '全33評価点は[metrics.csv](metrics.csv)、実行情報と検証結果は[summary.json](summary.json)。','',
 '| 方式 | best step | 学習時間（分） | 最初に完全一致率100%を下回る評価T best / final |',
 '|---|---:|---:|---|']
for r in summary['runs']:
 mode=r['mode'];first=summary['first_imperfect']
 def cell(cp):
  v=first[f'{mode}/{cp}'];return '全評価点100%' if v is None else str(v)
 lines.append(f"| {mode} | {r['best_step']} | {r['train_elapsed_sec']/60:.2f} | {cell('best')} / {cell('final')} |")
lines+=['','| 方式 | checkpoint | T64 | T1024 | T2048 | T4096 | T8192 |',
 '|---|---|---:|---:|---:|---:|---:|']
for mode in ('phase2','retrieval-rope','compressor-rope'):
 for cp in ('best','final'):
  cells=[]
  for T in (64,1024,2048,4096,8192):
   r=next(r for r in rows if r['mode']==mode and r['checkpoint']==cp and int(r['T'])==T)
   cells.append(r['string_correct']+'/256')
  lines.append('| '+' | '.join([mode,cp,*cells])+' |')
lines+=['','数値は10桁すべて完全一致した例数。評価点間のすべてのTでの成功や16M保持を保証しない。',
 '', '![評価](comparison.png)', '', '![学習曲線](learning.png)', '']
(HERE/'copying/README.md').write_text('\n'.join(lines))
report=HERE.parents[1]/'logkv-phase2-rope.md'
body=report.read_text().replace('現在は学習中で、性能結果はまだ未確定。',
 'Copying段階は完了し、保存結果のCPU検証も成功した。Selectiveは続行確認待ち。')
body=body.replace('本評価の結果は完了後に追記する。',
 '[Copyingの全結果・図表](experiments/logkv-phase2-rope-20260908/copying/README.md)を保存した。')
report.write_text(body)
status=json.loads((ROOT/'AGENT_STATUS.json').read_text())
status.update(status='copying-complete-awaiting-review',finished=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              gpu_work_completed=True,selective_started=False,next_stage_queued=False,
              results=str(HERE/'copying/README.md'))
(ROOT/'AGENT_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
print('Copying results audited and archived. No further stage started.',flush=True)
