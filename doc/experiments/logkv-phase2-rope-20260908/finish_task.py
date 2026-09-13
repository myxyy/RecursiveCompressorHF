"""Audit and render one completed task. CPU only; no further jobs or git actions."""
import argparse,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('--task',choices=('copying','selective-copying'),required=True)
args=p.parse_args()
subprocess.run([sys.executable,str(HERE/'summarize_stage.py'),'--task',args.task],check=True)
import csv
rows=list(csv.DictReader((HERE/args.task/'metrics.csv').open()))
summary=json.loads((HERE/args.task/'summary.json').read_text())
lines=[f'# {args.task}段階の評価結果','',
 'CPU自動集計。2 GPU以内の継続は承認済み。16M評価・追加アブレーションは開始しない。','',
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
(HERE/args.task/'README.md').write_text('\n'.join(lines))
report=HERE.parents[1]/'logkv-phase2-rope.md'
text=report.read_text()
if args.task=='copying':
 text=text.replace('現在は学習中で、性能結果はまだ未確定。',
  'Copying段階は完了し、保存結果のCPU検証も成功した。Selectiveの状態は継続実行記録を参照。')
 text=text.replace('本評価の結果は完了後に追記する。',
  '[Copyingの全結果・図表](experiments/logkv-phase2-rope-20260908/copying/README.md)を保存した。')
else:
 text+='\nSelective Copying段階も完了し、保存結果のCPU検証が成功した。\n[Selectiveの全結果・図表](experiments/logkv-phase2-rope-20260908/selective-copying/README.md)を保存した。\n'
report.write_text(text)
print(args.task, 'results audited and archived.',flush=True)
