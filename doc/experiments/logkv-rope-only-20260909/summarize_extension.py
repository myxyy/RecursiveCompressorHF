"""CPU-only audit and archival of the T131072 retrieval-RoPE extension."""
import csv,hashlib,json,re,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/raid0/RecursiveCompressor/experiments/logkv-rope-only-20260909/extension-131072')

def main():
 m=json.loads((ROOT/'manifest.json').read_text());assert m['state']=='complete-awaiting-review'
 assert len(m['commands'])==2 and all(c['returncode']==0 for c in m['commands'])
 dest=HERE/'extension-131072';dest.mkdir(exist_ok=True)
 expected=sorted(set(range(1,15))|{2**k for k in range(4,18)}|{3*2**(k-1) for k in range(4,17)})
 assert len(expected)==41
 rows=[];overlap={}
 for cp in ('best','final'):
  data=json.loads((ROOT/f'results_{cp}.json').read_text())
  assert data['samples']==256 and data['precision']=='bf16' and data['train_max_t']==2028
  assert sorted(map(int,data['results']))==expected
  logged=re.findall(r'T=\s*(\d+) \| token ([\d.]+) \| string ([\d.]+)',(ROOT/f'{cp}.log').read_text())
  assert len(logged)==41
  for T,tok,st in logged:
   cell=data['results'][T]
   assert tok==format(cell['token_acc'],'.4f') and st==format(cell['string_acc'],'.4f')
  old=json.loads((HERE/'copying/retrieval-rope'/f'results_{cp}.json').read_text())['results']
  overlap[cp]=dict(cells=len(old),exact=all(data['results'][T]==v for T,v in old.items()))
  assert overlap[cp]['exact'],'Overlapping horizons differ; inspect precision/hardware effects'
  for T in expected:
   cell=data['results'][str(T)];assert cell['n']==256
   tok=cell['token_acc']*2560;st=cell['string_acc']*256
   assert abs(tok-round(tok))<1e-8 and abs(st-round(st))<1e-8
   assert 0<=st<=256 and 10*st<=tok<=9*256+st
   rows.append(dict(checkpoint=cp,T=T,token_correct=round(tok),string_correct=round(st),**cell))
  for name in (f'results_{cp}.json',f'plot_{cp}.png',f'{cp}.log'):shutil.copy2(ROOT/name,dest/name)
 shutil.copy2(ROOT/'manifest.json',dest/'manifest.json')
 shutil.copy2(ROOT/'exp/copying/retrieval-rope-no-phase-fixed10-20260909/run_config.json',dest/'run_config.json')
 with (dest/'metrics.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 summary=dict(evaluation_cells=82,new_horizons_per_checkpoint=[T for T in expected if T>8192],
  overlapping_horizons=overlap,weight_sha256=m['checkpoints_sha256'],all_checks_passed=True,
  final_horizon=[r for r in rows if r['T']==131072],
  all_horizons_string_acc_range={cp:[min(r['string_acc'] for r in rows if r['checkpoint']==cp),
   max(r['string_acc'] for r in rows if r['checkpoint']==cp)] for cp in ('best','final')})
 (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
 for cp,style in (('best','--'),('final','-')):
  rs=[r for r in rows if r['checkpoint']==cp]
  for ax,field in zip(axes,('string_acc','token_acc')):
   ax.plot([r['T'] for r in rs],[r[field] for r in rs],style+'o',ms=3,label=cp)
 for ax,title in zip(axes,('Exact string accuracy','Token accuracy')):
  ax.set(title=title,xlabel='T',xscale='log',ylim=(-.02,1.02));ax.axvline(8192,color='gray',ls=':');ax.grid(alpha=.2);ax.legend()
 fig.savefig(dest/'comparison.png',dpi=160);plt.close(fig)
 lines=['# phase2なし・読み出しRoPE Copying：T131072までの追加評価','',
 'GPU 2を1台追加するユーザーの明示承認に基づく評価。再学習なし、固定M=10、各T256例、bf16。',
 'best/finalの重みは前回のCopying評価と同一。T8192までの各33点のtoken/string精度も完全一致した。',
 '16M評価ではなく、Selectiveの学習とは独立して実施した。','',
 '| T | best 完全一致例数 | final 完全一致例数 | best 桁正答率 | final 桁正答率 |',
 '|---:|---:|---:|---:|---:|']
 for T in expected:
  if T<8192:continue
  a,b=[next(r for r in rows if r['checkpoint']==cp and r['T']==T) for cp in ('best','final')]
  lines.append(f"| {T:,} | {a['string_correct']}/256 | {b['string_correct']}/256 | {a['token_acc']*100:.2f}% | {b['token_acc']*100:.2f}% |")
 lines+=['','[全82評価セル](metrics.csv)、[検証結果](summary.json)、[実行記録](manifest.json)。',
  '各41点は同じseedで連続生成した評価であり、T間の記憶列を固定した診断ではない。',
  'bestは訓練中の指標で選択したcheckpoint。長距離の評価値で選び直していない。',
  '', '![全評価点](comparison.png)','']
 (dest/'README.md').write_text('\n'.join(lines))
 files=sorted(f for f in dest.rglob('*') if f.is_file() and f.name!='manifest.sha256')
 (dest/'manifest.sha256').write_text(''.join(f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(dest)}\n' for f in files))
 print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
