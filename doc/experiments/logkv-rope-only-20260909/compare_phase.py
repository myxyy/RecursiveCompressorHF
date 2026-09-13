"""CPU comparison of matched protocols with/without phase embeddings."""
import argparse,csv,hashlib,json,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'logkv-phase2-rope-20260908'
MODES=('retrieval-rope','compressor-rope')
p=argparse.ArgumentParser();p.add_argument('--task',choices=('copying','selective-copying'),required=True)
args=p.parse_args();dest=HERE/args.task
rows=[];references=[]
for mode in MODES:
 newcfg=json.loads((dest/mode/'run_config.json').read_text())
 oldcfg=json.loads((PREVIOUS/args.task/mode/'run_config.json').read_text())
 assert not newcfg['phase_emb'] and oldcfg['phase_emb']
 exclude={'run_name','num_params','phase_emb'}
 assert {k:v for k,v in newcfg.items() if k not in exclude}=={k:v for k,v in oldcfg.items() if k not in exclude}
 assert oldcfg['num_params']-newcfg['num_params']==8192
 ref=dest/'phase-on-reference'/mode;ref.mkdir(parents=True,exist_ok=True)
 shutil.copy2(PREVIOUS/args.task/mode/'run_config.json',ref/'run_config.json')
 for cp in ('best','final'):
  oldpath=PREVIOUS/args.task/mode/f'results_{cp}.json'
  shutil.copy2(oldpath,ref/oldpath.name)
  references.append(dict(path=str(oldpath),sha256=hashlib.sha256(oldpath.read_bytes()).hexdigest()))
  for phase,path in ((True,oldpath),(False,dest/mode/f'results_{cp}.json')):
   data=json.loads(path.read_text());assert data['samples']==256 and data['precision']=='bf16'
   for T,cell in data['results'].items():rows.append(dict(phase_emb=phase,mode=mode,checkpoint=cp,T=int(T),**cell))
with (dest/'phase_comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
(dest/'phase_references.json').write_text(json.dumps(dict(references=references,
 common_training_flags_match_except_phase=True,common_initial_weights_matched=False),indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
for mode,color in zip(MODES,('tab:orange','tab:green')):
 for phase,style in ((True,'--'),(False,'-')):
  r=sorted((r for r in rows if r['mode']==mode and r['phase_emb']==phase and r['checkpoint']=='final'),key=lambda r:r['T'])
  for ax,metric in zip(axes,('string_acc','token_acc')):
   ax.plot([x['T'] for x in r],[x[metric] for x in r],style+'o',color=color,ms=3,
    label=mode+(' + phase2' if phase else ' only'))
for ax,title in zip(axes,('Final exact string accuracy','Final token accuracy')):
 ax.set(title=title,xlabel='T',xscale='log',ylim=(-.02,1.02));ax.axvline(2028,color='gray',ls=':');ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.savefig(dest/'phase_comparison.png',dpi=160);plt.close(fig)
lines=['','## phase2ありとの比較','',
 '各欄は完全一致率（%）のbest / final。前回のphase2あり結果と同じ学習・評価条件、各256例。',
 'phase2除去で共通重みの初期化も変わるため、同じ初期重みでの因果比較ではない。','',
 '| T | 読み出しRoPE + phase2 | 読み出しRoPEのみ | Compressor RoPE + phase2 | Compressor RoPEのみ |',
 '|---:|---:|---:|---:|---:|']
for T in (16,32,64,128,256,512,1024,2048,4096,8192):
 cells=[]
 for mode in MODES:
  for phase in (True,False):
   cells.append(' / '.join(f"{next(x for x in rows if x['mode']==mode and x['phase_emb']==phase and x['checkpoint']==cp and x['T']==T)['string_acc']*100:.2f}" for cp in ('best','final')))
 lines.append('| '+str(T)+' | '+' | '.join(cells)+' |')
lines+=['','![phase比較](phase_comparison.png)','']
with (dest/'README.md').open('a') as f:f.write('\n'.join(lines))
print('Verified phase comparison:',len(rows),'cells; copied reference results with SHA256')
