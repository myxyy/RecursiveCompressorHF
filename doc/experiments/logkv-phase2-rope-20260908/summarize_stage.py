"""CPU-only audit and archive of a completed single-task stage."""
import argparse,csv,hashlib,json,math,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/raid0/RecursiveCompressor/experiments/logkv-phase2-rope-20260908')
MODES=('phase2','retrieval-rope','compressor-rope')

def main():
 p=argparse.ArgumentParser();p.add_argument('--task',choices=('copying','selective-copying'),required=True)
 args=p.parse_args();source=ROOT/args.task
 stage=json.loads((source/'stage.json').read_text())
 assert stage['state']=='complete-awaiting-review'
 dest=HERE/args.task;dest.mkdir(exist_ok=True)
 shutil.copy2(source/'stage.json',dest/'stage.json')
 rows=[];runs=[];curves={};configs=[]
 expected=sorted(set(range(1,15))|{2**k for k in range(4,14)}|{3*2**(k-1) for k in range(4,13)})
 assert len(expected)==33
 for mode in MODES:
  manifest=json.loads((source/f'{mode}.json').read_text())
  assert manifest['state']=='complete' and len(manifest['commands'])==3
  assert all(c['returncode']==0 for c in manifest['commands'])
  assert manifest['source_commit']==stage['commit']
  shutil.copy2(source/f'{mode}.json',dest/f'{mode}.json')
  folder=ROOT/'exp'/args.task/f'{mode}-fixed10-20260908';out=dest/mode;out.mkdir(exist_ok=True)
  for c in ('train','best','final'):shutil.copy2(source/f'{mode}-{c}.log',out/f'{c}.log')
  for name in ('run_config.json','train_log.jsonl','best.json','results_best.json','results_final.json','plot_best.png','plot_final.png'):
   shutil.copy2(folder/name,out/name)
  cfg=json.loads((folder/'run_config.json').read_text());configs.append(cfg)
  assert cfg['phase_emb'] and cfg['phase_levels']==2 and cfg['steps']==50000
  assert cfg['retrieval_rope']==(mode=='retrieval-rope')
  assert cfg['compressor_rope']==(mode=='compressor-rope')
  assert cfg['level_decay_scale']==1 and not cfg['relative_position_kv'] and not cfg['compressor_position_transform']
  records=[json.loads(line) for line in (folder/'train_log.jsonl').read_text().splitlines()]
  train=[r for r in records if 'loss' in r];quick=[r for r in records if 'quick_eval' in r]
  assert [r['step'] for r in train]==list(range(100,50001,100))
  assert len(quick)==10
  assert all(math.isfinite(r['loss']) and math.isfinite(r['ema_loss']) for r in train)
  best=json.loads((folder/'best.json').read_text())
  chosen=max(train,key=lambda r:(r['string_acc'],r['token_acc'],-r['ema_loss']))
  assert best['step']==chosen['step']
  curves[mode]=train
  hashes={}
  for cp,sub in (('best','model_best'),('final','model')):
   hashes[cp]=hashlib.sha256((folder/sub/'model.safetensors').read_bytes()).hexdigest()
   shutil.copy2(folder/sub/'config.json',out/f'config_{cp}.json')
   result=json.loads((folder/f'results_{cp}.json').read_text())
   assert result['samples']==256 and result['precision']=='bf16' and result['train_max_t']==2028
   assert sorted(map(int,result['results']))==expected
   for T in expected:
    cell=result['results'][str(T)];assert cell['n']==256
    tok=cell['token_acc']*2560;st=cell['string_acc']*256
    assert abs(tok-round(tok))<1e-8 and abs(st-round(st))<1e-8
    assert 0<=st<=256 and 0<=tok<=2560
    assert st*10<=tok and tok<=9*256+st
    rows.append(dict(mode=mode,checkpoint=cp,T=T,token_correct=round(tok),string_correct=round(st),**cell))
  runs.append(dict(mode=mode,best_step=best['step'],final_ema_loss=train[-1]['ema_loss'],
    train_elapsed_sec=train[-1]['elapsed_sec'],weight_sha256=hashes,
    checkpoints_identical=hashes['best']==hashes['final']))
 ignore={'run_name','retrieval_rope','compressor_rope'}
 common=[{k:v for k,v in c.items() if k not in ignore} for c in configs]
 assert all(c==common[0] for c in common)
 with (dest/'metrics.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 summary=dict(stage=stage,runs=runs,evaluation_cells=len(rows),all_checks_passed=True,
  scope='Fixed M=10, P=0; 33 horizons through T8192, n256 per checkpoint; not a 16M test',
  first_imperfect={f'{m}/{c}':next((r['T'] for r in rows if r['mode']==m and r['checkpoint']==c and r['string_correct']<256),None)
   for m in MODES for c in ('best','final')})
 (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
 for mode,color in zip(MODES,('tab:blue','tab:orange','tab:green')):
  for cp,style in (('best','--'),('final','-')):
   rs=[r for r in rows if r['mode']==mode and r['checkpoint']==cp]
   for ax,key in zip(axes,('string_acc','token_acc')):
    ax.plot([r['T'] for r in rs],[r[key] for r in rs],style+'o',color=color,ms=3,label=f'{mode} {cp}')
 for ax,title in zip(axes,('Exact string accuracy','Token accuracy')):
  ax.set(xscale='log',xlabel='T',ylabel=title,ylim=(-.02,1.02));ax.axvline(2028,color='gray',ls=':');ax.grid(alpha=.2);ax.legend(fontsize=7)
 fig.savefig(dest/'comparison.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
 for mode in MODES:
  for ax,key in zip(axes,('ema_loss','string_acc')):ax.plot([r['step'] for r in curves[mode]],[r[key] for r in curves[mode]],label=mode,lw=.8)
 axes[0].set(yscale='log',ylabel='Training EMA loss');axes[1].set(ylabel='Training interval string accuracy')
 for ax in axes:ax.set(xlabel='Step');ax.grid(alpha=.2);ax.legend()
 fig.savefig(dest/'learning.png',dpi=160);plt.close(fig)
 files=sorted(f for f in dest.rglob('*') if f.is_file() and f.name!='manifest.sha256')
 (dest/'manifest.sha256').write_text(''.join(f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(dest)}\n' for f in files))
 print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
