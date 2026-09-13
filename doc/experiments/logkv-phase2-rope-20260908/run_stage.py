"""Run exactly one task's three-mode comparison, on at most two GPUs.

No chaining to another task, long-horizon probe, or automatic retries.
"""
import argparse,concurrent.futures,datetime,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path('/mnt/raid0/RecursiveCompressor/experiments/logkv-phase2-rope-20260908')
SOURCE=ROOT/'source'
MODES=('phase2','retrieval-rope','compressor-rope')

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def main():
 p=argparse.ArgumentParser();p.add_argument('--task',choices=('copying','selective-copying'),required=True)
 args=p.parse_args()
 root=ROOT/args.task;root.mkdir(exist_ok=False)
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=SOURCE,text=True)
 deadline=time.monotonic()+8*3600
 stage=dict(task=args.task,source=str(SOURCE),commit=commit,started=now(),gpu_limit=2,
  steps=50000,max_t_exp=13,samples=256,state='running',next_stage_queued=False)
 save(root/'stage.json',stage)
 def worker(gpu,modes):
  for mode in modes:
   name=f'{mode}-fixed10-20260908';run_dir=ROOT/'exp'/args.task/name
   if run_dir.exists():raise FileExistsError(run_dir)
   env={**os.environ,'DATA_DIR':str(ROOT),'CUDA_VISIBLE_DEVICES':str(gpu),
        'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}
   record=dict(mode=mode,task=args.task,gpu=gpu,source_commit=commit,started=now(),commands=[],state='starting')
   flags=[] if mode=='phase2' else [f'--{mode}']
   commands=[('train',[sys.executable,f'exp/{args.task}/train.py','--run-name',name,
    '--arch','logkv','--phase-emb','--phase-levels','2','--gated-attention','--self-slot',
    '--t-dist','loguniform','--max-t','2028','--steps','50000','--batch-size','64',
    '--grad-accum','1','--lr','0.0003','--warmup','1000','--d-model','512','--num-heads','8',
    '--d-ff','1024','--num-layers','2','--chunk-size','4','--loss-positions','all','--seed','0','--device','0',*flags])]
   for checkpoint in ('best','final'):
    commands.append((checkpoint,[sys.executable,f'exp/{args.task}/evaluate.py','--run-name',name,
     '--samples','256','--max-t-exp','13','--seed','12345','--precision','bf16',
     '--checkpoint',checkpoint,'--device','0']))
   try:
    for label,command in commands:
     entry=dict(stage=label,command=command,started=now(),returncode=None)
     record['state']=label;record['commands'].append(entry);save(root/f'{mode}.json',record)
     remaining=deadline-time.monotonic()
     if remaining<=0:raise TimeoutError('Eight-hour stage limit; confirmation required')
     with (root/f'{mode}-{label}.log').open('w') as log:
      result=subprocess.run(command,cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=remaining)
     entry.update(returncode=result.returncode,finished=now());save(root/f'{mode}.json',record)
     if result.returncode:raise RuntimeError(f'{mode} {label}: exit {result.returncode}')
     if label!='train':
      for filename in ('results.json','plot.png'):
       path=Path(filename);shutil.copy2(run_dir/path,run_dir/f'{path.stem}_{label}{path.suffix}')
    record['state']='complete'
   except Exception as exc:
    record.update(state='stopped',error=str(exc));raise
   finally:
    record['finished']=now();save(root/f'{mode}.json',record)
 try:
  # Longer retrieval mode gets its own GPU; baseline then compressor use GPU 0.
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   jobs=[pool.submit(worker,0,('phase2','compressor-rope')),pool.submit(worker,1,('retrieval-rope',))]
   for job in jobs:job.result()
  stage['state']='complete-awaiting-review'
 except Exception as exc:
  stage.update(state='stopped',error=str(exc));raise
 finally:
  stage['finished']=now();save(root/'stage.json',stage)

if __name__=='__main__':main()
