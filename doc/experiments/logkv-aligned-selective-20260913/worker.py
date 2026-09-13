import argparse
import json
import os
import subprocess
import sys
from common import HERE, ROOT, SOURCE, MODES, OLD_CONFIG, run_dir, command, now, save, sha

def main():
    p=argparse.ArgumentParser(); p.add_argument('--mode',choices=MODES,required=True); a=p.parse_args()
    out=ROOT/a.mode; out.mkdir(parents=True,exist_ok=True)
    record=dict(mode=a.mode,position_mode=a.mode,task='selective-copying',deterministic_training=True,started=now(),state='running',commands=[])
    def execute(stage,cmd):
        row=dict(stage=stage,command=cmd,started=now()); record['commands'].append(row)
        record['state']=stage; save(out/'worker.json',record)
        with (out/f'{stage}.log').open('w') as log:
            proc=subprocess.run(cmd,cwd=SOURCE,stdout=log,stderr=subprocess.STDOUT)
        row.update(returncode=proc.returncode,finished=now()); save(out/'worker.json',record)
        if proc.returncode: raise RuntimeError(f'{stage}: exit {proc.returncode}')
    try:
        execute('train',command(a.mode,sys.executable))
        cfg=json.loads((run_dir(a.mode)/'run_config.json').read_text())
        old=json.loads(OLD_CONFIG.read_text()); ignore={'run_name','retrieval_rope_scale','aligned_rope','aligned_rope_scale','retrieval_rope'}
        assert {k:v for k,v in cfg.items() if k not in ignore}=={k:v for k,v in old.items() if k not in ignore}
        assert cfg['aligned_rope']==(a.mode=='aligned')
        assert cfg['retrieval_rope']==(a.mode=='local-control')
        assert cfg['aligned_rope_scale']==cfg['retrieval_rope_scale']==1.0
        init=json.loads((run_dir(a.mode)/'initialization_audit.json').read_text())
        assert init['passed'] and init['task']=='selective-copying'
        weights={cp:sha(run_dir(a.mode)/folder/'model.safetensors') for cp,folder in [('best','model_best'),('final','model')]}
        for cp in ['best','final']:
            execute(cp,[sys.executable,str(HERE/'evaluate.py'),'--mode',a.mode,'--checkpoint',cp])
        assert weights=={cp:sha(run_dir(a.mode)/folder/'model.safetensors') for cp,folder in [('best','model_best'),('final','model')]}
        record.update(state='complete',weights=weights)
    except BaseException as exc:
        record.update(state='failed',error=str(exc)); raise
    finally:
        record['finished']=now(); save(out/'worker.json',record)
if __name__=='__main__': main()
