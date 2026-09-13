"""Matched initialization, deterministic training repeat and real-size timing."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from common import HERE,ROOT,SOURCE,MODES,command,save,sha,now
sys.path.insert(0,str(SOURCE))
import torch
from configuration_logkv import LogKVConfig
from logkv_lm import LogKVLM
from safetensors.torch import load_file

def main():
    torch.set_num_threads(1)
    baseline=None;initial_rng=None
    for mode in MODES:
        torch.manual_seed(0)
        cfg=LogKVConfig(vocab_size=10,d_model=512,num_heads=8,d_ff=1024,num_layers=2,
            chunk_size=4,phase_emb=False,phase_levels=2,gated_attention=True,self_slot=True,
            retrieval_rope=mode=='local-control',aligned_rope=mode=='aligned',
            pad_token_id=None,bos_token_id=None,eos_token_id=None)
        model=LogKVLM(cfg)
        assert sum(p.numel() for p in model.parameters())==5786112
        if baseline is None:
            baseline=model.state_dict();initial_rng=torch.get_rng_state()
            model.save_pretrained(ROOT/'initial_model')
        else:
            assert torch.equal(initial_rng,torch.get_rng_state())
            assert all(torch.equal(v,baseline[k]) for k,v in model.state_dict().items())
    record=dict(passed=False,initial_weights_identical=True,initial_rng_identical=True,
        num_params=5786112,initial_weights_sha256=sha(ROOT/'initial_model/model.safetensors'),
        deterministic_algorithms=True,cublas_workspace_config=':4096:8',commands=[])
    record['gpu_started_unix']=time.time();record['gpu_started']=now()
    def batch(label,modes,steps):
        procs=[];logs=[]
        try:
            for gpu,mode in enumerate(modes):
                runname=f'{label}-gpu{gpu}-{mode}'
                cmd=command(mode,sys.executable)
                for flag,value in [('--run-name',runname),('--steps',str(steps))]:cmd[cmd.index(flag)+1]=value
                cmd += ['--eval-interval','0','--log-interval',str(min(steps,100)),'--save-interval',str(steps)]
                data=ROOT/'preflight-training'
                assert not (data/'exp/copying'/runname).exists()
                env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'DATA_DIR':str(data),
                    'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8',
                    'PYTHONUNBUFFERED':'1','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}
                log=(ROOT/f'{runname}.log').open('w');logs.append(log)
                proc=subprocess.Popen(cmd,cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT)
                procs.append(proc)
                record['commands'].append(dict(label=label,mode=mode,gpu=gpu,command=cmd,run_name=runname))
            for proc in procs:
                code=proc.wait(timeout=600)
                if code: raise RuntimeError(f'Preflight failed; inspect {label} logs: exit {code}')
        finally:
            for proc in procs:
                if proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
            for log in logs:log.close()
        return [ROOT/'preflight-training/exp/copying'/f'{label}-gpu{g}-{m}' for g,m in enumerate(modes)]
    try:
        # Each mode repeats on both physical GPUs from identical initial state.
        repeats={}
        for mode in MODES:
            paths=batch('repeat-'+mode,[mode,mode],20)
            a,b=[load_file(p/'model/model.safetensors') for p in paths]
            assert a.keys()==b.keys() and all(torch.equal(v,b[k]) for k,v in a.items())
            metrics=[[json.loads(l) for l in (p/'train_log.jsonl').read_text().splitlines()] for p in paths]
            assert [{k:v for k,v in r.items() if k!='elapsed_sec'} for r in metrics[0]]==[{k:v for k,v in r.items() if k!='elapsed_sec'} for r in metrics[1]]
            repeats[mode]=dict(steps=20,physical_gpus=[0,1],weights_bitexact=True,logged_metrics_bitexact=True,
                sha256=[sha(p/'model/model.safetensors') for p in paths])
            print(f'{mode}: 20-step repeat bitexact across GPUs0/1',flush=True)
        paths=batch('benchmark',MODES,300)
        timing={}
        for mode,p in zip(MODES,paths):
            rows=[json.loads(l) for l in (p/'train_log.jsonl').read_text().splitlines() if '"loss"' in l]
            assert rows[-1]['step']==300
            elapsed=rows[-1]['elapsed_sec']
            timing[mode]=dict(steps=300,elapsed_seconds=elapsed,seconds_per_step=elapsed/300,
                train_hours_projected=elapsed/300*50000/3600)
        # 15% timing margin plus one hour for two checkpoints + paired evaluation.
        estimate=max(r['train_hours_projected'] for r in timing.values())*1.15+1.0+(time.time()-record['gpu_started_unix'])/3600
        record.update(passed=True,repeat_training_bitexact=True,repeats=repeats,timing=timing,
            estimated_campaign_hours=estimate,gpu_preflight_hours=(time.time()-record['gpu_started_unix'])/3600,
            finished=now(),limitation='20-step repeat does not prove 50k-step or cross-environment bitwise reproducibility.')
        print(json.dumps(record,indent=2),flush=True)
    except BaseException as exc:
        record.update(error=str(exc),finished=now());raise
    finally:
        save(HERE/'preflight.json',record)
if __name__=='__main__':main()
