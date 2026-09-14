"""CPU audit: replay each evaluation cell and compare historical variable-M runs."""
import json
import math
import shutil
import sys
from pathlib import Path
import numpy as np
from common import ROOT, SOURCE, HERE, REPO, TASKS, audit_source, sha, save

def main():
    audit_source()
    sys.path.insert(0,str(SOURCE))
    import torch
    from exp.variable_memory.task import make_batch
    from exp.variable_memory.common import cell_seed
    from exp.variable_memory.evaluate import grid
    torch.set_num_threads(1)
    out=HERE/'results'; out.mkdir(exist_ok=True)
    rows=[]; training=[]; weights={}; observed_hashes={}
    expected_grid=grid()
    for task in TASKS:
        folder=ROOT/'runs'/task
        worker=json.loads((ROOT/f'worker-{task}.json').read_text())
        assert worker['state']=='complete' and worker['first_300_steps_match_benchmark']
        assert len(worker['commands'])==3 and all(c['returncode']==0 for c in worker['commands'])
        cfg=json.loads((folder/'run_config.json').read_text())
        assert cfg['steps']==50000 and cfg['batch_size']==64 and cfg['grad_accum']==2
        assert cfg['memory_lengths']==[10,16,32,64] and cfg['prefix_range']==[0,63]
        assert cfg['initial_common_sha256']=='2898927c4fec087b5f0d10ab5886d8bc40184c79bdff06bed551da9fe73369b5'
        log=[json.loads(s) for s in (folder/'train_log.jsonl').read_text().splitlines()]
        intervals=[r for r in log if 'loss' in r]; validation=[r for r in log if 'validation' in r]
        assert [r['step'] for r in intervals]==list(range(100,50001,100))
        assert [r['step'] for r in validation]==list(range(2000,50001,2000))
        assert all(math.isfinite(r['loss']) and math.isfinite(r['ema_loss']) for r in intervals)
        # Replay M/P/T metadata independently; validate cumulative training histograms.
        gen=torch.Generator().manual_seed(2); histogram={str(m):0 for m in (10,16,32,64)}
        for step in range(1,50001):
            m=(10,16,32,64)[int(torch.randint(4,(1,),generator=gen))]
            torch.randint(64,(1,),generator=gen); torch.rand(1,generator=gen)
            histogram[str(m)]+=1
            if step%100==0: assert histogram==intervals[step//100-1]['memory_histogram']
        ema={r['step']:r['ema_loss'] for r in intervals}
        for v in validation:
            assert len(v['validation'])==32
            assert v['macro_string']==sum(c['string_acc'] for c in v['validation'])/32
            assert v['macro_token']==sum(c['token_acc'] for c in v['validation'])/32
        chosen=max(validation,key=lambda r:(r['macro_string'],r['macro_token'],-ema[r['step']]))
        best=json.loads((folder/'best.json').read_text())
        assert best['step']==chosen['step'] and best['ema_loss']==ema[chosen['step']]
        training.append(dict(task=task,best=best,final_interval=intervals[-1]))
        dest=out/task; dest.mkdir(exist_ok=True)
        for name in ['run_config.json','train_log.jsonl','best.json']:
            shutil.copy2(folder/name,dest/name)
        for checkpoint,sub in [('best','model_best'),('final','model')]:
            weights[f'{task}/{checkpoint}']=sha(folder/sub/'model.safetensors')
            assert weights[f'{task}/{checkpoint}']==worker['weights'][sub]
            model_cfg=json.loads((folder/sub/'config.json').read_text())
            assert model_cfg['conv_kernel_size']==4 and not model_cfg['phase_emb']
            assert model_cfg['num_layers']==2 and model_cfg['self_slot'] and model_cfg['gated_attention']
            shutil.copy2(folder/sub/'config.json',dest/f'config_{checkpoint}.json')
            result=json.loads((folder/f'results_{checkpoint}.json').read_text())
            assert result['complete'] and result['samples']==256 and result['seed']==12345
            assert [(r['memory_len'],r['T'],r['prefix'],r['split']) for r in result['results']]==expected_grid
            for r in result['results']:
                m,t,p=r['memory_len'],r['T'],r['prefix']
                path=Path(r['observations']); assert sha(path)==r['observations_sha256']
                observed_hashes[str(path)]=r['observations_sha256']
                with np.load(path,allow_pickle=False) as obs:
                    target=obs['target']; prediction=obs['prediction']; positions=obs['positions']
                    assert target.shape==prediction.shape==positions.shape==(256,m)
                    assert np.isfinite(obs['margin']).all()
                    correct=target==prediction
                    assert int(correct.sum())==r['token_correct']
                    assert int(correct.all(-1).sum())==r['string_correct']
                    assert (~correct).sum(0).tolist()==r['digit_errors']
                    assert r['n']==256 and r['token_acc']==r['token_correct']/(256*m)
                    assert r['string_acc']==r['string_correct']/256
                    gen=torch.Generator().manual_seed(cell_seed(task,m,t,p,12345))
                    batch=min(64,max(1,2**18//(p+t+2*m)))
                    for start in range(0,256,batch):
                        x,y=make_batch(task,m,t,p,min(batch,256-start),gen)
                        actual=((x>=1)&(x<=8)).nonzero(as_tuple=False)[:,1].reshape(x.shape[0],m)
                        assert np.array_equal(y[:,-m:].numpy(),target[start:start+batch])
                        assert np.array_equal(actual.numpy(),positions[start:start+batch])
                rows.append(dict(task=task,checkpoint=checkpoint,mode='causal-conv4',**r))
            shutil.copy2(folder/f'results_{checkpoint}.json',dest/f'results_{checkpoint}.json')
    # Historical references use the same tasks, cell seeds and fp32/autocast evaluation.
    # Strict training determinism differs, so this is a one-seed historical comparison.
    prior=REPO/'doc/experiments/logkv-position-study-20260907/runs'
    comparisons=[]
    for task in TASKS:
        for cp in ['best','final']:
            new=[r for r in rows if r['task']==task and r['checkpoint']==cp]
            for mode in ['none','phase2','binding','relative-kv','combined','combined-no-decay']:
                old=json.loads((prior/f'{mode}-{task}'/f'results_{cp}.json').read_text())['results']
                assert len(old)==len(new)==220
                for a,b in zip(new,old):
                    assert (a['memory_len'],a['T'],a['prefix'],a['n'])==(b['memory_len'],b['T'],b['prefix'],b['n'])
                    comparisons.append(dict(task=task,checkpoint=cp,reference=mode,M=a['memory_len'],T=a['T'],P=a['prefix'],
                        new_string=a['string_correct'],reference_string=b['string_correct'],
                        new_token=a['token_acc'],reference_token=b['token_acc']))
    save(out/'metrics.json',rows); save(out/'training.json',training)
    save(out/'historical_comparison.json',comparisons)
    save(out/'review.json',dict(passed=True,cpu_only=True,cells_recounted=len(rows),weights=weights,
        generator_replay_verified=True,best_selection_verified=True,observations=observed_hashes,
        files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='review.json'}))
    print('Completed audit: 880 cells, sample replay, checkpoint selection and historical comparisons.',flush=True)

if __name__=='__main__':main()
