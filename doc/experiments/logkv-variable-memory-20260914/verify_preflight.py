"""Replay all440 benchmark cells on CPU before committing to50k training."""
import json
import sys
import numpy as np
from common import ROOT, HERE, SOURCE, TASKS, audit_source, sha, save

def main():
    audit_source(); sys.path.insert(0,str(SOURCE))
    import torch
    from exp.variable_memory.task import make_batch
    from exp.variable_memory.common import cell_seed
    from exp.variable_memory.evaluate import grid
    torch.set_num_threads(1)
    files={}; count=0
    for task in TASKS:
        folder=ROOT/'benchmark'/task
        result=json.loads((folder/'results_best.json').read_text())
        assert result['complete'] and result['samples']==16
        assert [(r['memory_len'],r['T'],r['prefix'],r['split']) for r in result['results']]==grid()
        for r in result['results']:
            m,t,p=r['memory_len'],r['T'],r['prefix']; n=16
            from pathlib import Path
            path=Path(r['observations']); assert sha(path)==r['observations_sha256']
            files[str(path)]=sha(path)
            with np.load(path,allow_pickle=False) as obs:
                target=obs['target']; pred=obs['prediction']; positions=obs['positions']
                assert target.shape==pred.shape==positions.shape==(n,m)
                correct=target==pred
                assert r['n']==n and int(correct.sum())==r['token_correct']
                assert int(correct.all(-1).sum())==r['string_correct']
                assert (~correct).sum(0).tolist()==r['digit_errors']
                gen=torch.Generator().manual_seed(cell_seed(task,m,t,p,12345))
                batch=min(64,max(1,2**18//(p+t+2*m)))
                for start in range(0,n,batch):
                    x,y=make_batch(task,m,t,p,min(batch,n-start),gen)
                    actual=((x>=1)&(x<=8)).nonzero(as_tuple=False)[:,1].reshape(x.shape[0],m)
                    assert np.array_equal(y[:,-m:].numpy(),target[start:start+batch])
                    assert np.array_equal(actual.numpy(),positions[start:start+batch])
            count+=1
    save(HERE/'evaluation_smoke.json',dict(passed=True,cpu_only=True,cells=count,
        source_task_sha256=sha(SOURCE/'exp/variable_memory/task.py'),observations=files))
    print(f'CPU input/position replay and recount passed: {count} benchmark cells.')

if __name__=='__main__':main()
