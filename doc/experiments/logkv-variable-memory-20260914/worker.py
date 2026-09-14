"""One independent task: train, evaluate best/final, verify unchanged checkpoints."""
import json
import subprocess
import sys
from common import ROOT, SOURCE, TASKS, audit_source, train_command, eval_command, save, now, sha

def main():
    task=sys.argv[1]; assert task in TASKS
    audit_source()
    folder=ROOT/'runs'/task
    record=dict(task=task,state='running',started=now(),commands=[])
    path=ROOT/f'worker-{task}.json'
    save(path,record)
    try:
        commands=[train_command(sys.executable,task,folder,50000)]
        commands += [eval_command(sys.executable,task,folder,cp) for cp in ('best','final')]
        for index,command in enumerate(commands):
            item=dict(command=command,started=now()); record['commands'].append(item); save(path,record)
            if index:
                weights={cp:sha(folder/cp/'model.safetensors') for cp in ['model','model_best']}
            code=subprocess.call(command,cwd=SOURCE)
            item.update(returncode=code,finished=now()); save(path,record)
            if code: raise RuntimeError(f'Command failed: {code}')
            if index:
                assert weights=={cp:sha(folder/cp/'model.safetensors') for cp in weights}
                record['weights']=weights
            else:
                assert sha(folder/'initial_model/model.safetensors') == sha(ROOT/'benchmark'/task/'initial_model/model.safetensors')
                strip=lambda p:[{k:v for k,v in json.loads(s).items() if k!='elapsed_sec'} for s in p.read_text().splitlines() if '"loss"' in s]
                assert strip(folder/'train_log.jsonl')[:3] == strip(ROOT/'benchmark'/task/'train_log.jsonl')
                record['first_300_steps_match_benchmark']=True
        audit_source(); record['state']='complete'
    except BaseException as exc:
        record.update(state='failed',error=str(exc)); raise
    finally:
        record['finished']=now(); save(path,record)

if __name__=='__main__': main()
