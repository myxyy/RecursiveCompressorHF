"""One independently trained task and both checkpoints per GPU."""
import argparse
import json
import os
import subprocess
import sys
from common import HERE, ROOT, SOURCE, MODES, NUM_LAYERS, NUM_PARAMS, BASE_ROOT, run_dir, command, now, save, sha

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', choices=MODES, required=True)
    mode = parser.parse_args().mode; os.environ['LOGKV_TASK'] = mode
    out = ROOT / mode; out.mkdir(parents=True, exist_ok=True)
    record = dict(task=mode, mode=mode, state='running', started=now(), commands=[])
    def execute(stage, cmd):
        row = dict(stage=stage, command=cmd, started=now()); record['commands'].append(row)
        record['state'] = stage; save(out / 'worker.json', record)
        with (out / f'{stage}.log').open('w') as log:
            result = subprocess.run(cmd, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT)
        row.update(returncode=result.returncode, finished=now()); save(out / 'worker.json', record)
        if result.returncode: raise RuntimeError(f'{stage}: exit {result.returncode}')
    try:
        execute('train', command(mode, sys.executable))
        cfg = json.loads((run_dir(mode) / 'run_config.json').read_text())
        assert cfg['steps'] == 50000 and cfg['num_params'] == NUM_PARAMS and cfg['num_layers'] == NUM_LAYERS and cfg['conv_kernel_size'] == 4
        assert cfg['learnable_decay']
        assert not cfg['phase_emb'] and cfg['self_slot'] and cfg['gated_attention']
        init = json.loads((run_dir(mode) / 'initialization_audit.json').read_text())
        assert init['passed'] and init['task'] == mode
        baseline=json.loads((BASE_ROOT/'exp'/mode/'causal-conv4-fixed10-20260914/run_config.json').read_text())
        differences={k for k in cfg.keys()|baseline.keys() if cfg.get(k)!=baseline.get(k)}
        assert differences=={'run_name','learnable_decay','num_params'}, differences
        benchmark=ROOT/'preflight-training/exp'/mode/f'benchmark-gpu{(0 if mode=="copying" else 1)}-{mode}'
        records=lambda p:[{k:v for k,v in json.loads(line).items() if k!='elapsed_sec'}
                          for line in p.read_text().splitlines() if '"loss"' in line]
        assert records(run_dir(mode)/'train_log.jsonl')[:3]==records(benchmark/'train_log.jsonl')
        record['first_300_steps_match_benchmark']=True
        record['baseline_config_differences']=sorted(differences)
        weights = {cp: sha(run_dir(mode) / folder / 'model.safetensors')
                   for cp, folder in [('best', 'model_best'), ('final', 'model')]}
        for cp in ['best', 'final']:
            execute(cp, [sys.executable, str(HERE / 'evaluate.py'), '--mode', mode, '--checkpoint', cp])
        assert weights == {cp: sha(run_dir(mode) / folder / 'model.safetensors')
                           for cp, folder in [('best', 'model_best'), ('final', 'model')]}
        record.update(state='complete', weights=weights)
    except BaseException as exc:
        record.update(state='failed', error=str(exc)); raise
    finally:
        record['finished'] = now(); save(out / 'worker.json', record)

if __name__ == '__main__': main()
