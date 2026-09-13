"""One independently trained task and both checkpoints per GPU."""
import argparse
import json
import os
import subprocess
import sys
from common import HERE, ROOT, SOURCE, MODES, run_dir, command, now, save, sha

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
        assert cfg['steps'] == 50000 and cfg['num_params'] == 5786112
        assert not cfg['phase_emb'] and cfg['self_slot'] and cfg['gated_attention']
        init = json.loads((run_dir(mode) / 'initialization_audit.json').read_text())
        assert init['passed'] and init['task'] == mode
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
