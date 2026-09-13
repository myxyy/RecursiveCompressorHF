"""Short real-checkpoint test of both standard and paired evaluation paths."""
import json
import os
import shutil
import sys
from pathlib import Path
import common
actual=common.ROOT
common.ROOT=actual/'evaluation-smoke'
common.ROOT.mkdir(exist_ok=True)
os.environ['DATA_DIR']=str(common.ROOT)
for gpu,mode in enumerate(common.MODES):
    src=actual/'preflight-training/exp/copying'/f'benchmark-gpu{gpu}-{mode}'
    for folder in ['model_best','model']:
        shutil.copytree(src/folder,common.run_dir(mode)/folder)
    shutil.copy2(src/'run_config.json',common.run_dir(mode)/'run_config.json')
import evaluate
evaluate.ROOT=common.ROOT; evaluate.PAIRED_TS=[15,16,17]
evaluate.ev.build_t_grid=lambda e:[3,16,65]
for mode in common.MODES:
    sys.argv=['evaluate.py','--mode',mode,'--checkpoint','best']
    evaluate.main()
    records=json.loads((common.ROOT/mode/'digits_best.json').read_text())
    metrics=json.loads((common.ROOT/mode/'results_best.json').read_text())['results']
    for T in [3,16,65]:
        target=[r for b in records[str(T)] for r in b['target']]
        pred=[r for b in records[str(T)] for r in b['prediction']]
        assert len(target)==len(pred)==256
        assert sum(t==p for t,p in zip(target,pred))/256==metrics[str(T)]['string_acc']
    paired=json.loads((common.ROOT/mode/'paired_best.json').read_text())
    assert len(paired['records'])==3
common.save(common.HERE/'evaluation_smoke.json',dict(passed=True,modes=list(common.MODES),
    standard_horizons=[3,16,65],standard_samples=256,paired_horizons=[15,16,17],paired_samples=32,
    checkpoint_training_steps=300))
