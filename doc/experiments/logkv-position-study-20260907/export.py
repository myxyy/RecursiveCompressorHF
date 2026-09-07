"""Archive completed runs without copying model/optimizer weights into git."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from safetensors import safe_open
import torch

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent,help='Original experiment directory, including checkpoints and frozen source')
parser.add_argument('--dest',type=Path,default=Path('/home/myxy/RecursiveCompressor/doc/experiments/logkv-position-study-20260907'))
args=parser.parse_args()
ROOT=args.root.resolve(); DEST=args.dest.resolve()
torch.set_num_threads(1)
MODES=('none','phase2','binding','relative-kv','combined','combined-no-decay')
manifests=[ROOT/f'{mode}-{task}.json' for mode in MODES for task in ('copying','selective-copying')]
assert all(p.exists() and json.loads(p.read_text())['state']=='complete' for p in manifests)
DEST.mkdir(parents=True,exist_ok=True)
shutil.copy2(ROOT/'archive-README.md',DEST/'README.md')
if Path(__file__).resolve()!=DEST/'export.py': shutil.copy2(__file__,DEST/'export.py')
for p in manifests:
    manifest=json.loads(p.read_text())
    assert len(manifest['commands'])==3 and all(c['returncode']==0 for c in manifest['commands'])
    src=ROOT/'runs'/p.stem; dst=DEST/'runs'/src.name; dst.mkdir(parents=True,exist_ok=True)
    for name in ('run_config.json','train_log.jsonl','best.json','results_best.json','results_final.json'):
        shutil.copy2(src/name,dst/name)
    for checkpoint,folder in [('best','model_best'),('final','model')]:
        shutil.copy2(src/folder/'config.json',dst/f'model_config_{checkpoint}.json')
        stats=[]
        model_weights=src/folder/'model.safetensors'
        weight_digest=hashlib.sha256(model_weights.read_bytes()).hexdigest()
        with safe_open(model_weights,framework='pt',device='cpu') as weights:
            for layer in range(2):
                prefix=f'layers.{layer}.attention.'
                rec=dict(layer=layer,checkpoint_sha256=weight_digest)
                name=prefix+'compressor.position_vectors'
                if name in weights.keys():
                    vectors=weights.get_tensor(name).double()
                    perms=weights.get_tensor(prefix+'compressor.position_permutations')
                    norms=vectors.norm(dim=-1)
                    rec.update(reflection_norm_min=float(norms.min()),reflection_norm_max=float(norms.max()))
                    errors=[]; commutators=[]
                    for h in range(8):
                        matrices=[]; eye=torch.eye(64,dtype=torch.float64)
                        for c in range(4):
                            mat=eye[:,perms[h,c]]
                            for r in range(2):
                                u=vectors[h,c,r]; u=u/u.norm().clamp_min(1e-12)
                                mat=mat@(eye-2*u[:,None]*u[None,:])
                            errors.append(float((mat.T@mat-eye).abs().max())); matrices.append(mat)
                        commutators.append(float((matrices[0]@matrices[1]-matrices[1]@matrices[0]).norm()))
                    rec.update(orthogonality_max_error=max(errors),commutator_01_frobenius=commutators)
                for kind in ('relative_key','relative_value'):
                    if prefix+kind in weights.keys():
                        table=weights.get_tensor(prefix+kind).double()
                        rec[kind+'_norms']=table.norm(dim=-1).tolist()
                stats.append(rec)
        (dst/f'position_stats_{checkpoint}.json').write_text(json.dumps(stats,indent=2)+'\n')
    shutil.copy2(p,DEST/p.name)
for name in ('environment.json','compatibility.json','probe.json','probe-reflections-only.json','summarize.py','probe_trained.py',
             'trained-encoder-copying.json','trained-encoder-selective-copying.json'):
    shutil.copy2(ROOT/name,DEST/name)
for p in ROOT.glob('smoke-*.json'): shutil.copy2(p,DEST/p.name)
shutil.copy2(ROOT/'source'/'exp'/'position_study'/'probe.py',DEST/'probe.py')
shutil.copy2(ROOT/'source'/'exp'/'position_study'/'smoke.py',DEST/'smoke.py')
subprocess.run([sys.executable,str(DEST/'summarize.py')],check=True)
print(DEST)
