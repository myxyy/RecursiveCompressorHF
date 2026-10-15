"""CPU-only audit of both 16M checkpoints against saved logits and original controls."""
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from common import ROOT,SOURCE,sha,save,now
OUT=ROOT/'extension-16777216'


def read(path):return json.loads(path.read_text())


def main():
    checks={}
    def verify(path,digest):
        assert sha(path)==digest,path
        checks[str(path)]=digest
    launch=read(OUT/'full-stage.json');assert launch['state']=='complete' and launch['returncodes']==[0,0]
    assert launch['elapsed_seconds']<launch['timeout_seconds']
    for f,h in launch['scripts'].items():verify(HERE/f,h)
    expected=torch.randint(1,9,(8,10),generator=torch.Generator().manual_seed(12345)).numpy()
    baseline=HERE.parent.parent/'logkv-causal-conv-20260914/extension-16777216'
    base_review=read(baseline/'review.json');assert base_review['passed']
    for f,h in base_review['files'].items():verify(baseline/f,h)
    base=read(baseline/'full.json')['rows'][0]
    assert np.array_equal(expected,np.asarray([x for r in base['records'] for x in r['target']]))
    rows=[]
    for cp in ['best','final']:
        pre,full=[read(OUT/cp/(stage+'.json')) for stage in ['preflight','full']]
        assert pre['passed'] and full['passed'] and pre['provenance']==full['provenance']
        info=full['provenance'];assert info['checkpoint_label']==cp and info['no_level_bias']
        verify(HERE/'evaluate.py',info['script_sha256'])
        verify(OUT/cp/'preflight.json',launch['preflight_hashes'][cp])
        verify(HERE.parent/'source_manifest.json',info['source_manifest_sha256'])
        for f,h in read(HERE.parent/'source_manifest.json')['files'].items():verify(SOURCE/f,h)
        model=Path(info['checkpoint'])
        verify(model/'model.safetensors',info['checkpoint_sha256']);verify(model/'config.json',info['config_sha256'])
        assert read(model/'config.json')['no_level_bias']
        assert pre['rows'][0]['standard_evaluator_counts_match']
        for payload,ts in [(pre,[2**17,2**20]),(full,[2**24])]:
            assert [r['T'] for r in payload['rows']]==ts
            for r in payload['rows']:
                assert (r['n'],r['batch_size'],r['chunk_len'],r['seed'])==(8,4,8192,12345)
                assert r['sequence_length']==r['T']+20 and len(r['records'])==2
                arrays={k:np.asarray([x for batch in r['records'] for x in batch[k]]) for k in ['memory','target','prediction','margin','logits']}
                assert np.array_equal(arrays['target'],expected) and np.array_equal(arrays['memory'],expected)
                logits=arrays['logits'];assert logits.shape==(8,10,10) and np.isfinite(logits).all()
                pred=logits.argmax(-1);assert np.array_equal(pred,arrays['prediction'])
                correct=np.take_along_axis(logits,expected[...,None],-1).squeeze(-1)
                rivals=logits.copy();np.put_along_axis(rivals,expected[...,None],-np.inf,-1)
                margins=correct-rivals.max(-1);assert np.array_equal(margins,arrays['margin'])
                match=pred==expected
                assert int(match.sum())==r['token_correct'] and int(match.all(-1).sum())==r['string_correct']
                rows.append(dict(checkpoint=cp,T=r['T'],n=8,string_correct=r['string_correct'],token_correct=r['token_correct'],
                    digit_errors=(~match).sum(0).tolist(),minimum_margin=float(margins.min()),median_margin=float(np.median(margins)),seconds=r['seconds']))
        dest=HERE/cp;dest.mkdir(exist_ok=True)
        for f in ['preflight.json','preflight.log','full.json','full.log']:shutil.copy2(OUT/cp/f,dest/f)
    review=dict(passed=True,checked=now(),cpu_only=True,rows=rows,hashes_verified=checks,
        original_fixed_decay_same_eight_memories=True,baseline_string_correct=base['string_correct'],
        baseline_minimum_margin=min(v for r in base['records'] for m in r['margin'] for v in m),
        files={str(p.relative_to(HERE)):sha(p) for cp in ['best','final'] for p in (HERE/cp).iterdir()},
        limitations='One seed, 8 fixed random 10-digit memories shared across horizons and checkpoints; not 256 examples or proof for all strings or lengths. Copying only; no training.')
    save(HERE/'review.json',review)
    print(json.dumps(dict(passed=True,rows=rows),indent=2))


if __name__=='__main__':main()
