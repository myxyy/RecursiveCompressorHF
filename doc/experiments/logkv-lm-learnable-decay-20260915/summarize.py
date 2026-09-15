"""CPU-only completion audit and portable artifacts for the 5000-step LM run."""
import json
import math
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from safetensors import safe_open
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from common import HERE, ROOT, DATA, RUN_NAME, save, sha


def main():
    run=DATA/'checkpoints_logkv'/RUN_NAME
    out=HERE/'results';out.mkdir(exist_ok=True)
    history=[json.loads(s) for s in (run/'beta_log.jsonl').read_text().splitlines()]
    assert [r['step'] for r in history]==list(range(0,5001,10))
    beta=np.array([r['beta'] for r in history]);assert beta.shape==(501,16,8) and np.isfinite(beta).all()
    assert np.allclose(beta[0],math.log(4),rtol=0,atol=1e-7)
    bench=[json.loads(s) for s in (DATA/'checkpoints_logkv/benchmark-30/beta_log.jsonl').read_text().splitlines()]
    prefix_match=[r['beta'] for r in history[:4]]==[r['beta'] for r in bench]
    # Cross-DDP executions can use different collective reduction orders; report parity, do not require it.
    checkpoint_hashes={}
    for step in range(1000,5001,1000):
        model=run/f'checkpoint-{step}'/'model';weights=model/'model.safetensors'
        with safe_open(weights,framework='pt',device='cpu') as f:
            values=np.array([f.get_tensor(f'layers.{i}.attention.level_decay').tolist() for i in range(16)])
        assert np.array_equal(values,beta[step//10])
        checkpoint_hashes[str(step)]=sha(weights)
        shutil.copy2(model/'config.json',out/f'config-{step}.json')
    review=json.loads((ROOT/'evaluation/review.json').read_text())
    assert review['passed'] and review['step']==5000 and review['generation_count']==21
    assert review['weights_sha256']==checkpoint_hashes['5000']
    for name,expected in review['hashes'].items():assert sha(ROOT/'evaluation'/name)==expected
    for p in (ROOT/'evaluation').glob('*.json'):shutil.copy2(p,out/p.name)
    for name in ['beta_log.jsonl','run_config.json','samples.log']:shutil.copy2(run/name,out/name)
    shutil.copy2(ROOT/'train.log',out/'train.log')
    accumulator=EventAccumulator(str(DATA/'tensorboard/logkv-pretrain'/RUN_NAME),size_guidance={'scalars':0}).Reload()
    scalars={tag:[dict(step=r.step,value=r.value) for r in accumulator.Scalars(tag)] for tag in accumulator.Tags()['scalars']}
    for tag in ['train/loss','train/grad_norm','train/lr']:
        assert len(scalars[tag])==5000 and [r['step'] for r in scalars[tag]]==list(range(1,5001))
        assert all(math.isfinite(r['value']) for r in scalars[tag])
    save(out/'training_metrics.json',scalars)
    fig,ax=plt.subplots(figsize=(9,6));lim=max(abs(beta[-1].min()),abs(beta[-1].max()))
    im=ax.imshow(beta[-1],cmap='coolwarm',vmin=-lim,vmax=lim,aspect='auto')
    ax.set(xticks=range(8),xticklabels=range(1,9),yticks=range(16),yticklabels=range(1,17),xlabel='Head',ylabel='Layer',title='Learned beta at step5000 (negative = amplification)')
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(out/'beta_final.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(10,7))
    for layer in range(16):axes[0].plot(np.arange(501)*10,beta[:,layer].mean(1),label=str(layer+1))
    axes[0].axhline(math.log(4),color='gray',ls='--');axes[0].axhline(0,color='black',lw=1)
    axes[0].set(title='Layer mean beta',xlabel='Step');axes[0].legend(ncol=8,fontsize=8)
    raw=np.array([r['value'] for r in scalars['train/loss']]);smooth=[];ema=raw[0]
    for v in raw:ema=.99*ema+.01*v;smooth.append(ema)
    axes[1].plot(np.arange(1,5001),raw,alpha=.2,label='rank0 batch loss');axes[1].plot(np.arange(1,5001),smooth,label='EMA .99')
    axes[1].set(xlabel='Step',ylabel='Cross entropy');axes[1].legend()
    fig.tight_layout();fig.savefig(out/'training.png',dpi=160);plt.close(fig)
    save(out/'completion_review.json',dict(passed=True,steps=5000,beta_snapshots=501,
        benchmark_beta_prefix_bitexact=prefix_match,checkpoint_sha256=checkpoint_hashes,
        final_rank0_loss=float(raw[-1]),final_rank0_ema=float(smooth[-1]),
        source_manifest=json.loads((HERE/'source_manifest.json').read_text()),
        output_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='completion_review.json'},
        limitations='One newly trained model. Inference beta resets are not retraining controls. Heldout packed rows can share documents with training; rank0 training loss is not a six-rank average.'))
    print('All LM completion audits passed.',flush=True)


if __name__=='__main__':main()
