"""CPU-only audit and analysis of the real compression traces."""
import csv
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/raid0/RecursiveCompressor/experiments/logkv-compression-diagnostic-20260908')


def norm(x):return np.linalg.norm(x.astype(np.float64),axis=-1)
def cosine(x,y):
    x=x.astype(np.float64);y=y.astype(np.float64)
    denominator=norm(x)*norm(y)
    result=np.full(denominator.shape,np.nan)
    np.divide((x*y).sum(-1),denominator,out=result,where=denominator>0)
    return result
def flat(x):return x.reshape(len(x),-1)
def avg(x):return float(np.nanmean(x)) if np.isfinite(x).any() else None
def table(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def main():
    representation=[];heads=[];readout=[];performance=[];transforms=[]
    for mode in ('phase2','combined-no-decay'):
        for precision in ('bf16','fp32'):
            folder=ROOT/f'{mode}-{precision}'
            m=json.loads((folder/'metadata.json').read_text())
            assert m['complete'] and m['instrumentation_output_exact']
            assert m['pooling_reconstruction_max_abs']==0
            x=np.load(folder/'trace.npz',allow_pickle=False)
            assert all(np.isfinite(x[k]).all() for k in x.files)
            dest=HERE/'runs'/folder.name;dest.mkdir(parents=True,exist_ok=True)
            shutil.copy2(folder/'metadata.json',dest/'metadata.json')
            shutil.copy2(folder/'trace.npz',dest/'trace.npz')
            shutil.copy2(ROOT/f'{folder.name}.log',dest/'run.log')
            labels=x['digits'].astype(int)
            for r in m['results']:
                d=r['depth'];logits=x[f'memory_d{d}_answer_logits']
                good=logits.argmax(-1)==labels
                assert int(good.sum())==r['token_correct'] and int(good.all(-1).sum())==r['string_correct']
                correct=np.take_along_axis(logits,labels[...,None],axis=-1)[...,0]
                other=logits.copy();np.put_along_axis(other,labels[...,None],-np.inf,axis=-1)
                margin=correct-other.max(-1)
                np.testing.assert_array_equal(margin,x[f'memory_d{d}_answer_margin'])
                performance.append(dict(mode=mode,precision=precision,**r,
                                        token_acc=r['token_correct']/160,string_acc=r['string_correct']/16))
                for li in range(2):
                    prefix=f'l{li}_d{d}_'
                    ref=f'l{li}_d2_';prev=f'l{li}_d{max(2,d-1)}_'
                    for quantity in ('k','v'):
                        a=x['memory_'+prefix+quantity];blank=x['blank_'+prefix+quantity]
                        delta=a-blank
                        base=x['memory_'+ref+quantity]-x['blank_'+ref+quantity]
                        prior=x['memory_'+prev+quantity]-x['blank_'+prev+quantity]
                        pre=x['memory_'+prefix+quantity+'_pre_transform']-x['blank_'+prefix+quantity+'_pre_transform']
                        representation.append(dict(mode=mode,precision=precision,layer=li,depth=d,quantity=quantity,
                            raw_norm=avg(norm(flat(a))),signal_norm=avg(norm(flat(delta))),
                            signal_norm_ratio_d2=avg(norm(flat(delta))/norm(flat(base))),
                            signal_cosine_d2=avg(cosine(flat(delta),flat(base))),
                            signal_cosine_previous=avg(cosine(flat(delta),flat(prior))),
                            transform_cosine_pre_post=avg(cosine(flat(pre),flat(delta)))))
                        for h in range(8):
                            ca=cosine(delta[:,h],prior[:,h]);den=norm(base[:,h]);ratios=np.full(16,np.nan)
                            np.divide(norm(delta[:,h]),den,out=ratios,where=den>0)
                            heads.append(dict(mode=mode,precision=precision,layer=li,depth=d,quantity=quantity,head=h,
                                signal_norm=avg(norm(delta[:,h])),signal_norm_ratio_d2=avg(ratios),
                                signal_cosine_previous=avg(ca),defined_cosine_samples=int(np.isfinite(ca).sum()),
                                memory_first_child_weight=avg(x['memory_'+prefix+'weights'][:,h,0])))
                    q=x['memory_'+prefix+'read_q'][:,:,1:]
                    fixed_q=x[f'memory_l{li}_d5_read_q'][:,:,1:]
                    k=x['memory_'+prefix+'k'];dk=k-x['blank_'+prefix+'k']
                    # Frozen depth-5 queries applied offline in float64: content score only.
                    fixed_logit=np.einsum('bhjd,bhd->bhj',fixed_q.astype('float64'),k.astype('float64'))/8
                    fixed_signal=np.einsum('bhjd,bhd->bhj',fixed_q.astype('float64'),dk.astype('float64'))/8
                    for h in range(8):
                        readout.append(dict(mode=mode,precision=precision,layer=li,depth=d,head=h,
                            query_cosine_d5=avg(cosine(q[:,h],fixed_q[:,h])),
                            memory_raw_logit=avg(x['memory_'+prefix+'read_memory_raw_logit'][:,h,1:]),
                            memory_effective_logit=avg(x['memory_'+prefix+'read_memory_effective_logit'][:,h,1:]),
                            memory_vs_max_other=avg(x['memory_'+prefix+'read_memory_vs_max_other'][:,h,1:]),
                            memory_attention_weight=avg(x['memory_'+prefix+'read_memory_weight'][:,h,1:]),
                            fixed_d5_query_content_logit=avg(fixed_logit[:,h]),
                            fixed_d5_query_memory_minus_blank_logit=avg(fixed_signal[:,h])))
            if mode=='combined-no-decay':
                for li in range(2):
                    u=x[f'parameters_l{li}_position_vectors'].astype('float64')
                    perms=x[f'parameters_l{li}_position_permutations'].astype(int)
                    for h in range(8):
                        R=np.eye(64)[:,perms[h,0]]
                        for j in range(2):
                            v=u[h,0,j];v=v/np.linalg.norm(v);R=R-2*(R@v)[:,None]*v[None]
                        # Offline geometric probe of U0; this is not a model intervention.
                        for d in range(2,9):
                            key=f'memory_l{li}_d{d}_v_child'
                            a=x[key][:,h,0].astype('float64')
                            transforms.append(dict(precision=precision,layer=li,head=h,depth=d,
                                identity_distance_fro_normalized=float(np.linalg.norm(R-np.eye(64))/8),
                                orthogonality_max_abs=float(np.abs(R.T@R-np.eye(64)).max()),
                                first_child_cosine_before_after_U0=avg(cosine(a,a@R))))
    table(HERE/'representation.csv',representation);table(HERE/'heads.csv',heads)
    table(HERE/'readout.csv',readout);table(HERE/'performance.csv',performance);table(HERE/'first_child_transform.csv',transforms)
    original=np.load(ROOT/'compat-original.npz',allow_pickle=False)
    current=np.load(ROOT/'compat-current.npz',allow_pickle=False)
    assert set(original.files)==set(current.files)
    compatibility={k:dict(exact=bool(np.array_equal(original[k],current[k])),
                         max_abs=float(np.abs(original[k]-current[k]).max())) for k in original.files}
    assert all(v['exact'] for v in compatibility.values())
    (ROOT/'compatibility.json').write_text(json.dumps(compatibility,indent=2)+'\n')
    for filename in ('compatibility.json','compat-original.npz','compat-current.npz','compat-original.log','compat-current.log'):
        shutil.copy2(ROOT/filename,HERE/filename)
    # Main plot uses concatenated heads, so near-zero inactive heads do not dominate direction averages.
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for li in range(2):
        for mode,color in (('phase2','tab:blue'),('combined-no-decay','tab:orange')):
            for precision,style in (('bf16','-'),('fp32','--')):
                rows=[r for r in representation if (r['layer'],r['mode'],r['precision'],r['quantity'])==(li,mode,precision,'v')]
                for ax,field,title in zip(axes[li],('signal_cosine_previous','signal_norm_ratio_d2','transform_cosine_pre_post'),
                    ('Cosine to previous depth','Norm / depth-2 norm','Cosine before / after position transform')):
                    ax.plot([r['depth'] for r in rows],[r[field] for r in rows],style+'o',color=color,label=f'{mode}, {precision}',ms=3)
                    ax.set(title=f'Layer {li}: {title}',xlabel='Compression depth',xticks=range(2,9));ax.axvline(5.5,color='gray',ls=':',lw=1)
                    ax.grid(alpha=.2)
        for ax in axes[li]:ax.legend(fontsize=7)
    fig.savefig(HERE/'representation.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for mode,color in (('phase2','tab:blue'),('combined-no-decay','tab:orange')):
        for precision,style in (('bf16','-'),('fp32','--')):
            rows=[r for r in performance if (r['mode'],r['precision'])==(mode,precision)]
            for ax,field,title in zip(axes,('string_acc','token_acc','correct_logit_margin_mean'),
                                     ('Exact string accuracy','Token accuracy','Mean correct-vs-other output logit margin')):
                ax.plot([r['depth'] for r in rows],[r[field] for r in rows],style+'o',color=color,label=f'{mode}, {precision}',ms=3)
                ax.set(title=title,xlabel='Compression depth',xticks=range(2,9));ax.axvline(5.5,color='gray',ls=':',lw=1);ax.grid(alpha=.2)
    for ax in axes:ax.legend(fontsize=7)
    fig.savefig(HERE/'performance.png',dpi=160);plt.close(fig)
    selected=[r for r in representation if r['depth'] in (5,6,8)]
    (HERE/'summary.json').write_text(json.dumps(dict(representation=selected,performance=performance,
                    compatibility=compatibility,representation_rows=len(representation),head_rows=len(heads),
                    readout_rows=len(readout),all_checks_passed=True),indent=2)+'\n')
    print('Validated',len(performance),'performance cells;',len(representation),'representation rows;',len(readout),'readout rows')


if __name__=='__main__':main()
