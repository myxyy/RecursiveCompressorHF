"""Fixed-control LM evaluation: identical unseen packed rows and generation settings."""
import argparse
import collections
import json
import math
import sys
import time
import types
import numpy as np
from common import HERE, ROOT, DATA, SOURCE, RUN_NAME, save, sha, BASE
sys.path.insert(0, str(SOURCE))
import torch
from logkv_lm import LogKVLM
from dataset import prepare_all_datasets


def inputs():
    dataset, tok = prepare_all_datasets(2048, cache_dir=str(DATA/'hf_cache'), prefault=False)
    # DistributedSampler(seed=0, epoch=0), six ranks, batch4, 5000 steps:
    # the consumed global indices are precisely the first 120000 of randperm.
    gen = torch.Generator().manual_seed(0)
    consumed = set(torch.randperm(len(dataset), generator=gen)[:120000].tolist())
    rng = np.random.default_rng(12345)
    ids, sources = [], []
    offset = 0
    for source, part in zip(['wiki_ja','wiki_en','cc100_ja','minipile'], dataset.datasets):
        local = []
        while len(local) < 32:
            index = offset + int(rng.integers(len(part)))
            if index not in consumed and index not in local: local.append(index)
        ids.extend(local); sources.extend([source]*len(local)); offset += len(part)
    x, y = zip(*(dataset[i] for i in ids))
    x, y = torch.stack(x), torch.stack(y)
    manifest = dict(indices=ids,sources=sources,dataset_length=len(dataset),
        consumed_rows=120000, disjoint_from_training=not bool(set(ids)&consumed),
        input_sha256=__import__('hashlib').sha256(x.numpy().tobytes()).hexdigest(),
        labels_sha256=__import__('hashlib').sha256(y.numpy().tobytes()).hexdigest(),
        caveat='Unconsumed packed rows, not a document-deduplicated external validation set.')
    return x,y,tok,manifest


def repetition(ids,text):
    chars=list(text);q4=chars[3*len(chars)//4:]
    bigrams=list(zip(q4,q4[1:]))
    grams=list(zip(ids,ids[1:],ids[2:],ids[3:]))
    longest=run=0;previous=None
    for t in ids:
        run=run+1 if t==previous else 1;longest=max(longest,run);previous=t
    return dict(q4_char_bigram_distinct=None if not bigrams else len(set(bigrams))/len(bigrams),
        token_4gram_repeat_fraction=None if not grams else 1-len(set(grams))/len(grams),
        longest_same_token_run=longest)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    torch.set_float32_matmul_precision('high');torch.set_num_threads(1)
    run=DATA/'checkpoints_logkv'/('benchmark-30' if args.smoke else RUN_NAME)
    step=30 if args.smoke else 5000
    ckpt=run/f'checkpoint-{step}'/'model'
    out=ROOT/('evaluation_smoke' if args.smoke else 'evaluation');out.mkdir(exist_ok=True)
    started=time.monotonic(); weights=ckpt/'model.safetensors';before=sha(weights)
    model=LogKVLM.from_pretrained(ckpt).cuda().eval()
    assert not model.config.learnable_decay and all(l.attention.level_decay is None for l in model.layers)
    x,y,tok,manifest=inputs();assert manifest==json.loads((BASE/'results/heldout_manifest.json').read_text());save(out/'heldout_manifest.json',manifest)
    tok.save_pretrained(ROOT/'tokenizer')
    losses=[]
    for mode in ['fixed']:
        totals=collections.defaultdict(lambda:[0.,0])
        count=2 if args.smoke else len(x)
        for i in range(count):
            target=y[i:i+1].cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):r=model(x[i:i+1].cuda(),labels=target)
            n=int((target!=-100).sum()); totals[manifest['sources'][i]][0]+=r.loss.item()*n;totals[manifest['sources'][i]][1]+=n
        for source,(nll,n) in totals.items(): losses.append(dict(mode=mode,source=source,tokens=n,loss=nll/n,perplexity=math.exp(nll/n)))
    save(out/'heldout_loss.json',losses)
    prompts=['日本の首都は','昔々あるところに','人工知能とは']
    specs=[(p,seed,temp,1024) for temp in [.7,1.] for seed in range(3) for p in prompts]
    specs += [(p,0,.7,4096) for p in prompts]
    if args.smoke:specs=[(prompts[0],0,.7,8)]
    samples=[]
    for prompt,seed,temp,limit in specs:
        torch.manual_seed(seed);ids=tok(prompt,return_tensors='pt').input_ids.cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output=model.generate(ids,max_new_tokens=limit,do_sample=True,temperature=temp,top_p=.9,
                repetition_penalty=1.,pad_token_id=tok.pad_token_id)
        new=output[0,ids.size(1):].cpu().tolist();text=tok.decode(new,skip_special_tokens=True)
        row=dict(prompt=prompt,seed=seed,temperature=temp,top_p=.9,max_new_tokens=limit,
            generated_tokens=len(new),eos_terminated=bool(new and new[-1]==tok.eos_token_id),
            token_ids=new,text=text,**repetition(new,text))
        samples.append(row);save(out/'generations.json',samples)
        print('generated',len(samples),'/',len(specs),len(new),flush=True)
    assert sha(weights)==before
    summary=[]
    for limit in sorted(set(r['max_new_tokens'] for r in samples)):
        for temp in sorted(set(r['temperature'] for r in samples if r['max_new_tokens']==limit)):
            rows=[r for r in samples if r['max_new_tokens']==limit and r['temperature']==temp]
            rates=[r['q4_char_bigram_distinct'] for r in rows if r['q4_char_bigram_distinct'] is not None]
            summary.append(dict(limit=limit,temperature=temp,n=len(rows),eos=sum(r['eos_terminated'] for r in rows),
                mean_length=float(np.mean([r['generated_tokens'] for r in rows])),
                mean_q4_distinct=float(np.mean(rates)) if rates else None,
                q4_below_half=sum(v<.5 for v in rates)))
    save(out/'generation_summary.json',summary)
    save(out/'review.json',dict(passed=True,step=step,elapsed_seconds=time.monotonic()-started,
        weights_sha256=before,heldout_disjoint=True,
        generation_count=len(samples),precision='fp32 weights + bf16 autocast; original fixed scalar bias',
        caveat='Independent fixed-decay training control; scalar/tensor bias arithmetic differs under autocast from learned option.',
        hashes={p.name:sha(p) for p in out.glob('*.json') if p.name!='review.json'}))
    print('Evaluation complete',flush=True)


if __name__=='__main__':main()
