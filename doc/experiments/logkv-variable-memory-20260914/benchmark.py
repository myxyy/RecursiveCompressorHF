"""Single-GPU maximum-shape check, then actual 300-step train/full-grid timing."""
import json
import subprocess
import sys
import time
from common import ROOT, SOURCE, TASKS, audit_source, train_command, eval_command, save, now, sha

def main():
    task = sys.argv[1]; assert task in TASKS
    audit_source()
    sys.path.insert(0, str(SOURCE))
    import torch
    from exp.variable_memory.common import initialize
    from exp.variable_memory.task import make_batch
    torch.set_num_threads(1); torch.set_float32_matmul_precision('high')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    model, digest = initialize('causal-conv4',0)
    model.cuda().train()
    opt = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0)
    generator = torch.Generator().manual_seed(2026)
    for _ in range(2):
        x,y = make_batch(task,64,2028,63,32,generator,torch.device('cuda'))
        with torch.autocast('cuda',dtype=torch.bfloat16): out = model(x,labels=y)
        (out.loss/2).backward()
        assert torch.isfinite(out.loss)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    assert torch.isfinite(norm); opt.step()
    peak = torch.cuda.max_memory_allocated()
    del x,y,out,opt,model; torch.cuda.empty_cache()
    folder = ROOT/'benchmark'/task
    started = now()
    # Subprocesses release all GPU storage at stage boundaries.
    for command in [train_command(sys.executable,task,folder,300),
                    eval_command(sys.executable,task,folder,'best',16)]:
        subprocess.run(command,cwd=SOURCE,check=True,timeout=1200)
    rows = [json.loads(s) for s in (folder/'train_log.jsonl').read_text().splitlines()]
    train = [r for r in rows if 'loss' in r]
    result = json.loads((folder/'results_best.json').read_text())
    assert result['complete'] and len(result['results']) == 220
    assert all(r['n']==16 for r in result['results'])
    train_seconds = train[-1]['elapsed_sec']
    # The training timer excludes final validation; measure the complete validation
    # suite separately on the benchmark checkpoint, without updating any weights.
    from logkv_lm import LogKVLM
    from exp.variable_memory.common import evaluate_cell, MEMORIES
    model = LogKVLM.from_pretrained(folder/'model_best').float().cuda().eval()
    t0=time.monotonic()
    for m in MEMORIES:
        for t in (16,64,256,1024):
            for p in (0,7): evaluate_cell(model,task,m,t,p,32,54321,torch.device('cuda'))
    torch.cuda.synchronize()
    validation_seconds = time.monotonic()-t0
    eval_seconds = sum(r['elapsed_sec'] for r in result['results'])
    # Conservative sample-linear evaluation scaling, 25 validations, both checkpoints.
    estimate = 1.25*(train_seconds/300*50000 + validation_seconds*25 + eval_seconds*16*2)+300
    save(ROOT/f'benchmark-{task}.json',dict(passed=True,task=task,started=started,finished=now(),
        training_steps=300,train_seconds=train_seconds,validation_seconds=validation_seconds,
        evaluation_220_cells_16_samples_seconds=eval_seconds,projected_seconds=estimate,
        maximum_shape=[32,2219],peak_allocated_bytes=peak,initial_common_sha256=digest,
        initial_weights_sha256=sha(folder/'initial_model/model.safetensors')))
    print(f'{task}: projected {estimate/3600:.2f}h',flush=True)

if __name__=='__main__': main()
