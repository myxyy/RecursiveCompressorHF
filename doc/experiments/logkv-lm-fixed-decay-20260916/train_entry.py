"""Observe original train_logkv.main; no changes to optimizer updates or model math."""
import json
import os
import sys
import time
from common import SOURCE, ROOT, DATA, save, sha, shared_hash, HERE
sys.path.insert(0, str(SOURCE))
import torch
import torch.distributed as dist
import train_logkv as train

args = train.parse_args()
rank = int(os.environ['RANK'])
run = DATA / 'checkpoints_logkv' / args.run_name
run.mkdir(parents=True, exist_ok=True)
train.CONTROL_FILE = str(ROOT / f'control-{args.run_name}.cmd')
model = None
step = 0
last = None

original_split = train.split_params_for_muon
original_step = torch.optim.AdamW.step
original_samples = train.generate_samples


def record():
    global last
    torch.cuda.synchronize()
    now = time.monotonic()
    assert all(l.attention.level_decay is None for l in model.layers)
    if rank == 0:
        with (run/'progress.jsonl').open('a') as f:
            f.write(json.dumps(dict(step=step,
                seconds_since_record=None if last is None else now-last,
                allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                reserved_gib=torch.cuda.max_memory_reserved()/2**30))+'\n')
    last = now


def split(m):
    global model
    model = m
    muon, adam = original_split(m)
    assert not m.config.learnable_decay
    assert all(l.attention.level_decay is None for l in m.layers)
    if rank == 0:
        assert not (run/'progress.jsonl').exists(), 'Refuse overwrite'
        actual_hash=shared_hash(m)
        expected=json.loads((HERE/'initialization_audit.json').read_text())['shared_initial_sha256']
        assert actual_hash==expected, 'initial weights differ'
        save(run/'initialization_audit.json',dict(passed=True,shared_initial_sha256=actual_hash))
        save(run/'run_config.json',dict(arguments=vars(args),world_size=dist.get_world_size(),
            num_params=sum(p.numel() for p in m.parameters()),
            effective_batch=24, input_tokens_per_step=24*2047,
            torch_version=torch.__version__, cuda_version=torch.version.cuda,
            source_manifest=json.loads((__import__('pathlib').Path(__file__).parent/'source_manifest.json').read_text()),
            decay='Fixed Python scalar -level*log(4); no level_decay parameters'))
    record()
    return muon, adam


def observed_step(opt, *a, **kw):
    global step
    result = original_step(opt, *a, **kw)
    step += 1
    if step % 10 == 0 or step == args.max_steps: record()
    return result


def samples(*a, **kw):
    # Keep periodic sampling from perturbing rank0's subsequent training RNG.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(12345 + int(a[3]))
        return original_samples(*a, **kw)

train.split_params_for_muon = split
torch.optim.AdamW.step = observed_step
train.generate_samples = samples
train.main()
