"""Shared study configuration and streaming evaluation; no task-specific model cues."""
import hashlib
import json
from pathlib import Path

import torch

from models.logkv.configuration import LogKVConfig
from models.logkv.modeling import LogKVLM
from exp.variable_memory.task import make_batch, score

MODES = ("causal-conv4",)
MEMORIES = (10, 16, 32, 64)


def model_config(mode, conv=4):
    if mode not in MODES:
        raise ValueError(mode)
    return LogKVConfig(vocab_size=10, d_model=512, num_heads=8, d_ff=1024,
                       num_layers=2, chunk_size=4, gated_attention=True, self_slot=True,
                       phase_emb=False, phase_levels=2, conv_kernel_size=conv,
                       pad_token_id=None, bos_token_id=None, eos_token_id=None)


def initialize(mode, seed):
    # Reproduce the original variable-M study's common initialization exactly.
    torch.manual_seed(seed)
    baseline = LogKVLM(model_config(mode, conv=0))
    common = baseline.state_dict()
    torch.manual_seed(seed)
    model = LogKVLM(model_config(mode))
    missing, unexpected = model.load_state_dict(common, strict=False)
    assert not unexpected and missing and all('.causal_conv.' in name for name in missing)
    digest = hashlib.sha256()
    for name, tensor in sorted(common.items()):
        assert torch.equal(model.state_dict()[name], tensor)
        digest.update(name.encode()); digest.update(tensor.numpy().tobytes())
    if seed == 0:
        assert digest.hexdigest() == '2898927c4fec087b5f0d10ab5886d8bc40184c79bdff06bed551da9fe73369b5'
    torch.manual_seed(seed)
    return model, digest.hexdigest()


def cell_seed(task, memory, horizon, prefix, seed):
    text = f"{task}:{memory}:{horizon}:{prefix}:{seed}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little") % (2**63 - 1)


@torch.no_grad()
def evaluate_cell(model, task, memory, horizon, prefix, samples, seed, device,
                  chunk_size=4096, precision="autocast", batch_limit=64, observations=None):
    length = prefix + horizon + 2 * memory
    batch = min(batch_limit, max(1, 2**18 // length))
    gen = torch.Generator().manual_seed(cell_seed(task, memory, horizon, prefix, seed))
    totals = [0, 0, 0, 0]
    for start in range(0, samples, batch):
        inputs, labels = make_batch(task, memory, horizon, prefix, min(batch, samples-start), gen, device)
        hidden = None
        answer = []
        answer_start = length - memory
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda" and precision == "autocast"):
            for offset in range(0, length, chunk_size):
                logits, hidden = model.step(inputs[:, offset:offset+chunk_size], hidden)
                if offset + logits.shape[1] > answer_start:
                    answer.append(logits[:, max(0, answer_start-offset):])
        # Keep the entire answer even if it straddles a chunk boundary.
        answer_logits = torch.cat(answer, dim=1).float()
        counts = score(answer_logits, labels, memory)
        if observations is not None:
            target = labels[:, -memory:]
            mask = (inputs >= 1) & (inputs <= 8)
            positions = mask.nonzero(as_tuple=False)[:, 1].reshape(inputs.shape[0], memory)
            assert torch.equal(inputs.gather(1, positions), target)
            rivals = answer_logits.clone().scatter_(-1, target[..., None], float('-inf'))
            margin = answer_logits.gather(-1, target[..., None]).squeeze(-1) - rivals.max(-1).values
            assert torch.isfinite(answer_logits).all()
            observations.append(dict(target=target.cpu().numpy(), positions=positions.cpu().numpy(),
                prediction=answer_logits.argmax(-1).cpu().numpy(), margin=margin.cpu().numpy()))
        totals = [a+b for a,b in zip(totals,counts)]
    return dict(memory_len=memory, T=horizon, prefix=prefix, n=samples,
                token_correct=totals[0], string_correct=totals[1],
                token_acc=totals[0]/totals[2], string_acc=totals[1]/totals[3])


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(data, indent=2)+"\n")
    temp.replace(path)
