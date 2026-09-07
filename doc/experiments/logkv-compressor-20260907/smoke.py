import json
import math
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path.cwd()))
from configuration_logkv import LogKVConfig
from logkv_lm import LogKVLM

torch.set_float32_matmul_precision('high')
rows = []
for learned in [False, True]:
    torch.manual_seed(0)
    cfg = LogKVConfig(vocab_size=10, d_model=512, d_ff=1024, num_layers=2,
                      num_heads=8, chunk_size=4, relative_position_bias=True,
                      gated_attention=True, self_slot=True, compressor_decay=math.log(4)/3,
                      learnable_compressor_decay=learned, bos_token_id=None, eos_token_id=None)
    model = LogKVLM(cfg).cuda().train()
    assert all(layer.attention.phase_emb is None for layer in model.layers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0)
    ids = torch.randint(0, 10, (64, 2048), device='cuda')
    before = model.layers[0].attention.compressor.raw_decay
    before = before.detach().clone() if before is not None else None
    torch.cuda.reset_peak_memory_stats()
    for _ in range(2):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(ids, labels=ids)
        assert torch.isfinite(out.loss)
        out.loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); optimizer.zero_grad()
    if learned:
        assert not torch.equal(before, model.layers[0].attention.compressor.raw_decay)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    rows.append(dict(learnable_compressor_decay=learned, loss=out.loss.item(),
                     params=sum(p.numel() for p in model.parameters()),
                     peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
    del out, optimizer, model, ids
    torch.cuda.empty_cache()
Path(__file__).with_suffix('.json').write_text(json.dumps(rows, indent=2)+'\n')
print(json.dumps(rows, indent=2))
