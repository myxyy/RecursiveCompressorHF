"""Verify convolution-enabled batch64/T2028 training fits before launching any long run."""
import os
import sys
from common import HERE, SOURCE, INITIAL, BASE_INITIAL, NUM_PARAMS, bind_task, save, sha

assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
import torch
sys.path.insert(0, str(SOURCE))
from logkv_lm import LogKVLM
from configuration_logkv import LogKVConfig

torch.set_num_threads(1)
torch.set_float32_matmul_precision('high')
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
task = bind_task('copying')
streaming = []
for dtype, autocast in [(torch.float64, False), (torch.float32, True),
                        (torch.bfloat16, False), (torch.bfloat16, True)]:
    torch.manual_seed(0)
    tiny = LogKVLM(LogKVConfig(vocab_size=10, d_model=16, num_heads=4, d_ff=32,
        num_layers=2, conv_kernel_size=4, self_slot=True, gated_attention=True)).to('cuda', dtype).eval()
    ids = torch.randint(0, 10, (2, 67), device='cuda')
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=autocast):
        ref = tiny(ids).logits
        a, h = tiny.step(ids[:, :5]); b, h = tiny.step(ids[:, 5:65], h)
        c, h = tiny.predict(ids[:, 65], h); d, h = tiny.predict(ids[:, 66], h)
        split = torch.cat([a, b, c[:, None], d[:, None]], dim=1)
        error = (split-ref).abs().max().item()
        tolerance = 1e-12 if dtype == torch.float64 else 5e-3
        assert error < tolerance, (dtype, autocast, error)
        assert all(cache.shape == (2, 3, 16) for cache, _ in h)
    streaming.append(dict(dtype=str(dtype), autocast=autocast, max_abs_error=error, tolerance=tolerance))
del tiny, h, ref, split, a, b, c, d, ids
torch.cuda.empty_cache()
model = LogKVLM.from_pretrained(INITIAL.parent).to('cuda').train()
assert model.config.num_layers == 2 and model.config.conv_kernel_size == 4 and not model.config.phase_emb
assert sum(p.numel() for p in model.parameters()) == NUM_PARAMS
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0)
gen = torch.Generator().manual_seed(1)
torch.cuda.reset_peak_memory_stats()
losses = []
for step in range(2):
    ids, labels = task.make_batch(2028, 64, generator=gen, device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = model(ids, labels=labels)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert torch.isfinite(norm)
    optimizer.step(); optimizer.zero_grad()
    losses.append(out.loss.item())
torch.cuda.synchronize()
save(HERE / 'fullsize_smoke.json', dict(passed=True, physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],
    num_layers=2, conv_kernel_size=4, batch_size=64, T=2028, sequence_length=2048, updates=2,
    losses=losses, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
    streaming_checks=streaming,
    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
    initial_weights_sha256=sha(INITIAL), base_initial_sha256=sha(BASE_INITIAL)))
