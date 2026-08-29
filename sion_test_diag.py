"""Usage: uv run python sion_test_diag.py {asis|emb002} STEPS [TRIALS] [L1,L2,...]
Diagnostics for sion_test.py with the LogKV backbone (friend's script is left untouched).
Variants: (1) initial logit scale, (2) embedding init std=0.02 only, (3) as-is init but more steps."""
import sys, math, torch, torch.nn as nn
from sion_test import LanguageModel, generate_needle_batch, train_needle_retrieval, evaluate_needle_in_a_haystack
from logkv import LogKVBlock

class LogKVBackbone(nn.Module):
    def __init__(self, dim, num_layers=2, num_heads=4, d_ff=512, chunk_size=4):
        super().__init__()
        self.layers = nn.ModuleList([LogKVBlock(dim, chunk_size, d_ff, num_heads, phase_emb=True,
                                                phase_levels=2, gated_attention=True) for _ in range(num_layers)])
    def forward(self, x):
        for layer in self.layers: x = layer(x)
        return x

VOCAB, HID, device = 16000, 256, "cuda"

if __name__ != "__main__":
    sys.exit  # imported for LogKVBackbone/VOCAB/HID only; the script body below is skipped
else:
  variant, steps = sys.argv[1], int(sys.argv[2])
  trials = int(sys.argv[3]) if len(sys.argv) > 3 else 20
  lengths = [int(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4 else [512, 1024, 2048, 4096, 8192]
  torch.manual_seed(0)
  lm = LanguageModel(VOCAB, HID, LogKVBackbone(HID)).to(device)
  if variant == "emb002":
      nn.init.normal_(lm.tok_embeddings.weight, std=0.02)   # tied with lm_head
  with torch.no_grad():
      x, y = generate_needle_batch(8, 512, VOCAB, device)
      lg = lm(x)[:, -1]
      print(f"[{variant}] 初期 logit std {lg.std():.2f}, 初期 CE {nn.functional.cross_entropy(lg, y):.1f} (一様なら {math.log(VOCAB):.1f})")
  train_needle_retrieval(lm, VOCAB, device, steps=steps, batch_size=16, h_dim=HID)
  torch.save(lm.state_dict(), f"/mnt/raid0/RecursiveCompressor/exp/sion_test_{variant}_{steps}.pt")

  # Same metric as evaluate_needle_in_a_haystack, but trials are processed in
  # sub-batches under a token budget so long contexts do not OOM.
  lm.eval()
  TOKEN_BUDGET = 2 ** 20
  print("=" * 60)
  with torch.no_grad():
      for length in lengths:
          bs = max(1, TOKEN_BUDGET // length)
          correct = done = 0
          while done < trials:
              b = min(bs, trials - done)
              x, y = generate_needle_batch(b, length, VOCAB, device)
              correct += (lm(x)[:, -1].argmax(-1) == y).sum().item()
              done += b
          print(f"Context Length: {length:6d} tokens | Retrieval Acc: {100*correct/trials:6.2f}% ({correct}/{trials})", flush=True)
  print("=" * 60)
