"""Evaluate a trained sion_test LogKV model (state_dict saved by sion_test_diag.py) on long
contexts. Same metric as evaluate_needle_in_a_haystack, but (a) trials are sub-batched under a
token budget and (b) norm/head are applied to the last position only (the prediction is
identical; the wrapper's full-sequence logits (B, L, 16000) would OOM at long L).
Usage: uv run python sion_test_eval.py CKPT TRIALS L1,L2,..."""
import sys, torch
from sion_test import LanguageModel, generate_needle_batch
from sion_test_diag import LogKVBackbone, VOCAB, HID

device = "cuda"
ckpt, trials = sys.argv[1], int(sys.argv[2])
lengths = [int(v) for v in sys.argv[3].split(",")]
lm = LanguageModel(VOCAB, HID, LogKVBackbone(HID)).to(device)
lm.load_state_dict(torch.load(ckpt, map_location=device))
lm.eval()
CHUNK = 4096            # step() feed size: memory ~ batch x CHUNK regardless of length
TOKEN_BUDGET = 2 ** 18  # batch x CHUNK per forward
torch.manual_seed(12345)

def last_hidden(x):
    """Backbone output at the last position via chunked step() with hidden carry-over
    (numerically equivalent to the one-shot forward; see test_logkv.py)."""
    h = None
    hidden = [None] * len(lm.backbone.layers)
    for i in range(0, x.size(1), CHUNK):
        h = lm.tok_embeddings(x[:, i:i + CHUNK])
        for j, layer in enumerate(lm.backbone.layers):
            h, hidden[j] = layer.step(h, hidden[j])
    return h[:, -1]

print("=" * 60)
with torch.no_grad():
    for length in lengths:
        bs = max(1, TOKEN_BUDGET // min(length, CHUNK))
        correct = done = 0
        while done < trials:
            b = min(bs, trials - done)
            x, y = generate_needle_batch(b, length, VOCAB, device)
            pred = lm.lm_head(lm.rms_norm(last_hidden(x))).argmax(-1)
            correct += (pred == y).sum().item()
            done += b
        print(f"Context Length: {length:6d} tokens | Retrieval Acc: {100*correct/trials:6.2f}% ({correct}/{trials})", flush=True)
print("=" * 60)
