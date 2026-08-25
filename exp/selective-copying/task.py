"""
Selective Copying task (Mamba, arXiv:2312.00752 Fig. 2 — scattered variant of
the Copy Memory Problem).

Identical to exp/copying except the 10 data digits are SCATTERED at random
positions inside the first T+9 tokens instead of sitting at the start:

Input  (length T+20): [T+9 tokens: blanks with 10 random positions holding
                       random digits 1-8] [11 nines]
Target (length T+20): [T+10 zeros] [the 10 data digits in order of appearance]

Unlike plain copying, constant-offset (time-only) solutions do not exist —
the model must select content-dependently. Scoring/masking helpers and the
vocabulary are shared with the copying task.
"""

import importlib.util
from pathlib import Path

import torch

# Load the copying task under a distinct module name (a plain `from task
# import ...` would hit THIS module via sys.modules["task"] -> circular).
_spec = importlib.util.spec_from_file_location(
    "copying_task", Path(__file__).resolve().parents[1] / "copying" / "task.py")
_copying = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_copying)

BLANK = _copying.BLANK
MARKER = _copying.MARKER
MARKER_LEN = _copying.MARKER_LEN
MEMORY_LEN = _copying.MEMORY_LEN
VOCAB_SIZE = _copying.VOCAB_SIZE
mask_non_answer = _copying.mask_non_answer
score_logits = _copying.score_logits
seq_len_for = _copying.seq_len_for

TASK_NAME = "selective-copying"  # run artifacts go to $DATA_DIR/exp/{TASK_NAME}/


def make_batch(T, batch_size, generator=None, device=None):
    """Generate one batch of the selective copy task at horizon T.

    Returns (input_ids, labels), both (batch_size, T+20) int64 tensors.
    The 10 data digits are placed at sorted random distinct positions within
    the first T+9 tokens, so the target (appearance order) is the memory
    vector itself."""
    assert T >= 1, f"T must be >= 1, got {T}"
    L = seq_len_for(T)
    data_region = L - MARKER_LEN  # T + 9 positions that may hold data

    memory = torch.randint(1, 9, (batch_size, MEMORY_LEN), generator=generator)

    # 10 distinct random positions per sample (sorted = appearance order)
    scores = torch.rand(batch_size, data_region, generator=generator)
    pos = scores.topk(MEMORY_LEN, dim=1, largest=False).indices
    pos, _ = pos.sort(dim=1)

    input_ids = torch.zeros(batch_size, L, dtype=torch.long)
    input_ids[:, data_region:] = MARKER
    input_ids.scatter_(1, pos, memory)

    labels = torch.zeros(batch_size, L, dtype=torch.long)
    labels[:, -MEMORY_LEN:] = memory

    if device is not None:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
    return input_ids, labels
