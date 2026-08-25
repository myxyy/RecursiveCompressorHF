"""
Copy Memory Problem (CKConv / arXiv:2102.02611 formulation).

Vocabulary: digits 0-9 (vocab size 10).
  1-8 : data symbols
  0   : blank
  9   : delimiter/marker

Input  (length T+20): [10 random digits in 1..8] [T-1 zeros] [11 nines]
Target (length T+20): [T+10 zeros] [the first 10 input digits]

The model is trained seq2seq-style with POSITION-ALIGNED targets
(RecursiveCompressorLM.forward compares logits and labels at the same
position, so no shifting is applied).
"""

import torch

TASK_NAME = "copying"  # run artifacts go to $DATA_DIR/exp/{TASK_NAME}/
VOCAB_SIZE = 10
BLANK = 0
MARKER = 9
MEMORY_LEN = 10   # number of digits to memorize
MARKER_LEN = 11   # trailing run of 9s
EXTRA_LEN = MEMORY_LEN + MARKER_LEN - 1  # sequence length is T + 20


def seq_len_for(T):
    return T + MEMORY_LEN + MARKER_LEN - 1  # = T + 20


def make_batch(T, batch_size, generator=None, device=None):
    """Generate one batch of the copy task at horizon T (all samples share T).

    Returns (input_ids, labels), both (batch_size, T+20) int64 tensors.
    labels are position-aligned (loss over ALL positions; blanks included,
    matching the CKConv formulation)."""
    assert T >= 1, f"T must be >= 1, got {T}"
    L = seq_len_for(T)

    memory = torch.randint(1, 9, (batch_size, MEMORY_LEN), generator=generator)

    input_ids = torch.zeros(batch_size, L, dtype=torch.long)
    input_ids[:, :MEMORY_LEN] = memory
    input_ids[:, T + MEMORY_LEN - 1:] = MARKER  # last 11 positions

    labels = torch.zeros(batch_size, L, dtype=torch.long)
    labels[:, -MEMORY_LEN:] = memory

    if device is not None:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
    return input_ids, labels


def mask_non_answer(labels):
    """Return a copy of labels with loss restricted to the last MEMORY_LEN
    positions (everything else -100). Optional variant; the default protocol
    computes loss on all positions like prior work."""
    masked = torch.full_like(labels, -100)
    masked[:, -MEMORY_LEN:] = labels[:, -MEMORY_LEN:]
    return masked


@torch.no_grad()
def score_logits(logits, labels):
    """Token / string accuracy on the answer region (last MEMORY_LEN positions).

    logits: (B, L, vocab) position-aligned model output
    labels: (B, L) position-aligned targets (answer = last MEMORY_LEN columns;
            works for both copying and selective-copying, and is unaffected by
            mask_non_answer, which only masks earlier positions)

    Returns (token_correct, string_correct, total_tokens, total_strings)."""
    pred = logits[:, -MEMORY_LEN:, :].argmax(dim=-1)      # (B, 10)
    target = labels[:, -MEMORY_LEN:]                      # (B, 10)
    tok = (pred == target)
    token_correct = int(tok.sum().item())
    string_correct = int(tok.all(dim=1).sum().item())
    batch = labels.size(0)
    return token_correct, string_correct, batch * MEMORY_LEN, batch
