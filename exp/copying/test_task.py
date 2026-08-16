"""Format tests for the Copy Memory Problem generator.

Run: uv run pytest exp/copying/test_task.py -v
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task import (  # noqa: E402
    BLANK, MARKER, MEMORY_LEN, MARKER_LEN, VOCAB_SIZE,
    make_batch, mask_non_answer, score_logits, seq_len_for,
)


@pytest.mark.parametrize("T", [1, 2, 5, 100, 2028])
def test_format(T):
    """入力: 先頭10がランダム1..8、次T-1が0、末尾11が9。長さT+20。"""
    g = torch.Generator().manual_seed(0)
    input_ids, labels = make_batch(T, batch_size=8, generator=g)

    L = seq_len_for(T)
    assert L == T + 20
    assert input_ids.shape == (8, L)
    assert labels.shape == (8, L)

    memory = input_ids[:, :MEMORY_LEN]
    assert ((memory >= 1) & (memory <= 8)).all()

    blanks = input_ids[:, MEMORY_LEN:T + MEMORY_LEN - 1]  # T-1 positions
    assert blanks.shape[1] == T - 1
    assert (blanks == BLANK).all()

    markers = input_ids[:, T + MEMORY_LEN - 1:]
    assert markers.shape[1] == MARKER_LEN
    assert (markers == MARKER).all()


@pytest.mark.parametrize("T", [1, 5, 100])
def test_target(T):
    """ターゲット: 先頭T+10が0、末尾10が入力の先頭10桁。"""
    g = torch.Generator().manual_seed(1)
    input_ids, labels = make_batch(T, batch_size=8, generator=g)

    assert (labels[:, :T + MEMORY_LEN] == BLANK).all()
    assert torch.equal(labels[:, -MEMORY_LEN:], input_ids[:, :MEMORY_LEN])


def test_vocab_range():
    g = torch.Generator().manual_seed(2)
    input_ids, labels = make_batch(50, batch_size=16, generator=g)
    for t in (input_ids, labels):
        assert t.min() >= 0 and t.max() < VOCAB_SIZE


def test_mask_non_answer():
    g = torch.Generator().manual_seed(3)
    input_ids, labels = make_batch(7, batch_size=4, generator=g)
    masked = mask_non_answer(labels)
    assert (masked[:, :-MEMORY_LEN] == -100).all()
    assert torch.equal(masked[:, -MEMORY_LEN:], input_ids[:, :MEMORY_LEN])
    # original labels untouched
    assert (labels[:, :7 + MEMORY_LEN] == BLANK).all()


def test_reproducible():
    a = make_batch(30, 8, generator=torch.Generator().manual_seed(42))
    b = make_batch(30, 8, generator=torch.Generator().manual_seed(42))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_score_logits():
    """完全予測でtoken/string accuracy 100%、1トークン誤りでstringが落ちる。"""
    g = torch.Generator().manual_seed(4)
    input_ids, labels = make_batch(10, batch_size=4, generator=g)
    L = input_ids.size(1)

    logits = torch.zeros(4, L, VOCAB_SIZE)
    logits.scatter_(2, labels.unsqueeze(-1), 10.0)  # perfect prediction
    tok, st, tok_n, st_n = score_logits(logits, input_ids)
    assert (tok, st, tok_n, st_n) == (40, 4, 40, 4)

    # corrupt one answer token of sample 0
    wrong = (input_ids[0, 0].item() % 8) + 1
    if wrong == input_ids[0, 0].item():
        wrong = wrong % 8 + 1
    logits[0, -MEMORY_LEN, :] = 0.0
    logits[0, -MEMORY_LEN, wrong] = 10.0
    tok, st, _, _ = score_logits(logits, input_ids)
    assert tok == 39 and st == 3
