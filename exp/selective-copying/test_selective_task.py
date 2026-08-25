"""Format tests for the Selective Copying generator.

Run: uv run pytest exp/selective-copying/test_task.py -v
"""

import importlib.util
from pathlib import Path

import pytest
import torch

# Load our task.py under a distinct module name so this test file does not
# poison sys.modules["task"] for exp/copying/test_task.py in the same session.
_spec = importlib.util.spec_from_file_location(
    "selective_task", Path(__file__).resolve().parent / "task.py")
_task = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_task)

BLANK = _task.BLANK
MARKER = _task.MARKER
MARKER_LEN = _task.MARKER_LEN
MEMORY_LEN = _task.MEMORY_LEN
VOCAB_SIZE = _task.VOCAB_SIZE
make_batch = _task.make_batch
mask_non_answer = _task.mask_non_answer
seq_len_for = _task.seq_len_for


@pytest.mark.parametrize("T", [1, 2, 5, 100, 2028])
def test_format(T):
    """入力: 先頭T+9にランダム位置の10桁(1..8)+blank、末尾11が9。長さT+20。"""
    g = torch.Generator().manual_seed(0)
    input_ids, labels = make_batch(T, batch_size=8, generator=g)

    L = seq_len_for(T)
    assert L == T + 20
    assert input_ids.shape == (8, L)

    data_region = input_ids[:, :L - MARKER_LEN]  # T+9 positions
    assert data_region.shape[1] == T + 9
    # exactly 10 data digits (1..8), rest blank
    is_data = (data_region >= 1) & (data_region <= 8)
    assert (is_data.sum(dim=1) == MEMORY_LEN).all()
    assert ((data_region == BLANK) | is_data).all()

    markers = input_ids[:, L - MARKER_LEN:]
    assert (markers == MARKER).all()


@pytest.mark.parametrize("T", [1, 5, 100])
def test_target_appearance_order(T):
    """ターゲット末尾10桁 = データ領域の1..8を出現順に並べたもの。"""
    g = torch.Generator().manual_seed(1)
    input_ids, labels = make_batch(T, batch_size=8, generator=g)
    L = input_ids.size(1)

    assert (labels[:, :T + MEMORY_LEN] == BLANK).all()
    for b in range(8):
        row = input_ids[b, :L - MARKER_LEN]
        appeared = row[(row >= 1) & (row <= 8)]
        assert torch.equal(labels[b, -MEMORY_LEN:], appeared)


def test_positions_are_random():
    """散らばり: 複数サンプルでデータ位置が固定でない(=Copyingと異なる)。"""
    g = torch.Generator().manual_seed(2)
    input_ids, _ = make_batch(100, batch_size=32, generator=g)
    data_mask = (input_ids >= 1) & (input_ids <= 8)
    # 位置パターンが全サンプルで同一なら選択性がない
    assert not (data_mask == data_mask[0]).all()
    # 先頭10連続に固まっているサンプルばかりではない
    front_packed = data_mask[:, :MEMORY_LEN].all(dim=1)
    assert front_packed.sum() < 32


def test_t1_reduces_to_copying_layout():
    """T=1はdata_region=10なので全部詰まる(Copyingと同じ配置)。"""
    g = torch.Generator().manual_seed(3)
    input_ids, labels = make_batch(1, batch_size=4, generator=g)
    memory = input_ids[:, :MEMORY_LEN]
    assert ((memory >= 1) & (memory <= 8)).all()
    assert torch.equal(labels[:, -MEMORY_LEN:], memory)


def test_mask_and_reproducibility():
    g = torch.Generator().manual_seed(4)
    input_ids, labels = make_batch(7, batch_size=4, generator=g)
    masked = mask_non_answer(labels)
    assert (masked[:, :-MEMORY_LEN] == -100).all()
    assert torch.equal(masked[:, -MEMORY_LEN:], labels[:, -MEMORY_LEN:])

    a = make_batch(30, 8, generator=torch.Generator().manual_seed(42))
    b = make_batch(30, 8, generator=torch.Generator().manual_seed(42))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    for t in (input_ids, labels):
        assert t.min() >= 0 and t.max() < VOCAB_SIZE
