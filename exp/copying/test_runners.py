"""Both fixed-length tasks can share runners without global import overrides."""

import pytest
import torch

from exp.copying import evaluate, task as copying_task, train
from exp.selective_copying import task as selective_task
from exp.selective_copying import evaluate as selective_evaluate
from exp.selective_copying import train as selective_train


@pytest.mark.parametrize("task", [copying_task, selective_task])
def test_shared_evaluation_uses_requested_task(task):
    expected_ids, labels = task.make_batch(
        31, 3, generator=torch.Generator().manual_seed(19))

    class Oracle(torch.nn.Module):
        def step(self, ids, hidden):
            # Fails if a runner silently generates the other task's inputs.
            torch.testing.assert_close(ids, expected_ids)
            return torch.nn.functional.one_hot(labels, task.VOCAB_SIZE).float(), None

    model = Oracle()
    assert evaluate.eval_horizon(
        model, 31, 3, torch.Generator().manual_seed(19),
        torch.device("cpu"), False, task=task) == (1.0, 1.0)
    assert train.quick_eval(
        model, torch.device("cpu"), [31], 3,
        torch.Generator().manual_seed(19), torch.bfloat16, task=task) == {31: 1.0}


@pytest.mark.parametrize("entry", [selective_train, selective_evaluate])
def test_selective_entry_passes_task_without_rebinding_copying(entry, monkeypatch):
    received = []
    monkeypatch.setattr(entry, "run", lambda *, task: received.append(task))
    entry.main()
    assert received == [selective_task]
    assert train.copying_task is copying_task
    assert evaluate.copying_task is copying_task
    assert selective_task.TASK_NAME == "selective-copying"
