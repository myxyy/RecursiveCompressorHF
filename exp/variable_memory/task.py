"""Variable payload length and blank prefix; position-aligned copying targets."""
import torch


def make_batch(task, memory_len, horizon, prefix, batch_size, generator, device=None):
    if task not in ("copying", "selective-copying"):
        raise ValueError(task)
    if memory_len < 1 or horizon < 1 or prefix < 0 or batch_size < 1:
        raise ValueError("positive memory/horizon/batch and nonnegative prefix required")
    M, T, P = memory_len, horizon, prefix
    length = P + T + 2 * M
    memory = torch.randint(1, 9, (batch_size, M), generator=generator)
    inputs = torch.zeros(batch_size, length, dtype=torch.long)
    if task == "copying":
        inputs[:, P:P + M] = memory
    else:
        positions = torch.rand(batch_size, T + M - 1, generator=generator)
        positions = positions.topk(M, dim=1, largest=False).indices.sort(dim=1).values + P
        inputs.scatter_(1, positions, memory)
    inputs[:, P + T + M - 1:] = 9  # M+1 identical markers, no answer feedback
    labels = torch.zeros_like(inputs)
    labels[:, -M:] = memory
    if device is not None:
        inputs, labels = inputs.to(device), labels.to(device)
    return inputs, labels


def score(logits, labels, memory_len):
    correct = logits[:, -memory_len:].argmax(-1) == labels[:, -memory_len:]
    return int(correct.sum()), int(correct.all(-1).sum()), correct.numel(), correct.shape[0]
