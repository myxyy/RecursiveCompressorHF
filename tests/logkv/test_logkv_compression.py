"""Compression variants checked against an independent recursive span oracle."""

import copy
import math
from unittest.mock import patch

import pytest
import torch

from models.logkv.attention import LogKV


def span_reference(model, x):
    """Build only requested spans, without calling any LogKV hierarchy helpers.

    A node [start, end) is summarized using the original query at end (next)
    or end - 1 (last). Greedy base-C tiling independently partitions the past.
    """
    batch, length, dim = x.shape
    heads, width, chunk = model.num_heads, model.head_dim, model.chunk_size
    if model.phase_emb is not None:
        positions = torch.arange(length, device=x.device)
        phase = sum(model.phase_emb[level, (positions // chunk**level) % chunk]
                    for level in range(model.phase_levels))
        x = x + phase[None].to(x.dtype)

    def project(layer):
        return layer(x).reshape(batch, length, heads, width).transpose(1, 2)

    q, k, v = project(model.lq), project(model.lk), project(model.lv)
    if model.k_norm is not None:
        k = model.k_norm(k)
    if model.v_norm is not None:
        v = model.v_norm(v)
    nodes = {}

    def summarize(start, size):
        if size == 1:
            return k[:, :, start], v[:, :, start]
        if (start, size) not in nodes:
            children = [summarize(start + offset, size // chunk)
                        for offset in range(0, size, size // chunk)]
            keys = torch.stack([child[0] for child in children], dim=2)
            values = torch.stack([child[1] for child in children], dim=2)
            query_pos = start + size - (model.compression_query == 'last')
            query = q[:, :, query_pos]
            weights = torch.softmax((keys * query[:, :, None]).sum(-1) / math.sqrt(width), -1)
            compressed_k = (keys.mean(2) if model.compression_key == 'mean'
                            else (weights[..., None] * keys).sum(2))
            compressed_v = (weights[..., None] * values).sum(2)
            if model.k_norm is not None:
                compressed_k = model.k_norm(compressed_k)
            if model.v_norm is not None:
                compressed_v = model.v_norm(compressed_v)
            nodes[start, size] = compressed_k, compressed_v
        return nodes[start, size]

    outputs = []
    for position in range(length):
        slots = []
        start = 0
        while start < position:
            size, level = 1, 0
            while size * chunk <= position - start:
                size *= chunk
                level += 1
            key, value = summarize(start, size)
            score = (q[:, :, position] * key).sum(-1) / math.sqrt(width)
            if model.level_decay is None:
                score = score + model.level_sign * level * math.log(chunk)
            else:
                score = score - level * model.level_decay[None]
            slots.append((score, value))
            start += size
        if model.self_slot:
            score = (q[:, :, position] * k[:, :, position]).sum(-1) / math.sqrt(width)
            slots.append((score, v[:, :, position]))
        if slots:
            weights = torch.softmax(torch.stack([slot[0] for slot in slots], dim=-1), -1)
            values = torch.stack([slot[1] for slot in slots], dim=2)
            outputs.append((weights[..., None] * values).sum(2))
        else:
            outputs.append(q[:, :, position] * 0)
    out = torch.stack(outputs, dim=2) if outputs else q
    if model.lg is not None:
        out = out * torch.sigmoid(project(model.lg))
    return model.lo(out.transpose(1, 2).reshape(batch, length, dim))


def assert_state_equal(actual, expected):
    assert actual[1] == expected[1]
    assert len(actual[0]) == len(expected[0])
    for actual_level, expected_level in zip(actual[0], expected[0]):
        for a, e in zip(actual_level, expected_level):
            torch.testing.assert_close(a, e, atol=1e-12, rtol=1e-12)


def split_forward(model, x, cuts):
    hidden, outputs = None, []
    for start, end in zip(cuts, cuts[1:]):
        out, hidden = model.step(x[:, start:end], hidden)
        outputs.append(out)
    return torch.cat(outputs, dim=1), hidden


@pytest.mark.parametrize('query_mode', ['last', 'next'])
@pytest.mark.parametrize('key_mode', ['attention', 'mean'])
@pytest.mark.parametrize('chunk', [2, 3, 4])
@pytest.mark.parametrize('self_slot', [False, True])
def test_compression_modes_match_independent_spans(query_mode, key_mode, chunk, self_slot):
    torch.manual_seed(101)
    model = LogKV(8, chunk, num_heads=2, self_slot=self_slot, level_amplify=True,
                  compression_query=query_mode, compression_key=key_mode).double()
    length = chunk**3 + 3
    x = torch.randn(2, length, 8, dtype=torch.double)
    cuts = sorted({0, 1, chunk - 1, chunk, chunk + 1, chunk**2 - 1,
                   chunk**2, chunk**2 + 1, chunk**3 - 1, chunk**3,
                   chunk**3 + 1, length})
    with torch.no_grad():
        expected = span_reference(model, x)
        torch.testing.assert_close(model(x), expected, atol=1e-12, rtol=1e-12)
        segmented, _ = split_forward(model, x, cuts)
        torch.testing.assert_close(segmented, expected, atol=1e-12, rtol=1e-12)
        hidden, outputs = None, []
        with patch.object(model, 'step', side_effect=AssertionError('predict called step')):
            for token in x.unbind(1):
                out, hidden = model.predict(token, hidden)
                outputs.append(out[:, None])
        torch.testing.assert_close(torch.cat(outputs, dim=1), expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('key_mode', ['attention', 'mean'])
@pytest.mark.parametrize('options', [
    {}, {'no_level_bias': True}, {'kv_norm': True}, {'v_norm_only': True},
    {'phase_emb': True, 'phase_levels': 3, 'gated_attention': True,
     'learnable_decay': True},
])
def test_next_query_optional_features_and_gradients(key_mode, options):
    torch.manual_seed(102)
    model = LogKV(8, 4, num_heads=2, self_slot=True, compression_query='next',
                  compression_key=key_mode, **options).double()
    if model.level_decay is not None:
        with torch.no_grad():
            model.level_decay.copy_(torch.tensor([-0.4, 0.8], dtype=torch.double))
    x = torch.randn(2, 19, 8, dtype=torch.double, requires_grad=True)
    objective = torch.randn_like(x)
    targets = (x, *model.parameters())
    expected = span_reference(model, x)
    reference_grads = torch.autograd.grad((expected * objective).sum(), targets)
    actual = model(x)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    for output in [actual, split_forward(model, x, [0, 3, 4, 5, 15, 16, 17, 19])[0]]:
        gradients = torch.autograd.grad((output * objective).sum(), targets)
        for actual_grad, reference_grad in zip(gradients, reference_grads):
            torch.testing.assert_close(actual_grad, reference_grad, atol=1e-10, rtol=1e-10)
    hidden, outputs = None, []
    for token in x.unbind(1):
        out, hidden = model.predict(token, hidden)
        outputs.append(out[:, None])
    gradients = torch.autograd.grad((torch.cat(outputs, 1) * objective).sum(), targets)
    for actual_grad, reference_grad in zip(gradients, reference_grads):
        torch.testing.assert_close(actual_grad, reference_grad, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('chunk', [2, 3, 4])
@pytest.mark.parametrize('key_mode', ['attention', 'mean'])
def test_next_query_pending_state_and_mixed_execution(chunk, key_mode):
    torch.manual_seed(103)
    model = LogKV(8, chunk, num_heads=2, self_slot=True, compression_query='next',
                  compression_key=key_mode).double().eval()
    length = chunk**3 + 3
    x = torch.randn(2, length, 8, dtype=torch.double)
    with torch.no_grad():
        expected = span_reference(model, x)
        hidden = None
        for position in range(length):
            out, hidden = model.predict(x[:, position], hidden)
            torch.testing.assert_close(out, expected[:, position], atol=1e-12, rtol=1e-12)
            count = position + 1
            assert hidden[1] == count
            assert len(hidden[0]) <= math.ceil(math.log(count, chunk)) + 1
            for level, tensors in enumerate(hidden[0]):
                expected_count = ((count - 1) % chunk + 1 if level == 0
                                  else ((count - 1) // chunk**level) % chunk)
                for tensor in tensors:
                    assert tensor.size(1) == expected_count
                    assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
        for boundary in [chunk, chunk**2, chunk**3]:
            _, prefix = model.step(x[:, :boundary])
            snapshot = copy.deepcopy(prefix)
            actual, after_predict = model.predict(x[:, boundary], prefix)
            stepped, after_step = model.step(x[:, boundary:boundary + 1], prefix)
            torch.testing.assert_close(actual, expected[:, boundary], atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(actual, stepped[:, 0], atol=1e-12, rtol=1e-12)
            assert_state_equal(after_predict, after_step)
            assert_state_equal(prefix, snapshot)
            tail, _ = model.step(x[:, boundary + 1:], after_predict)
            torch.testing.assert_close(tail, expected[:, boundary + 1:], atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('self_slot', [False, True])
@pytest.mark.parametrize('key_mode', ['attention', 'mean'])
def test_next_query_does_not_leak_future_boundary_queries(self_slot, key_mode):
    torch.manual_seed(104)
    model = LogKV(8, 4, num_heads=2, self_slot=self_slot, compression_query='next',
                  compression_key=key_mode, phase_emb=True, phase_levels=3).double()
    x = torch.randn(1, 67, 8, dtype=torch.double)
    with torch.no_grad():
        original = model(x)
        for boundary in [1, 3, 4, 5, 15, 16, 17, 63, 64, 65]:
            changed = x.clone()
            changed[:, boundary:] = torch.randn_like(changed[:, boundary:]) * 5
            torch.testing.assert_close(model(changed)[:, :boundary], original[:, :boundary],
                                       atol=1e-12, rtol=1e-12)
            # A prefix with no boundary query must not compress its final chunk.
            torch.testing.assert_close(model(x[:, :boundary]), original[:, :boundary],
                                       atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('key_mode', ['attention', 'mean'])
@pytest.mark.parametrize('recompute', [False, True])
def test_next_query_training_keeps_grouped_forward(key_mode, recompute):
    model = LogKV(8, 4, num_heads=2, self_slot=True, compression_query='next',
                  compression_key=key_mode).double()
    model.recompute_attention = recompute
    x = torch.randn(2, 65, 8, dtype=torch.double, requires_grad=True)
    with patch.object(model, 'step', side_effect=AssertionError('streaming forward')), \
         patch.object(model, '_attend_contexts', side_effect=AssertionError('streaming gather')), \
         patch.object(model, '_level_attention', side_effect=AssertionError('per-query gather')):
        model(x).square().sum().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize('query_mode', ['last', 'next'])
def test_mean_key_amplification_uniform_scores_are_raw_token_average(query_mode):
    torch.manual_seed(105)
    model = LogKV(4, 4, self_slot=True, compression_query=query_mode,
                  compression_key='mean', level_amplify=True).double()
    with torch.no_grad():
        model.lq.weight.zero_()
        model.lv.weight.copy_(torch.eye(4, dtype=torch.double))
        model.lo.weight.copy_(torch.eye(4, dtype=torch.double))
        x = torch.randn(1, 257, 4, dtype=torch.double)
        expected = x.cumsum(1) / torch.arange(1, 258, dtype=torch.double)[None, :, None]
        torch.testing.assert_close(model(x), expected, atol=1e-12, rtol=1e-12)


def test_next_mean_boundary_matches_dense_attention_through_first_order():
    model = LogKV(3, 4, self_slot=True, compression_query='next',
                  compression_key='mean', level_amplify=True).double()
    keys = torch.tensor([1.4, -0.8, 0.2, -0.5, 0.3], dtype=torch.double)
    values = torch.tensor([-1., .5, 2., -.75, 1.25], dtype=torch.double)
    with torch.no_grad():
        for projection in [model.lq, model.lk, model.lv]:
            projection.weight.zero_()
        model.lq.weight[0, 0] = 1
        model.lk.weight[0, 1] = math.sqrt(3)
        model.lv.weight[0, 2] = 1
        model.lo.weight.copy_(torch.eye(3, dtype=torch.double))
        errors = []
        for epsilon in [.02, .01]:
            x = torch.stack([torch.full_like(keys, epsilon), keys, values], dim=-1)[None]
            dense = torch.dot(torch.softmax(epsilon * keys, dim=0), values)
            actual = model(x)[0, 4, 0]
            errors.append((actual - dense).abs().item())
        assert errors[0] > 1e-6
        assert 3.9 < errors[0] / errors[1] < 4.2


@pytest.mark.parametrize('self_slot', [False, True])
def test_next_query_empty_sequence(self_slot):
    model = LogKV(8, 4, num_heads=2, self_slot=self_slot,
                  compression_query='next', compression_key='mean')
    x = torch.empty(2, 0, 8)
    torch.testing.assert_close(model(x), model.step(x)[0])
