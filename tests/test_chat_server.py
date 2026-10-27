"""CPU chat decoding checks with real models; no checkpoints or downloads."""
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from inference import chat_server as chat
from models.logkv.configuration import LogKVConfig
from models.logkv.modeling import LogKVLM
from models.recursive_compressor.configuration import RecursiveCompressorConfig
from models.recursive_compressor.modeling import RecursiveCompressorLM


ARCHITECTURES = ['recursive_compressor', 'logkv', 'logkv_zero_bias',
                 'mamba2', 'mamba2_logkv', 'mamba2_logkv_gated']


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        assert text.startswith('[INST]') and text.endswith('[/INST]')
        return [4, 5, 6, 7, 8, 9, 10 if 'first' in text else 11]


@pytest.fixture(params=ARCHITECTURES)
def model(request):
    torch.manual_seed(17)
    common = dict(vocab_size=16, d_model=32, num_layers=2)
    logkv = dict(num_heads=2, d_ff=64, chunk_size=4, conv_kernel_size=4,
                 gated_attention=True, self_slot=True, phase_emb=False)
    arch = request.param
    if arch == 'recursive_compressor':
        return RecursiveCompressorLM(RecursiveCompressorConfig(
            **common, num_heads=2, d_ff=64, chunk_size=4, compress_size=1))
    if arch.startswith('logkv'):
        return LogKVLM(LogKVConfig(**common, **logkv, no_level_bias=arch == 'logkv_zero_bias'))

    pytest.importorskip('mamba_ssm')
    from models.mamba2.configuration import Mamba2Config
    from models.mamba2.modeling import Mamba2LM
    from models.mamba2_logkv.configuration import Mamba2LogKVConfig
    from models.mamba2_logkv.modeling import Mamba2LogKVLM
    from models.mamba2_logkv_gated.configuration import GatedMambaLogKVConfig
    from models.mamba2_logkv_gated.modeling import GatedMambaLogKVLM
    mamba = dict(d_state=8, headdim=16, scan_chunk_size=16)
    if arch == 'mamba2':
        return Mamba2LM(Mamba2Config(**common, **mamba))
    if arch == 'mamba2_logkv':
        logkv['conv_kernel_size'] = 0
        return Mamba2LogKVLM(Mamba2LogKVConfig(
            **common, **mamba, **logkv, num_logkv_layers=2))
    model = GatedMambaLogKVLM(GatedMambaLogKVConfig(
        **common, **mamba, **logkv, num_mamba_layers=2))
    with torch.no_grad():
        model.mamba_gate.fill_(0.1)  # Exercise both branches, not just the zero-gate baseline.
    return model


@pytest.fixture(params=[torch.float32, torch.bfloat16])
def server(model, request, monkeypatch):
    state = SimpleNamespace(model=model.to(dtype=request.param).eval(), tokenizer=Tokenizer(),
                            device=torch.device('cpu'), bos_id=2, eos_id=3)
    monkeypatch.setattr(chat, 'state', state)
    return state


@pytest.mark.parametrize('ending', ['eos', 'max_tokens'])
@torch.no_grad()
def test_two_turns_match_step_reference(server, ending, monkeypatch):
    monkeypatch.setattr(chat, 'MAX_NEW_TOKENS_PER_TURN', 3 if ending == 'eos' else 2)
    hidden, reference = None, None
    penalty = torch.zeros(16)
    for query in ['first', 'second']:
        # Replay the same tokens through the old all-step path as an independent
        # reference for logits, EOS state, and the next turn's prompt processing.
        prompt = [server.bos_id] + server.tokenizer.encode(f'[INST]{query}[/INST]')
        logits, reference = server.model.step(torch.tensor([prompt]), reference)
        expected_logits = [logits[0, -1]]
        for token in [4, 5, server.eos_id]:
            logits, reference = server.model.step(torch.tensor([[token]]), reference)
            expected_logits.append(logits[0, -1])
        scripted = iter([4, 5, server.eos_id])
        sampled_logits, emitted = [], []

        def sample(logits, temperature, top_p, penalties):
            assert logits.shape == (16,)
            sampled_logits.append(logits.clone())
            return next(scripted)

        with patch.object(chat, '_sample_token', side_effect=sample), \
             patch.object(server.model, 'predict', wraps=server.model.predict) as predict:
            hidden, reason, count = chat._generate_turn(
                hidden, query, 1., .95, emitted.append,
                penalty=penalty, penalty_add=.5, penalty_decay=.9)
        assert (reason, count, emitted) == (ending, 2, [4, 5])
        assert [call.args[0].tolist() for call in predict.call_args_list] == [[4], [5], [3]]
        tolerance = dict(atol=0.05, rtol=0.05) if next(server.model.parameters()).dtype == torch.bfloat16 \
            else dict(atol=1e-5, rtol=1e-4)
        for actual, expected in zip(sampled_logits, expected_logits):
            torch.testing.assert_close(actual, expected, **tolerance)
        torch.testing.assert_close(hidden, reference, **tolerance)
    assert penalty[server.eos_id] == 0
    assert penalty[4] > 0 and penalty[5] > 0


@pytest.mark.parametrize('before_prompt', [True, False])
def test_cancel_does_not_append_eos(server, before_prompt):
    cancel = threading.Event()
    if before_prompt:
        cancel.set()

    def on_token(token):
        cancel.set()

    with patch.object(chat, '_sample_token', return_value=4), \
         patch.object(server.model, 'predict', wraps=server.model.predict) as predict:
        hidden, reason, count = chat._generate_turn(
            None, 'first', 0., 1., on_token, cancel_event=cancel)
    assert reason == 'cancelled'
    assert count == (0 if before_prompt else 1)
    assert [call.args[0].tolist() for call in predict.call_args_list] == ([] if before_prompt else [[4]])
    if before_prompt:
        assert hidden is None
