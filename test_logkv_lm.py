"""Tests for LogKVLM / LogKVBlock.

Run: uv run pytest test_logkv_lm.py -v
"""

import pytest
import torch

from configuration_logkv import LogKVConfig
from logkv import LogKVBlock
from logkv_lm import LogKVLM


@pytest.fixture
def config():
    return LogKVConfig(
        vocab_size=32,
        d_model=16,
        d_ff=32,
        chunk_size=4,
        num_layers=2,
        pad_token_id=None, bos_token_id=None, eos_token_id=None,
    )


@pytest.fixture
def model(config):
    torch.manual_seed(0)
    return LogKVLM(config)


class TestLogKVBlock:
    def test_forward_step_predict_equivalence_fp64(self):
        """block単位でもforward/step/predictが機械精度一致する"""
        torch.manual_seed(0)
        block = LogKVBlock(16, 4, 32).double().eval()
        x = torch.randn(2, 50, 16, dtype=torch.float64)
        with torch.no_grad():
            y_fwd = block(x)
            y1, h = block.step(x[:, :23])
            y2, h = block.step(x[:, 23:40], h)
            parts = [y1, y2]
            for t in range(40, 50):
                y, h = block.predict(x[:, t], h)
                parts.append(y.unsqueeze(1))
            y_seq = torch.cat(parts, dim=1)
        assert (y_seq - y_fwd).abs().max().item() < 1e-12

    def test_backward(self):
        torch.manual_seed(0)
        block = LogKVBlock(16, 4, 32)
        x = torch.randn(2, 30, 16, requires_grad=True)
        block(x).sum().backward()
        assert torch.isfinite(x.grad).all()
        for name, p in block.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name


class TestLogKVLM:
    @pytest.mark.parametrize("seq_len", [1, 4, 17, 100])
    def test_output_shape(self, model, config, seq_len):
        input_ids = torch.randint(0, config.vocab_size, (2, seq_len))
        out = model(input_ids)
        assert out.logits.shape == (2, seq_len, config.vocab_size)
        assert torch.isfinite(out.logits).all()

    def test_loss_computation(self, model, config):
        input_ids = torch.randint(0, config.vocab_size, (2, 20))
        out = model(input_ids, labels=input_ids.clone())
        assert out.loss is not None and torch.isfinite(out.loss)

    def test_no_loss_without_labels(self, model, config):
        input_ids = torch.randint(0, config.vocab_size, (2, 20))
        assert model(input_ids).loss is None

    def test_loss_all_pad_labels_no_nan(self, model, config):
        """全ラベル-100でもloss=0 (0/0=NaNガード)"""
        input_ids = torch.randint(0, config.vocab_size, (2, 20))
        labels = torch.full_like(input_ids, -100)
        out = model(input_ids, labels=labels)
        assert out.loss.item() == 0.0

    def test_gradient_flow(self, model, config):
        input_ids = torch.randint(0, config.vocab_size, (2, 20))
        out = model(input_ids, labels=input_ids.clone())
        out.loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name

    def test_step_split_consistency_fp64(self, model, config):
        """任意分割のstep連結logitsがforwardと機械精度一致"""
        model = model.double().eval()
        input_ids = torch.randint(0, config.vocab_size, (2, 60))
        with torch.no_grad():
            logits_fwd = model(input_ids).logits
            hidden = None
            parts = []
            pos = 0
            for n in [7, 1, 25, 27]:
                logits, hidden = model.step(input_ids[:, pos:pos + n], hidden)
                parts.append(logits)
                pos += n
            logits_step = torch.cat(parts, dim=1)
        assert (logits_step - logits_fwd).abs().max().item() < 1e-12

    def test_predict_matches_forward_fp64(self, model, config):
        """1トークンずつのpredictがforwardと機械精度一致"""
        model = model.double().eval()
        input_ids = torch.randint(0, config.vocab_size, (2, 50))
        with torch.no_grad():
            logits_fwd = model(input_ids).logits
            hidden = None
            parts = []
            for t in range(50):
                logits, hidden = model.predict(input_ids[:, t], hidden)
                parts.append(logits.unsqueeze(1))
            logits_pred = torch.cat(parts, dim=1)
        assert (logits_pred - logits_fwd).abs().max().item() < 1e-12

    def test_forward_with_past_key_values(self, model, config):
        """forwardのpast_key_values/use_cacheで継続できる(generate経路)"""
        model = model.eval()
        input_ids = torch.randint(0, config.vocab_size, (2, 30))
        with torch.no_grad():
            full = model(input_ids).logits
            out1 = model(input_ids[:, :20], use_cache=True)
            out2 = model(input_ids[:, 20:], past_key_values=out1.past_key_values,
                         use_cache=True)
        torch.testing.assert_close(out2.logits, full[:, 20:], atol=1e-5, rtol=1e-4)

    def test_save_and_load(self, model, config, tmp_path):
        model.save_pretrained(tmp_path / "m")
        loaded = LogKVLM.from_pretrained(tmp_path / "m")
        input_ids = torch.randint(0, config.vocab_size, (2, 20))
        with torch.no_grad():
            a = model.eval()(input_ids).logits
            b = loaded.eval()(input_ids).logits
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_generate_basic(self, model, config):
        input_ids = torch.randint(0, config.vocab_size, (1, 10))
        out = model.generate(input_ids, max_new_tokens=5, do_sample=False)
        assert out.shape == (1, 15)
        assert torch.equal(out[:, :10], input_ids)

    def test_generate_with_sampling(self, model, config):
        torch.manual_seed(0)
        input_ids = torch.randint(0, config.vocab_size, (1, 10))
        out = model.generate(input_ids, max_new_tokens=5, do_sample=True,
                             temperature=1.0, top_p=0.9)
        assert out.shape == (1, 15)

    def test_generate_uses_past_key_values(self, model, config):
        """greedy generateがstep+argmaxの逐次生成と一致する
        (past_key_valuesの持ち回りが正しいことの検証)"""
        model = model.eval()
        input_ids = torch.randint(0, config.vocab_size, (1, 10))
        out = model.generate(input_ids, max_new_tokens=5, do_sample=False)
        with torch.no_grad():
            logits, hidden = model.step(input_ids, None)
            toks = []
            tok = logits[:, -1].argmax(dim=-1)
            for _ in range(5):
                toks.append(tok)
                logits, hidden = model.predict(tok, hidden)
                tok = logits.argmax(dim=-1)
        manual = torch.stack(toks, dim=1)
        assert torch.equal(out[:, 10:], manual)
