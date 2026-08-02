import json
import numpy as np
import pytest
import torch
from configuration_recursive_compressor import RecursiveCompressorConfig
from recursive_compressor_lm import RecursiveCompressorLM
from recursive_compressor_lm_pipeline import RecursiveCompressorLMPipelineStage
from dataset import (
    _extract_turns_sharegpt, _extract_turns_messages,
    _build_memmap_packed, _build_memmap_conversations,
    _units_doc_item, _turns_sharegpt_item, _turns_messages_item,
    _conversation_to_ids_and_mask, _conversation_to_samples,
    _text_to_chunks, _pack_chunks, MemmapDataset,
)


class TestRecursiveCompressorLM:
    @pytest.fixture
    def config(self):
        return RecursiveCompressorConfig(
            vocab_size=100,
            d_model=64,
            num_heads=4,
            d_ff=128,
            chunk_size=8,
            compress_size=4,
            num_layers=2,
        )

    @pytest.fixture
    def model(self, config):
        return RecursiveCompressorLM(config)

    def test_output_shape(self, model, config):
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        output = model(input_ids)
        assert output.logits.shape == (batch_size, seq_len, config.vocab_size)

    def test_output_shape_non_divisible_seq_len(self, model, config):
        """chunk_sizeで割り切れないシーケンス長でも正しく動作する"""
        batch_size, seq_len = 2, 30
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        output = model(input_ids)
        assert output.logits.shape == (batch_size, seq_len, config.vocab_size)

    def test_single_chunk(self, model, config):
        """チャンクが1つだけの場合（再帰なし）"""
        batch_size = 2
        seq_len = config.chunk_size
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        output = model(input_ids)
        assert output.logits.shape == (batch_size, seq_len, config.vocab_size)

    def test_loss_computation(self, model, config):
        """labelsを渡すとlossが計算される"""
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        output = model(input_ids, labels=labels)
        assert output.loss is not None
        assert output.loss.item() > 0
        output.loss.backward()

    def test_no_loss_without_labels(self, model, config):
        """labelsなしではlossはNone"""
        input_ids = torch.randint(0, config.vocab_size, (2, 32))
        output = model(input_ids)
        assert output.loss is None

    def test_gradient_flow(self, model, config):
        """全パラメータに勾配が流れる"""
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        output = model(input_ids, labels=labels)
        output.loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_num_layers(self, config):
        """レイヤー数が正しい"""
        model = RecursiveCompressorLM(config)
        assert len(model.layers) == config.num_layers

    def test_loss_all_pad_labels_no_nan(self, model, config):
        """labelsが全て-100でもlossがNaNにならない（0/0防止）"""
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
        output = model(input_ids, labels=labels)
        assert output.loss is not None
        assert not torch.isnan(output.loss).item()
        assert output.loss.item() == 0.0
        # Backward should produce zero (not NaN) gradients
        output.loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any().item(), f"NaN grad in {name}"

    def test_save_and_load(self, model, config, tmp_path):
        """save_pretrained / from_pretrained の動作確認"""
        model.save_pretrained(tmp_path)
        loaded = RecursiveCompressorLM.from_pretrained(tmp_path)
        assert loaded.config.d_model == config.d_model
        assert loaded.config.num_layers == config.num_layers

        input_ids = torch.randint(0, config.vocab_size, (1, 16))
        model.eval()
        loaded.eval()
        with torch.no_grad():
            orig = model(input_ids).logits
            reloaded = loaded(input_ids).logits
        torch.testing.assert_close(orig, reloaded)

    def test_generate_basic(self, model, config):
        """HuggingFace generate() で出力が得られる"""
        model.eval()
        prompt = torch.randint(0, config.vocab_size, (1, 8))
        out = model.generate(prompt, max_new_tokens=10, do_sample=False)
        # output is prompt + new tokens
        assert out.shape == (1, 18)
        # The prompt portion should be preserved
        assert torch.equal(out[0, :8], prompt[0])

    def test_generate_uses_past_key_values(self, model, config):
        """generate が past_key_values を使い回しても、毎回フルforwardと同じ結果"""
        model.eval()
        prompt = torch.randint(0, config.vocab_size, (1, 8))

        # Generate with caching (default). Force a fixed length by disabling EOS
        # stopping; otherwise a random small model often samples EOS early.
        with_cache = model.generate(
            prompt, max_new_tokens=8, min_new_tokens=8,
            do_sample=False, eos_token_id=None,
        )

        # Manually generate without caching: each step recomputes from scratch
        ids = prompt.clone()
        for _ in range(8):
            with torch.no_grad():
                logits = model(ids, use_cache=False).logits
            next_token = logits[:, -1, :].argmax(-1, keepdim=True)
            ids = torch.cat([ids, next_token], dim=1)

        torch.testing.assert_close(with_cache, ids)

    def test_generate_with_sampling(self, model, config):
        """sampling パラメータが動く"""
        model.eval()
        prompt = torch.randint(0, config.vocab_size, (1, 4))
        torch.manual_seed(0)
        out = model.generate(
            prompt, max_new_tokens=8,
            do_sample=True, temperature=0.8, top_p=0.9,
        )
        assert out.shape == (1, 12)

    @pytest.mark.parametrize("seq_len", [1, 7, 8, 16, 24, 32])
    def test_predict_matches_forward(self, config, seq_len):
        """predictを1トークンずつ呼んだ結果がforwardと一致する"""
        model = RecursiveCompressorLM(config)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (1, seq_len))

        with torch.no_grad():
            forward_logits = model(input_ids).logits

            hidden = None
            predict_logits_list = []
            for t in range(seq_len):
                token = input_ids[:, t]
                logits, hidden = model.predict(token, hidden)
                predict_logits_list.append(logits)
            predict_logits = torch.stack(predict_logits_list, dim=1)

        torch.testing.assert_close(predict_logits, forward_logits, atol=1e-2, rtol=5e-2)

    @pytest.mark.parametrize("splits", [
        ([3, 4],),
        ([4, 3],),
        ([10, 14],),
        ([14, 10],),
        ([8, 8, 8],),
        ([5, 11, 8],),
        ([11, 5, 8],),
        ([1, 1, 1, 21],),
    ])
    def test_step_split_consistency(self, config, splits):
        """異なる分割でstepを呼んだ結果がforwardと一致する"""
        splits = splits[0]
        total_len = sum(splits)
        model = RecursiveCompressorLM(config)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (1, total_len))

        with torch.no_grad():
            forward_logits = model(input_ids).logits

            hidden = None
            step_logits_list = []
            pos = 0
            for length in splits:
                chunk = input_ids[:, pos:pos + length]
                logits, hidden = model.step(chunk, hidden)
                step_logits_list.append(logits)
                pos += length
            step_logits = torch.cat(step_logits_list, dim=1)

        torch.testing.assert_close(step_logits, forward_logits, atol=1e-2, rtol=5e-2)

    def test_step_matches_predict_token_by_token(self, config):
        """stepを1トークンずつ呼んだ結果がpredictと一致する"""
        seq_len = 24
        model = RecursiveCompressorLM(config)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (1, seq_len))

        with torch.no_grad():
            hidden_p = None
            predict_logits_list = []
            for t in range(seq_len):
                token = input_ids[:, t]
                logits, hidden_p = model.predict(token, hidden_p)
                predict_logits_list.append(logits)
            predict_logits = torch.stack(predict_logits_list, dim=1)

            hidden_s = None
            step_logits_list = []
            for t in range(seq_len):
                chunk = input_ids[:, t:t + 1]
                logits, hidden_s = model.step(chunk, hidden_s)
                step_logits_list.append(logits)
            step_logits = torch.cat(step_logits_list, dim=1)

        torch.testing.assert_close(step_logits, predict_logits, atol=5e-3, rtol=5e-2)


class TestRecursiveCompressorLMPipeline:
    """Pipeline-stage wrapper must stay numerically equivalent to the monolithic
    RecursiveCompressorLM. These tests guard against the two halves drifting apart
    when the architecture changes (e.g. the per-chunk query refactor)."""

    @pytest.fixture
    def config(self):
        # chunk_size=4 so seq_len=64 recurses a few levels (64->16->4->1->0),
        # exercising both populated query slots and trailing None slots.
        return RecursiveCompressorConfig(
            vocab_size=100,
            d_model=64,
            num_heads=4,
            d_ff=128,
            chunk_size=4,
            compress_size=2,
            num_layers=6,
        )

    def _build_stages(self, config, full_model, num_stages):
        """Split a full model into pipeline stages sharing its weights."""
        full_sd = full_model.state_dict()
        splits = RecursiveCompressorLMPipelineStage.split_config(config.num_layers, num_stages)
        stages = []
        for sp in splits:
            stage = RecursiveCompressorLMPipelineStage(
                config, sp["layer_start"], sp["layer_end"], sp["is_first"], sp["is_last"],
            )
            stage.eval()
            stage.load_from_full_model(full_sd)
            stages.append(stage)
        return stages

    @pytest.mark.parametrize("num_stages", [1, 2, 3, 6])
    @pytest.mark.parametrize("seq_len", [64, 30])
    def test_pipeline_matches_full_model(self, config, num_stages, seq_len):
        """全ステージを通した logits が monolithic モデルと一致する（pack/unpack含む）"""
        torch.manual_seed(0)
        full = RecursiveCompressorLM(config)
        full.eval()

        input_ids = torch.randint(0, config.vocab_size, (2, seq_len))
        with torch.no_grad():
            logits_full = full(input_ids).logits

            stages = self._build_stages(config, full, num_stages)
            x = input_ids
            for stage in stages:
                x = stage(x)
            logits_pipe = x

        assert logits_pipe.shape == logits_full.shape
        torch.testing.assert_close(logits_pipe, logits_full, atol=1e-2, rtol=5e-2)

    def test_pack_unpack_roundtrip(self, config):
        """_pack_xs / _unpack_xs が None スロットを含めて往復で復元する"""
        stage = RecursiveCompressorLMPipelineStage(config, 0, 2, is_first=True, is_last=False)
        stage.eval()

        # Build an xs whose slot lengths match what the model actually produces:
        # slot k has S_k = seq_len // chunk_size**k (None once it hits 0).
        batch, seq_len, d_model, cs = 2, 64, config.d_model, config.compress_size
        n = stage._num_queries()
        xs = [torch.randn(batch, seq_len, d_model)]
        s = seq_len
        for _ in range(n):
            s = s // config.chunk_size
            if s > 0:
                xs.append(torch.randn(batch, s, cs, d_model))
            else:
                xs.append(None)

        packed = stage._pack_xs(xs)
        restored = stage._unpack_xs(packed)

        assert len(restored) == len(xs)
        for orig, rest in zip(xs, restored):
            if orig is None:
                assert rest is None
            else:
                torch.testing.assert_close(rest, orig)

    def test_reconstruct_full_state_dict(self, config):
        """ステージに分割→再構成した state dict が元モデルと一致する"""
        torch.manual_seed(1)
        full = RecursiveCompressorLM(config)
        full_sd = full.state_dict()

        num_stages = 3
        splits = RecursiveCompressorLMPipelineStage.split_config(config.num_layers, num_stages)
        gathered = []
        for rank, sp in enumerate(splits):
            stage = RecursiveCompressorLMPipelineStage(
                config, sp["layer_start"], sp["layer_end"], sp["is_first"], sp["is_last"],
            )
            stage.load_from_full_model(full_sd)
            gathered.append((rank, sp, stage.state_dict()))

        reconstructed = RecursiveCompressorLMPipelineStage.reconstruct_full_state_dict(gathered)

        assert set(reconstructed.keys()) == set(full_sd.keys())
        for key in full_sd:
            torch.testing.assert_close(reconstructed[key], full_sd[key])

    def test_split_config_covers_all_layers(self):
        """split_config が全レイヤを過不足なく分配する"""
        splits = RecursiveCompressorLMPipelineStage.split_config(num_layers=16, num_stages=6)
        assert splits[0]["is_first"] and not splits[0]["is_last"]
        assert splits[-1]["is_last"] and not splits[-1]["is_first"]
        assert splits[0]["layer_start"] == 0
        assert splits[-1]["layer_end"] == 16
        # contiguous, non-overlapping coverage
        for prev, nxt in zip(splits, splits[1:]):
            assert prev["layer_end"] == nxt["layer_start"]
        total = sum(s["layer_end"] - s["layer_start"] for s in splits)
        assert total == 16

    def test_split_config_custom_layer_counts(self):
        """明示的なlayer_countsによる傾斜分配"""
        counts = [1, 2, 3, 4, 5, 9]
        splits = RecursiveCompressorLMPipelineStage.split_config(
            num_layers=24, num_stages=6, layer_counts=counts)
        assert [s["layer_end"] - s["layer_start"] for s in splits] == counts
        assert splits[0]["layer_start"] == 0 and splits[-1]["layer_end"] == 24
        for prev, nxt in zip(splits, splits[1:]):
            assert prev["layer_end"] == nxt["layer_start"]
        # invalid: wrong sum
        with pytest.raises(AssertionError):
            RecursiveCompressorLMPipelineStage.split_config(
                num_layers=24, num_stages=6, layer_counts=[4, 4, 4, 4, 4, 5])
        # invalid: wrong length
        with pytest.raises(AssertionError):
            RecursiveCompressorLMPipelineStage.split_config(
                num_layers=24, num_stages=6, layer_counts=[12, 12])

    def test_pipeline_matches_full_model_uneven_split(self, config):
        """傾斜分配でも monolithic モデルと logits が一致する"""
        torch.manual_seed(2)
        full = RecursiveCompressorLM(config)
        full.eval()
        full_sd = full.state_dict()

        counts = [1, 2, 3]  # uneven split of 6 layers over 3 stages
        splits = RecursiveCompressorLMPipelineStage.split_config(
            config.num_layers, 3, layer_counts=counts)
        stages = []
        for sp in splits:
            st = RecursiveCompressorLMPipelineStage(
                config, sp["layer_start"], sp["layer_end"], sp["is_first"], sp["is_last"])
            st.eval()
            st.load_from_full_model(full_sd)
            stages.append(st)

        input_ids = torch.randint(0, config.vocab_size, (2, 64))
        with torch.no_grad():
            logits_full = full(input_ids).logits
            x = input_ids
            for st in stages:
                x = st(x)
        torch.testing.assert_close(x, logits_full, atol=1e-2, rtol=5e-2)


class TestDataFormatting:
    def test_extract_turns_sharegpt(self):
        conversations = [
            {"from": "human", "value": "Q1"},
            {"from": "gpt", "value": "A1"},
            {"from": "human", "value": "Q2"},
            {"from": "gpt", "value": "A2"},
        ]
        turns = _extract_turns_sharegpt(conversations)
        assert turns == [("Q1", "A1"), ("Q2", "A2")]

    def test_extract_turns_messages(self):
        messages = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]
        turns = _extract_turns_messages(messages)
        assert turns == [("Q1", "A1"), ("Q2", "A2")]

    def test_text_to_chunks_short(self):
        """短いテキストは1チャンクに収まる（先頭BOS、末尾EOS）"""
        from unittest.mock import MagicMock
        tokenizer = MagicMock()
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.encode.return_value = [10, 11, 12]
        chunks = _text_to_chunks(tokenizer, "x", context_length=8)
        assert chunks == [[1, 10, 11, 12, 2]]

    def test_text_to_chunks_long(self):
        """長いテキストはcontext_length単位で分割（先頭BOS、最終チャンク末尾EOS）"""
        from unittest.mock import MagicMock
        tokenizer = MagicMock()
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        # encode returns 10 tokens; with BOS+EOS = 12 tokens; context_length=4
        tokenizer.encode.return_value = list(range(10, 20))
        chunks = _text_to_chunks(tokenizer, "x", context_length=4)
        # full tokens: [1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2] (12 tokens)
        # split into 4: [1,10,11,12], [13,14,15,16], [17,18,19,2]
        assert chunks == [[1, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 2]]

    def test_pack_chunks_basic(self):
        """spec例の動作: <s>abcdefghij, <s>123, <s>あいうえお をcontext_length=8で"""
        # Tokens (1 char = 1 token), <s> = 1
        # "abcdefghij" -> [1,2,...,10]
        # "123" -> [101,102,103]
        # "あいうえお" -> [201,202,203,204,205]
        # Step 1: BOS prefix
        # "<s>abcdefghij" -> [1,2,3,4,5,6,7,8,9,10] (with BOS=1)... wait that uses 1 for both
        # Use distinct ids:
        BOS = 99
        # text1 = "abcdefghij" (10 tokens)
        # text2 = "123" (3 tokens)
        # text3 = "あいうえお" (5 tokens)
        chunks_text1 = [[BOS, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10]]  # split at 8
        chunks_text2 = [[BOS, 101, 102, 103]]
        chunks_text3 = [[BOS, 201, 202, 203, 204, 205]]
        all_chunks = chunks_text1 + chunks_text2 + chunks_text3
        packed = _pack_chunks(all_chunks, context_length=8, pad_token_id=0)
        # Expected:
        # 1. [BOS,1,2,3,4,5,6,7] (full)
        # 2. [8,9,10,BOS,101,102,103,0] (3+4=7, pad 1)
        # 3. [BOS,201,202,203,204,205,0,0] (6, pad 2)
        assert len(packed) == 3
        assert packed[0] == [BOS, 1, 2, 3, 4, 5, 6, 7]
        assert packed[1] == [8, 9, 10, BOS, 101, 102, 103, 0]
        assert packed[2] == [BOS, 201, 202, 203, 204, 205, 0, 0]

    def test_pack_chunks_no_trailing_bos(self):
        """末尾BOSが追加されないことを確認"""
        chunks = [[1, 10, 11], [1, 20]]
        packed = _pack_chunks(chunks, context_length=8, pad_token_id=0)
        # [1,10,11,1,20] + [0,0,0] = [1,10,11,1,20,0,0,0]
        assert packed == [[1, 10, 11, 1, 20, 0, 0, 0]]

    def test_pack_chunks_full_length(self):
        """ちょうどcontext_length長のチャンクはそのままemit"""
        chunks = [[1, 2, 3, 4, 5, 6, 7, 8], [1, 9]]
        packed = _pack_chunks(chunks, context_length=8, pad_token_id=0)
        assert packed[0] == [1, 2, 3, 4, 5, 6, 7, 8]
        assert packed[1] == [1, 9, 0, 0, 0, 0, 0, 0]

    def test_pack_chunks_drops_single_token_sample(self):
        """1トークンだけになるサンプルはドロップ（labels全PADでNaN防止）"""
        # context_length=4. First chunk fills exactly. Second chunk has just 1 token
        # (a continuation chunk where text length aligned poorly).
        chunks = [[1, 2, 3, 4], [5]]
        packed = _pack_chunks(chunks, context_length=4, pad_token_id=0)
        # The 1-token continuation should be dropped, not emitted as [5, 0, 0, 0]
        # which would produce all-PAD labels.
        assert len(packed) == 1
        assert packed[0] == [1, 2, 3, 4]

    def test_pack_chunks_keeps_two_token_sample(self):
        """2トークン以上は保持（labelsに有効位置が1つは残る）"""
        chunks = [[1, 2, 3, 4], [5, 6]]
        packed = _pack_chunks(chunks, context_length=4, pad_token_id=0)
        assert len(packed) == 2
        assert packed[0] == [1, 2, 3, 4]
        assert packed[1] == [5, 6, 0, 0]

    def test_turns_sharegpt_item(self):
        """ShareGPT対話が (q, a) タプルのリストになる"""
        item = {"conversations": [
            {"from": "human", "value": "Q1"},
            {"from": "gpt", "value": "A1"},
            {"from": "human", "value": "Q2"},
            {"from": "gpt", "value": "A2"},
        ]}
        turns = _turns_sharegpt_item(item)
        assert turns == [("Q1", "A1"), ("Q2", "A2")]

    def test_turns_messages_item(self):
        """messages対話が (q, a) タプルのリストになる"""
        item = {"messages": [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
        ]}
        assert _turns_messages_item(item) == [("Q1", "A1")]

    def test_turns_item_empty_returns_none(self):
        """ターンが取れない対話は None"""
        assert _turns_sharegpt_item({"conversations": []}) is None
        assert _turns_messages_item({"messages": []}) is None

    def test_conversation_to_ids_and_mask(self):
        """応答トークン(answer本文+EOS)だけ mask=1、prompt(BOS+[INST]q[/INST])は0"""
        from unittest.mock import MagicMock
        tokenizer = MagicMock()
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2

        # encode("[INST]q[/INST]") -> prompt body, encode(answer) -> answer body
        def fake_encode(text, add_special_tokens=False):
            return {"[INST]Q1[/INST]": [10, 11], "A1": [20],
                    "[INST]Q2[/INST]": [12], "A2": [21, 22]}[text]
        tokenizer.encode.side_effect = fake_encode

        ids, mask = _conversation_to_ids_and_mask(tokenizer, [("Q1", "A1"), ("Q2", "A2")])
        # turn1: <s>[10,11] (prompt) + [20] <eos> (answer)
        # turn2: <s>[12] (prompt) + [21,22] <eos> (answer)
        assert ids == [1, 10, 11, 20, 2, 1, 12, 21, 22, 2]
        assert mask == [0, 0, 0, 1, 1, 0, 0, 1, 1, 1]

    def test_conversation_to_samples_no_concat_and_pad(self):
        """1会話が context_length 長サンプル1つになり、会話をまたぐ連結はしない"""
        from unittest.mock import MagicMock
        tokenizer = MagicMock()
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 0
        tokenizer.encode.side_effect = lambda text, add_special_tokens=False: \
            {"[INST]Q[/INST]": [10, 11], "A": [20]}.get(text, [99])

        samples = _conversation_to_samples(tokenizer, [("Q", "A")], context_length=8)
        # ids: <s>10 11 20 <eos> = [1,10,11,20,2], pad to 8
        assert len(samples) == 1
        ids, mask = samples[0]
        assert ids == [1, 10, 11, 20, 2, 0, 0, 0]
        assert mask == [0, 0, 0, 1, 1, 0, 0, 0]

    def test_conversation_to_samples_splits_long(self):
        """context_lengthを超える会話は分割される（連結はしないが分割はする）"""
        from unittest.mock import MagicMock
        tokenizer = MagicMock()
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 0
        # prompt 2 tokens, answer 6 tokens -> total per turn 1+2+6+1=10 > ctx 8
        tokenizer.encode.side_effect = lambda text, add_special_tokens=False: \
            {"[INST]Q[/INST]": [10, 11], "A": [20, 21, 22, 23, 24, 25]}.get(text, [99])
        samples = _conversation_to_samples(tokenizer, [("Q", "A")], context_length=8)
        # ids total = [1,10,11,20,21,22,23,24,25,2] (10) -> chunk0 len8, chunk1 len2 (pad)
        assert len(samples) == 2
        assert samples[0][0] == [1, 10, 11, 20, 21, 22, 23, 24]
        assert samples[1][0] == [25, 2, 0, 0, 0, 0, 0, 0]

    def test_units_doc_item_no_prefix(self):
        """文書アイテムは[DOC]プリフィックスなしで生のtextが返る"""
        item = {"text": "本日は晴天なり"}
        assert _units_doc_item(item) == ["本日は晴天なり"]

    def test_memmap_packed(self, tmp_path):
        """パック付きMemmapDatasetの構築と読み出し"""
        from unittest.mock import MagicMock
        tokenizer = MagicMock()
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 0
        tokenizer.encode.return_value = [10, 11, 12]

        items = [{"text": "hello"}, {"text": "world"}]
        cache_path = str(tmp_path / "test.mmap")

        _build_memmap_packed(cache_path, items, tokenizer, context_length=16, units_fn=_units_doc_item)

        ds = MemmapDataset(cache_path, pad_token_id=0)
        # Each text -> [1,10,11,12,2] (5 tokens). Two texts: 10 tokens, fits in 16. Pads 6.
        assert len(ds) == 1

        input_ids, labels = ds[0]
        assert input_ids.shape == (15,)
        assert input_ids[0].item() == 1  # BOS

    def test_memmap_packed_parallel_matches_serial(self, tmp_path):
        """num_workers > 1 でもシリアルと同じ結果になる"""
        # Use the real tokenizer because workers re-load it
        from dataset import get_tokenizer
        tokenizer = get_tokenizer()
        items = [{"text": f"sample text number {i} " * 5} for i in range(20)]

        serial_path = str(tmp_path / "serial.mmap")
        parallel_path = str(tmp_path / "parallel.mmap")

        _build_memmap_packed(serial_path, list(items), tokenizer, context_length=64, units_fn=_units_doc_item, num_workers=1)
        _build_memmap_packed(parallel_path, list(items), tokenizer, context_length=64, units_fn=_units_doc_item, num_workers=2)

        serial_ds = MemmapDataset(serial_path, pad_token_id=tokenizer.pad_token_id)
        parallel_ds = MemmapDataset(parallel_path, pad_token_id=tokenizer.pad_token_id)
        assert len(serial_ds) == len(parallel_ds)
        for i in range(len(serial_ds)):
            assert torch.equal(serial_ds[i][0], parallel_ds[i][0])
            assert torch.equal(serial_ds[i][1], parallel_ds[i][1])

    def test_memmap_conversations_response_only_labels(self, tmp_path):
        """対話キャッシュ: 応答トークンのみ labels に残り、prompt/PADは -100"""
        from unittest.mock import MagicMock
        tokenizer = MagicMock()
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 0
        tokenizer.encode.side_effect = lambda text, add_special_tokens=False: \
            {"[INST]Q[/INST]": [10, 11], "A": [20, 21]}.get(text, [99])

        items = [{"turns": [("Q", "A")]}]
        cache_path = str(tmp_path / "conv.mmap")
        _build_memmap_conversations(
            cache_path, items, tokenizer, context_length=8,
            turns_fn=lambda it: it["turns"],
        )

        ds = MemmapDataset(cache_path, pad_token_id=0)
        assert len(ds) == 1
        input_ids, labels = ds[0]
        # seq = [1,10,11,20,21,2,0,0]; mask = [0,0,0,1,1,1,0,0]
        # input_ids = seq[:-1] = [1,10,11,20,21,2,0]
        # labels = seq[1:] with mask[1:]==0 -> -100
        #   seq[1:]  = [10,11,20,21, 2, 0, 0]
        #   mask[1:] = [ 0, 0, 1, 1, 1, 0, 0]
        #   labels   = [-100,-100,20,21,2,-100,-100]
        assert input_ids.tolist() == [1, 10, 11, 20, 21, 2, 0]
        assert labels.tolist() == [-100, -100, 20, 21, 2, -100, -100]

    def test_memmap_conversations_no_cross_conversation_packing(self, tmp_path):
        """対話キャッシュ: 短い会話同士を連結せず、各会話が独立サンプルになる"""
        from unittest.mock import MagicMock
        tokenizer = MagicMock()
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 0
        tokenizer.encode.side_effect = lambda text, add_special_tokens=False: \
            {"[INST]Q[/INST]": [10], "A": [20]}.get(text, [99])

        items = [{"turns": [("Q", "A")]}, {"turns": [("Q", "A")]}]
        cache_path = str(tmp_path / "conv2.mmap")
        _build_memmap_conversations(
            cache_path, items, tokenizer, context_length=8,
            turns_fn=lambda it: it["turns"],
        )
        ds = MemmapDataset(cache_path, pad_token_id=0)
        # Two short conversations -> two separate samples (NOT packed into one).
        assert len(ds) == 2
        for i in range(2):
            input_ids, _ = ds[i]
            # each is <s>10 20 <eos> padded = [1,10,20,2,0,0,0] (len ctx-1=7)
            assert input_ids.tolist() == [1, 10, 20, 2, 0, 0, 0]

    def test_memmap_conversations_parallel_matches_serial(self, tmp_path):
        """対話キャッシュ: num_workers>1 でもシリアルと一致（ids/maskとも）"""
        from dataset import get_tokenizer
        tokenizer = get_tokenizer()
        items = [{"turns": [(f"質問{i}", f"回答{i}番目のテキスト")]} for i in range(20)]

        serial = str(tmp_path / "cs.mmap")
        parallel = str(tmp_path / "cp.mmap")
        _build_memmap_conversations(serial, list(items), tokenizer, 64, lambda it: it["turns"], num_workers=1)
        _build_memmap_conversations(parallel, list(items), tokenizer, 64, lambda it: it["turns"], num_workers=2)

        sd = MemmapDataset(serial, pad_token_id=tokenizer.pad_token_id)
        pd = MemmapDataset(parallel, pad_token_id=tokenizer.pad_token_id)
        assert len(sd) == len(pd) > 0
        for i in range(len(sd)):
            assert torch.equal(sd[i][0], pd[i][0])
            assert torch.equal(sd[i][1], pd[i][1])  # labels (mask-derived) match too

    def test_memmap_cache_reuse(self, tmp_path):
        """キャッシュが存在する場合は再構築しない"""
        cache_path = str(tmp_path / "test.mmap")
        context_length = 8
        # Create a dummy cache
        data = np.zeros((3, context_length), dtype=np.uint16)
        mmap = np.memmap(cache_path, dtype=np.uint16, mode="w+", shape=(3, context_length))
        mmap[:] = data
        mmap.flush()
        with open(cache_path + ".meta.json", "w") as f:
            json.dump({"num_samples": 3, "context_length": context_length}, f)

        ds = MemmapDataset(cache_path, pad_token_id=0)
        assert len(ds) == 3

