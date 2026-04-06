import json
import numpy as np
import pytest
import torch
from configuration_recursive_compressor import RecursiveCompressorConfig
from recursive_compressor_lm import RecursiveCompressorLM
from dataset import (
    format_document, format_conversation, tokenize_with_bos,
    _extract_turns_sharegpt, _extract_turns_messages,
    _build_memmap, _build_memmap_packed, _format_doc_item,
    _pack_sequences, MemmapDataset,
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

        torch.testing.assert_close(predict_logits, forward_logits, atol=1e-4, rtol=1e-4)

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

        torch.testing.assert_close(step_logits, forward_logits, atol=1e-4, rtol=1e-4)

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

        torch.testing.assert_close(step_logits, predict_logits, atol=1e-4, rtol=1e-4)


class TestDataFormatting:
    def test_format_document(self):
        assert format_document("本日は晴天なり") == "[DOC]本日は晴天なり"

    def test_format_conversation(self):
        turns = [("質問1", "回答1"), ("質問2", "回答2")]
        result = format_conversation(turns)
        assert result == "[QUERY]質問1[ANSWER]回答1[QUERY]質問2[ANSWER]回答2"

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

    def test_tokenize_with_bos_short(self):
        """短いテキストはBOS+text+BOS, PADで埋める"""
        from unittest.mock import MagicMock
        tokenizer = MagicMock()
        tokenizer.bos_token_id = 1
        tokenizer.pad_token_id = 0
        tokenizer.encode.return_value = [10, 11, 12]  # 3 tokens

        input_ids, labels = tokenize_with_bos(tokenizer, "test", context_length=10)
        # seq = [1, 10, 11, 12, 1] (len=5)
        # input = [1, 10, 11, 12], labels = [10, 11, 12, 1]
        # pad to 9: input = [1,10,11,12,0,0,0,0,0], labels = [10,11,12,1,-100,-100,-100,-100,-100]
        assert input_ids[:4] == [1, 10, 11, 12]
        assert labels[:4] == [10, 11, 12, 1]
        assert all(x == 0 for x in input_ids[4:])
        assert all(x == -100 for x in labels[4:])
        assert len(input_ids) == 9
        assert len(labels) == 9

    def test_pack_sequences_short_docs(self):
        """短い文書が結合される"""
        seqs = [[1, 10, 11, 1], [1, 20, 21, 1], [1, 30, 1]]
        # total: 4+4+3=11, context_length=10
        # first pack: [1,10,11,1,1,20,21,1,0,0] (8 tokens + 2 pad)
        # second pack: [1,30,1,0,0,0,0,0,0,0]
        packed = _pack_sequences(seqs, context_length=10, pad_token_id=0)
        assert len(packed) == 2
        assert len(packed[0]) == 10
        assert len(packed[1]) == 10
        assert packed[0][:8] == [1, 10, 11, 1, 1, 20, 21, 1]
        assert packed[1][:3] == [1, 30, 1]

    def test_pack_sequences_long_doc(self):
        """長い文書はtruncateされる"""
        seqs = [list(range(20))]
        packed = _pack_sequences(seqs, context_length=10, pad_token_id=0)
        assert len(packed) == 1
        assert packed[0] == list(range(10))

    def test_memmap_packed(self, tmp_path):
        """パック付きMemmapDatasetの構築と読み出し"""
        from unittest.mock import MagicMock
        tokenizer = MagicMock()
        tokenizer.bos_token_id = 1
        tokenizer.pad_token_id = 0
        tokenizer.encode.return_value = [10, 11, 12]

        items = [{"text": "hello"}, {"text": "world"}]
        cache_path = str(tmp_path / "test.mmap")

        _build_memmap_packed(cache_path, items, tokenizer, context_length=16, format_fn=_format_doc_item)

        ds = MemmapDataset(cache_path, pad_token_id=0)
        # Two short docs (5 tokens each with BOS) should pack into 1 sample
        assert len(ds) == 1

        input_ids, labels = ds[0]
        assert input_ids.shape == (15,)
        assert input_ids[0].item() == 1  # BOS

    def test_memmap_dataset(self, tmp_path):
        """MemmapDatasetの構築と読み出し（パック有り）"""
        from unittest.mock import MagicMock
        tokenizer = MagicMock()
        tokenizer.bos_token_id = 1
        tokenizer.pad_token_id = 0
        tokenizer.encode.return_value = [10, 11, 12]

        items = [{"text": "hello"}, {"text": "world"}]
        cache_path = str(tmp_path / "test.mmap")

        _build_memmap(cache_path, items, tokenizer, context_length=8, format_fn=_format_doc_item)

        ds = MemmapDataset(cache_path, pad_token_id=0)

        input_ids, labels = ds[0]
        assert input_ids.shape == (7,)
        assert labels.shape == (7,)
        assert input_ids[0].item() == 1  # BOS
        # PAD positions in labels should be -100
        assert labels[-1].item() == -100

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

    def test_tokenize_with_bos_long(self):
        """長いテキストは末尾BOSなし、truncateされる"""
        from unittest.mock import MagicMock
        tokenizer = MagicMock()
        tokenizer.bos_token_id = 1
        tokenizer.pad_token_id = 0
        tokenizer.encode.return_value = list(range(10, 25))  # 15 tokens

        input_ids, labels = tokenize_with_bos(tokenizer, "test", context_length=10)
        # seq = [1, 10, 11, ..., 18] truncated to 10
        # input = [1,10,...,17] (9), labels = [10,11,...,18] (9)
        assert len(input_ids) == 9
        assert len(labels) == 9
        assert input_ids[0] == 1
        assert labels[0] == 10
