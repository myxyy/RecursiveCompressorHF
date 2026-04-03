import pytest
import torch
from unittest.mock import patch, MagicMock
from recursive_compressor_lm import RecursiveCompressorLM
from dataset import TextDataset


class TestRecursiveCompressorLM:
    @pytest.fixture
    def model_params(self):
        return dict(
            vocab_size=100,
            d_model=64,
            num_heads=4,
            d_ff=128,
            chunk_size=8,
            compress_size=4,
            num_layers=2,
        )

    @pytest.fixture
    def model(self, model_params):
        return RecursiveCompressorLM(**model_params)

    def test_output_shape(self, model, model_params):
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, model_params["vocab_size"], (batch_size, seq_len))
        logits = model(input_ids)
        assert logits.shape == (batch_size, seq_len, model_params["vocab_size"])

    def test_output_shape_non_divisible_seq_len(self, model, model_params):
        """chunk_sizeで割り切れないシーケンス長でも正しく動作する"""
        batch_size, seq_len = 2, 30  # 30 is not divisible by chunk_size=8
        input_ids = torch.randint(0, model_params["vocab_size"], (batch_size, seq_len))
        logits = model(input_ids)
        assert logits.shape == (batch_size, seq_len, model_params["vocab_size"])

    def test_single_chunk(self, model, model_params):
        """チャンクが1つだけの場合（再帰なし）"""
        batch_size = 2
        seq_len = model_params["chunk_size"]
        input_ids = torch.randint(0, model_params["vocab_size"], (batch_size, seq_len))
        logits = model(input_ids)
        assert logits.shape == (batch_size, seq_len, model_params["vocab_size"])

    def test_loss_computation(self, model, model_params):
        """CrossEntropyLossが正常に計算できる"""
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, model_params["vocab_size"], (batch_size, seq_len))
        target = torch.randint(0, model_params["vocab_size"], (batch_size, seq_len))
        logits = model(input_ids)
        loss = torch.nn.CrossEntropyLoss()(logits.view(-1, model_params["vocab_size"]), target.view(-1))
        assert loss.item() > 0
        loss.backward()

    def test_gradient_flow(self, model, model_params):
        """全パラメータに勾配が流れる"""
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, model_params["vocab_size"], (batch_size, seq_len))
        target = torch.randint(0, model_params["vocab_size"], (batch_size, seq_len))
        logits = model(input_ids)
        loss = torch.nn.CrossEntropyLoss()(logits.view(-1, model_params["vocab_size"]), target.view(-1))
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_num_layers(self, model_params):
        """レイヤー数が正しい"""
        model = RecursiveCompressorLM(**model_params)
        assert len(model.layers) == model_params["num_layers"]

    @pytest.mark.parametrize("seq_len", [1, 7, 8, 16, 24, 32])
    def test_predict_matches_forward(self, model_params, seq_len):
        """predictを1トークンずつ呼んだ結果がforwardと一致する"""
        model = RecursiveCompressorLM(**model_params)
        model.eval()

        input_ids = torch.randint(0, model_params["vocab_size"], (1, seq_len))

        with torch.no_grad():
            forward_logits = model(input_ids)

            hidden = None
            predict_logits_list = []
            for t in range(seq_len):
                token = input_ids[:, t]
                logits, hidden = model.predict(token, hidden)
                predict_logits_list.append(logits)
            predict_logits = torch.stack(predict_logits_list, dim=1)

        torch.testing.assert_close(predict_logits, forward_logits, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("splits", [
        ([3, 4],),          # chunk_size未満の分割
        ([4, 3],),
        ([10, 14],),        # chunk_sizeをまたぐ分割
        ([14, 10],),
        ([8, 8, 8],),       # chunk_size境界ぴったり
        ([5, 11, 8],),      # 不均等な3分割
        ([11, 5, 8],),
        ([1, 1, 1, 21],),   # 1トークンずつ + まとめて
    ])
    def test_step_split_consistency(self, model_params, splits):
        """異なる分割でstepを呼んだ結果がforwardと一致する"""
        splits = splits[0]
        total_len = sum(splits)
        model = RecursiveCompressorLM(**model_params)
        model.eval()

        input_ids = torch.randint(0, model_params["vocab_size"], (1, total_len))

        with torch.no_grad():
            forward_logits = model(input_ids)

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

    def test_step_matches_predict_token_by_token(self, model_params):
        """stepを1トークンずつ呼んだ結果がpredictと一致する"""
        seq_len = 24
        model = RecursiveCompressorLM(**model_params)
        model.eval()

        input_ids = torch.randint(0, model_params["vocab_size"], (1, seq_len))

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


class TestTextDataset:
    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.vocab_size = 100
        # 200 tokens -> (200-1)//32 = 6 samples for context_length=32
        tokenizer.encode.return_value = list(range(200))
        return tokenizer

    def test_dataset_length(self, mock_tokenizer, tmp_path):
        text_file = tmp_path / "test.txt"
        text_file.write_text("テスト文章です。" * 100)
        context_length = 32

        with patch("dataset.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
            dataset = TextDataset(str(tmp_path), "dummy", context_length)

        expected_samples = (200 - 1) // context_length
        assert len(dataset) == expected_samples

    def test_dataset_item_shape(self, mock_tokenizer, tmp_path):
        text_file = tmp_path / "test.txt"
        text_file.write_text("テスト文章です。" * 100)
        context_length = 32

        with patch("dataset.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
            dataset = TextDataset(str(tmp_path), "dummy", context_length)

        x, y = dataset[0]
        assert x.shape == (context_length,)
        assert y.shape == (context_length,)

    def test_dataset_target_shift(self, mock_tokenizer, tmp_path):
        """ターゲットが入力を1トークンずらしたものになっている"""
        text_file = tmp_path / "test.txt"
        text_file.write_text("テスト")
        context_length = 32

        with patch("dataset.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
            dataset = TextDataset(str(tmp_path), "dummy", context_length)

        x, y = dataset[0]
        # x = [0, 1, ..., 31], y = [1, 2, ..., 32]
        assert x[0].item() == 0
        assert y[0].item() == 1
        assert x[-1].item() == context_length - 1
        assert y[-1].item() == context_length

    def test_vocab_size(self, mock_tokenizer, tmp_path):
        text_file = tmp_path / "test.txt"
        text_file.write_text("テスト")
        context_length = 32

        with patch("dataset.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
            dataset = TextDataset(str(tmp_path), "dummy", context_length)

        assert dataset.vocab_size == 100
