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
