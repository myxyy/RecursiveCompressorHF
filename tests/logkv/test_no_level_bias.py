"""Zero fixed level bias: independent attention oracle and checkpoint semantics."""
import pytest
import torch
from models.logkv.attention import LogKV
from models.logkv.configuration import LogKVConfig
from models.logkv.modeling import LogKVLM
from models.logkv.pipeline import LogKVLMPipelineStage
from tests.logkv.test_logkv import reference_forward


def test_zero_bias_reference_split_predict_and_gradients():
    torch.manual_seed(41)
    model = LogKV(16, 4, num_heads=4, gated_attention=True, self_slot=True,
                  no_level_bias=True).double()
    assert model.level_decay is None and model.level_sign == 0
    x = torch.randn(1, 65, 16, dtype=torch.float64, requires_grad=True)
    actual = model(x)
    expected = reference_forward(model, x)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    ga = torch.autograd.grad(actual.square().sum(), (x, *model.parameters()), retain_graph=True)
    ge = torch.autograd.grad(expected.square().sum(), (x, *model.parameters()))
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, atol=1e-11, rtol=1e-11)
    with torch.no_grad():
        a, hidden = model.step(x[:, :31])
        b, hidden = model.step(x[:, 31:64], hidden)
        c, hidden = model.predict(x[:, 64], hidden)
        torch.testing.assert_close(torch.cat((a, b, c[:, None]), 1), expected, atol=1e-12, rtol=1e-12)


def test_save_load_and_pipeline(tmp_path):
    cfg = LogKVConfig(vocab_size=10, d_model=16, num_heads=4, d_ff=32, num_layers=2,
                      conv_kernel_size=4, gated_attention=True, self_slot=True, no_level_bias=True)
    model = LogKVLM(cfg).double().eval()
    assert all(layer.attention.level_decay is None for layer in model.layers)
    model.save_pretrained(tmp_path)
    loaded = LogKVLM.from_pretrained(tmp_path, dtype=torch.float64).eval()
    assert loaded.config.no_level_bias
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 1]])
    with torch.no_grad():
        expected = model(ids).logits
        torch.testing.assert_close(loaded(ids).logits, expected, atol=1e-12, rtol=1e-12)
        x = ids
        for info in LogKVLMPipelineStage.split_config(2, 2):
            stage = LogKVLMPipelineStage(cfg, **info).double().eval()
            stage.load_from_full_model(model.state_dict())
            x = stage(x)
        torch.testing.assert_close(x, expected, atol=1e-12, rtol=1e-12)
    assert not LogKVConfig().no_level_bias


@pytest.mark.parametrize('flag', ['learnable_decay', 'level_amplify'])
def test_conflicting_bias_options_rejected(flag):
    with pytest.raises(ValueError):
        LogKVConfig(no_level_bias=True, **{flag: True})
    with pytest.raises(ValueError):
        LogKV(16, 4, no_level_bias=True, **{flag: True})


def test_training_cli_bias_modes(monkeypatch):
    from training.train_logkv import build_parser
    from training.train_logkv_pipeline import parse_args, model_config
    from tests.logkv.test_logkv_pipeline import tokenizer
    monkeypatch.delenv('WORLD_SIZE', raising=False)
    for parse in (build_parser().parse_args, parse_args):
        assert not parse([]).no_level_bias
        assert parse(['--no-level-bias']).no_level_bias
        for conflict in ['--learnable-decay', '--level-amplify']:
            with pytest.raises(SystemExit):
                parse(['--no-level-bias', conflict])
    args = parse_args(['--no-level-bias', '--d-model', '16', '--d-ff', '32',
                       '--num-heads', '2', '--num-layers', '2'])
    cfg = model_config(args, tokenizer())
    stage = LogKVLMPipelineStage(cfg, **LogKVLMPipelineStage.split_config(2, 2)[0])
    assert all(layer.attention.level_sign == 0 and layer.attention.level_decay is None
               for layer in stage.layers.values())


def test_legacy_checkpoint_and_zero_bias_warm_start(tmp_path, monkeypatch):
    import json
    from inference.predict_stream import _load_model
    from training.train_logkv import load_checkpoint_config
    from training.train_logkv_pipeline import parse_args, model_config
    from tests.logkv.test_logkv_pipeline import tokenizer
    from models.logkv.pipeline import export_model
    monkeypatch.delenv('WORLD_SIZE', raising=False)
    tk = tokenizer()
    old = LogKVLM(LogKVConfig(vocab_size=32, d_model=16, d_ff=32, num_heads=2,
        num_layers=2, conv_kernel_size=4, self_slot=True, gated_attention=True,
        pad_token_id=1, bos_token_id=2, eos_token_id=3)).eval()
    legacy = tmp_path / 'legacy'
    old.save_pretrained(legacy)
    # Simulate an actual pre-option checkpoint, not merely a new config with False.
    config_path = legacy / 'config.json'
    data = json.loads(config_path.read_text()); data.pop('no_level_bias')
    config_path.write_text(json.dumps(data))
    original_bytes = config_path.read_bytes()
    args = parse_args(['--resume', 'latest'])
    saved = model_config(args, tk, legacy)
    assert not saved.no_level_bias
    loaded = _load_model(str(legacy), torch.device('cpu'), torch.float32).eval()
    assert all(layer.attention.level_sign == -1 for layer in loaded.layers)
    ids = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 4, 5]])
    with torch.no_grad():
        torch.testing.assert_close(loaded(ids).logits, old(ids).logits, atol=1e-7, rtol=1e-6)
    with pytest.raises(ValueError, match='resumed checkpoint'):
        load_checkpoint_config(legacy, no_level_bias=True, resume=True)
    args = parse_args(['--start-checkpoint', str(legacy), '--no-level-bias'])
    zero = model_config(args, tk, legacy)
    assert zero.no_level_bias
    assert config_path.read_bytes() == original_bytes  # Never edit the source checkpoint.
    infos = LogKVLMPipelineStage.split_config(2, 2)
    export_dir = tmp_path / 'export'; export_dir.mkdir()
    stages = []
    for rank, info in enumerate(infos):
        stage = LogKVLMPipelineStage(zero, **info).eval()
        stage.load_model_directory(legacy)
        for k, v in stage.state_dict().items():
            assert torch.equal(v, old.state_dict()[k])
        stages.append(stage)
        torch.save({'stage_info': info, 'model': stage.state_dict()}, export_dir/f'stage_{rank}.pt')
    model_dir = export_model(export_dir, zero, tk, infos)
    resumed = model_config(parse_args(['--resume', 'latest']), tk, model_dir)
    assert resumed.no_level_bias
    assert load_checkpoint_config(model_dir, no_level_bias=True, resume=True).no_level_bias
    actual = _load_model(str(model_dir), torch.device('cpu'), torch.float32).eval()
    assert actual.config.no_level_bias
    assert actual.state_dict().keys() == old.state_dict().keys()
    with torch.no_grad():
        x = ids
        for stage in stages:
            x = stage(x)
        torch.testing.assert_close(actual(ids).logits, x, atol=1e-7, rtol=1e-6)
    # Loaded learned/amplified models cannot silently become zero-bias models.
    for flag in ['learnable_decay', 'level_amplify']:
        invalid = tmp_path / flag
        LogKVConfig(**{flag: True}).save_pretrained(invalid)
        with pytest.raises(ValueError, match='no_level_bias'):
            load_checkpoint_config(invalid, no_level_bias=True)
