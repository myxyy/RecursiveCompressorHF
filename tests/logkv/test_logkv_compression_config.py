"""Compression options survive trainer resolution, transfer and HF export."""
import json
from types import SimpleNamespace

import pytest
import torch

from models.logkv.configuration import LogKVConfig
from models.logkv.modeling import LogKVLM
from models.logkv.pipeline import LogKVLMPipelineStage
from training.train_logkv import build_parser, load_checkpoint_config
from training.train_logkv_pipeline import model_config, parse_args


TOKENIZER = SimpleNamespace(vocab_size=32, pad_token_id=0, bos_token_id=1, eos_token_id=2)


def tiny_config(**kwargs):
    return LogKVConfig(vocab_size=32, d_model=16, d_ff=24, num_heads=2,
                      num_layers=2, chunk_size=4, self_slot=True,
                      pad_token_id=0, bos_token_id=1, eos_token_id=2, **kwargs)


def test_legacy_config_and_unspecified_cli_keep_original_defaults(tmp_path):
    data = tiny_config().to_dict()
    data.pop('compression_query')
    data.pop('compression_key')
    (tmp_path / 'config.json').write_text(json.dumps(data))
    loaded = load_checkpoint_config(tmp_path)
    assert (loaded.compression_query, loaded.compression_key) == ('last', 'attention')
    args = parse_args([])
    assert args.compression_query is args.compression_key is None
    fresh = model_config(args, TOKENIZER)
    assert (fresh.compression_query, fresh.compression_key) == ('last', 'attention')


@pytest.mark.parametrize('query', ['last', 'next'])
@pytest.mark.parametrize('key', ['attention', 'mean'])
def test_independent_cli_options_and_config_roundtrip(tmp_path, query, key):
    flags = ['--compression-query', query, '--compression-key', key, '--level-amplify']
    ordinary = build_parser().parse_args(flags)
    pipeline = parse_args(flags)
    assert (ordinary.compression_query, ordinary.compression_key) == (query, key)
    config = model_config(pipeline, TOKENIZER)
    assert (config.compression_query, config.compression_key) == (query, key)
    assert config.level_amplify
    config.save_pretrained(tmp_path)
    loaded = LogKVConfig.from_pretrained(tmp_path, local_files_only=True)
    assert (loaded.compression_query, loaded.compression_key) == (query, key)
    assert loaded.level_amplify


@pytest.mark.parametrize('option,value', [('compression_query', 'future'),
                                         ('compression_key', 'max')])
def test_invalid_compression_settings_rejected(option, value):
    with pytest.raises(ValueError, match=option):
        tiny_config(**{option: value})
    with pytest.raises(SystemExit):
        build_parser().parse_args(['--' + option.replace('_', '-'), value])


@pytest.mark.parametrize('other', ['--no-level-bias', '--learnable-decay'])
def test_amplification_cli_exclusion(other):
    with pytest.raises(SystemExit):
        parse_args(['--level-amplify', other])


@pytest.mark.parametrize('zero_bias', [False, True])
def test_weights_only_explicit_overrides_and_reverse_compression(tmp_path, zero_bias):
    tiny_config(no_level_bias=zero_bias).save_pretrained(tmp_path)
    original = (tmp_path / 'config.json').read_bytes()
    args = parse_args(['--start-checkpoint', str(tmp_path), '--compression-query', 'next',
                       '--compression-key', 'mean', '--level-amplify'])
    changed = model_config(args, TOKENIZER, tmp_path)
    assert (changed.compression_query, changed.compression_key) == ('next', 'mean')
    assert changed.level_amplify and not changed.no_level_bias
    assert (tmp_path / 'config.json').read_bytes() == original
    changed.save_pretrained(tmp_path / 'new')
    unchanged = model_config(parse_args([]), TOKENIZER, tmp_path / 'new')
    assert (unchanged.compression_query, unchanged.compression_key) == ('next', 'mean')
    reverse = model_config(parse_args(['--compression-query', 'last',
                                      '--compression-key', 'attention']),
                           TOKENIZER, tmp_path / 'new')
    assert (reverse.compression_query, reverse.compression_key) == ('last', 'attention')
    assert reverse.level_amplify


@pytest.mark.parametrize('override', [dict(compression_query='next'),
                                     dict(compression_key='mean'),
                                     dict(level_amplify=True),
                                     dict(no_level_bias=True),
                                     dict(learnable_decay=True)])
def test_resume_rejects_changed_explicit_settings(tmp_path, override):
    tiny_config().save_pretrained(tmp_path)
    with pytest.raises(ValueError, match='conflicts'):
        load_checkpoint_config(tmp_path, resume=True, **override)


def test_pipeline_resume_preserves_saved_modes_and_accepts_matching_flags(tmp_path):
    tiny_config(compression_query='next', compression_key='mean',
                level_amplify=True).save_pretrained(tmp_path)
    for flags in [[], ['--compression-query', 'next', '--compression-key', 'mean',
                       '--level-amplify']]:
        args = parse_args(['--resume', str(tmp_path), *flags])
        config = model_config(args, TOKENIZER, tmp_path)
        assert (config.compression_query, config.compression_key) == ('next', 'mean')
        assert config.level_amplify
    args = parse_args(['--resume', str(tmp_path), '--compression-query', 'last'])
    with pytest.raises(ValueError, match='--compression-query conflicts'):
        model_config(args, TOKENIZER, tmp_path)


def test_learned_bias_parameters_are_not_silently_dropped(tmp_path):
    tiny_config(learnable_decay=True).save_pretrained(tmp_path)
    for overrides in [dict(level_amplify=True), dict(no_level_bias=True)]:
        with pytest.raises(ValueError, match='learned decay'):
            load_checkpoint_config(tmp_path, **overrides)
    changed = load_checkpoint_config(tmp_path, compression_query='next', compression_key='mean')
    assert changed.learnable_decay


def test_transferred_weights_export_and_pipeline_use_compression_settings(tmp_path):
    torch.manual_seed(61)
    source = LogKVLM(tiny_config())
    source.save_pretrained(tmp_path / 'original')
    cfg = load_checkpoint_config(tmp_path / 'original', compression_query='next',
                                 compression_key='mean', level_amplify=True)
    model, loading = LogKVLM.from_pretrained(tmp_path / 'original', config=cfg,
                                             output_loading_info=True)
    assert not loading['missing_keys'] and not loading['unexpected_keys']
    for name, value in source.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    for layer in model.layers:
        assert layer.attention.compression_query == 'next'
        assert layer.attention.compression_key == 'mean'
        assert layer.attention.level_sign == 1
    stages = [LogKVLMPipelineStage(cfg, **info) for info in
              LogKVLMPipelineStage.split_config(cfg.num_layers, 2)]
    for stage in stages:
        stage.load_model_directory(tmp_path / 'original')
        for layer in stage.layers.values():
            assert layer.attention.compression_query == 'next'
            assert layer.attention.compression_key == 'mean'
    ids = torch.randint(3, 32, (2, 18))
    with torch.no_grad():
        expected = model(ids).logits
        actual = ids
        for stage in stages:
            actual = stage(actual)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    model.save_pretrained(tmp_path / 'converted')
    restored = LogKVLM.from_pretrained(tmp_path / 'converted')
    assert restored.config.compression_query == 'next'
    assert restored.config.compression_key == 'mean'
    with torch.no_grad():
        torch.testing.assert_close(restored(ids).logits, expected, atol=0, rtol=0)
