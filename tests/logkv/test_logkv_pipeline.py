"""CPU regression tests and an explicit two-GPU torchrun integration smoke.

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
    --module tests.logkv.test_logkv_pipeline --distributed-smoke /path/to/empty/smoke-dir
"""
from datetime import timedelta
import json
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.utils.data import TensorDataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from models.logkv.configuration import LogKVConfig
from models.logkv.modeling import LogKVLM
from models.logkv.pipeline import LogKVLMPipelineStage, export_model
from training.train_logkv import build_parser
from training.train_logkv_pipeline import (EpochCursorSampler, PipelineEngine, checkpoints,
    clip_global_grad_norm, exclusive_run, parse_args, train)


def tokenizer():
    vocab = {'[UNK]': 0, '[PAD]': 1, '[BOS]': 2, '[EOS]': 3}
    vocab.update({f't{i}': i for i in range(4, 32)})
    backend = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]',
                                  pad_token='[PAD]', bos_token='[BOS]', eos_token='[EOS]')


def config():
    return LogKVConfig(vocab_size=32, d_model=16, d_ff=32, num_heads=2,
        num_layers=3, chunk_size=3, phase_emb=True, phase_levels=2,
        learnable_decay=True, self_slot=True, gated_attention=True, conv_kernel_size=4,
        pad_token_id=1, bos_token_id=2, eos_token_id=3)


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('counts', [[3], [1, 2], [2, 1], [1, 1, 1]])
def test_stage_forward_backward_cache_and_global_weight_names(counts):
    torch.manual_seed(12)
    cfg = config()
    model = LogKVLM(cfg).double()
    infos = LogKVLMPipelineStage.split_config(3, len(counts), counts)
    stages = [LogKVLMPipelineStage(cfg, **info).double() for info in infos]
    for stage in stages:
        stage.load_from_full_model(model.state_dict())
    ids = torch.randint(4, 32, (2, 19))
    ref = model(ids).logits
    out = ids
    for stage in stages:
        out = stage(out)
    torch.testing.assert_close(out, ref, atol=1e-12, rtol=1e-12)
    ref.square().sum().backward()
    out.square().sum().backward()
    actual = {name: p for stage in stages for name, p in stage.named_parameters()}
    assert set(actual) == set(dict(model.named_parameters()))
    for name, p in model.named_parameters():
        torch.testing.assert_close(actual[name].grad, p.grad, atol=1e-12, rtol=1e-12)
    with torch.no_grad():
        hidden = [None] * len(stages)
        pieces = []
        for start, end in [(0, 7), (7, 8), (8, 19)]:
            single = end - start == 1
            x = ids[:, start] if single else ids[:, start:end]
            for i, stage in enumerate(stages):
                x, hidden[i] = stage.step(x, hidden[i], single_token=single)
            pieces.append(x[:, None] if single else x)
        torch.testing.assert_close(torch.cat(pieces, 1), ref, atol=1e-12, rtol=1e-12)


def test_export_load_streaming_and_sharded_warm_start(tmp_path):
    torch.manual_seed(13)
    cfg, tk = config(), tokenizer()
    infos = LogKVLMPipelineStage.split_config(cfg.num_layers, 2)
    stages = [LogKVLMPipelineStage(cfg, **info) for info in infos]
    for rank, stage in enumerate(stages):
        torch.save({'stage_info': stage.stage_info, 'model': stage.state_dict()}, tmp_path / f'stage_{rank}.pt')
    path = export_model(tmp_path, cfg, tk, infos)
    # These are the exact functions imported by inference/predict_stream.py.
    from inference.predict_stream import _load_model, _load_tokenizer, stream_generate
    model = _load_model(str(path), torch.device('cpu'), torch.float32).eval()
    loaded_tk = _load_tokenizer(str(path))
    ids = torch.tensor([[4, 5, 6, 7]])
    with torch.no_grad():
        x = ids
        for stage in stages:
            x = stage(x)
        torch.testing.assert_close(model(ids).logits, x, atol=0, rtol=0)
    count, _, interrupted = stream_generate(model, loaded_tk, 't4 t5', 5, 1., .95,
                                           torch.device('cpu'), True, False)
    assert count == 3 and not interrupted
    model.save_pretrained(tmp_path / 'sharded', max_shard_size='10KB')
    assert (tmp_path / 'sharded/model.safetensors.index.json').exists()
    for info in LogKVLMPipelineStage.split_config(3, 2, [1, 2]):
        stage = LogKVLMPipelineStage(cfg, **info)
        stage.load_model_directory(tmp_path / 'sharded')
        for name, value in stage.state_dict().items():
            assert torch.equal(value, model.state_dict()[name])


def test_sampler_resume_across_epochs():
    dataset = range(24)
    sampler = EpochCursorSampler(dataset, seed=8)
    for epoch in [0, 1, 3]:
        sampler.set_cursor(epoch, 0)
        full = list(sampler)
        sampler.set_cursor(epoch, 16)
        assert list(sampler) == full[16:]
        assert len(sampler) == 8


def test_run_lock_released_after_error(tmp_path):
    with pytest.raises(ValueError):
        with exclusive_run(tmp_path):
            with pytest.raises(RuntimeError, match='Another trainer'):
                with exclusive_run(tmp_path):
                    pass
            raise ValueError('simulated failure')
    with exclusive_run(tmp_path):
        pass


@pytest.mark.parametrize('counts', [[0, 3], [2], [2, 2], [-1, 4]])
def test_invalid_splits(counts):
    with pytest.raises(ValueError):
        LogKVLMPipelineStage.split_config(3, 2, counts)


@pytest.mark.parametrize('argv', [
    ['--batch-size', '5', '--n-microbatches', '2'], ['--context-length', '1'],
    ['--grad-accum', '0'], ['--max-checkpoints', '0'], ['--run-name', '../oops'],
    ['--resume', 'latest', '--start-checkpoint', 'model'],
])
def test_invalid_cli(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_cli_shared_architecture_and_checkpoints(tmp_path):
    old = build_parser().parse_args([])
    new = parse_args([])
    for key, value in vars(old).items():
        if key != 'batch_size':
            assert getattr(new, key) == value
    assert new.batch_size == new.n_microbatches == 12
    for name in ['checkpoint-9', 'checkpoint-11', 'checkpoint-99.tmp', 'checkpoint-broken']:
        path = tmp_path / name
        (path / 'model').mkdir(parents=True)
        (path / 'model/config.json').write_text('{}')
        (path / 'trainer_state.json').write_text('{}')
    (tmp_path / 'checkpoint-100').mkdir()
    assert [p.name for p in checkpoints(tmp_path)] == ['checkpoint-9', 'checkpoint-11']


def distributed_smoke(root):
    """Tiny offline GPU training: gradients, bf16, resume, export, controls."""
    from inference.predict_stream import _load_model, _load_tokenizer, stream_generate
    root = Path(root).resolve()
    os.environ['DATA_DIR'] = str(root)
    dist.init_process_group('nccl', timeout=timedelta(minutes=3))
    rank = dist.get_rank()
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    try:
        # Full-model reference exists only for this tiny numerical test.
        torch.manual_seed(51)
        full = LogKVLM(config()).to(device)
        stage = LogKVLMPipelineStage(config(), **LogKVLMPipelineStage.split_config(3, 2)[rank]).to(device)
        stage.load_from_full_model(full.state_dict())
        generator = torch.Generator().manual_seed(52)
        ids = torch.randint(4, 32, (8, 8), generator=generator)
        labels = torch.randint(4, 32, (8, 8), generator=generator)
        labels[:2] = -100
        labels[4:, :6] = -100  # Deliberately unequal valid-token counts.
        batches = [(ids[:4], labels[:4]), (ids[4:], labels[4:])]
        engine = PipelineEngine(stage, device, 2, 'fp32')
        actual_loss = engine.backward(batches)
        ref_loss = full(ids.to(device), labels=labels.to(device)).loss
        ref_loss.backward()
        assert abs(actual_loss - ref_loss.item()) < 1e-6
        for name, p in stage.named_parameters():
            torch.testing.assert_close(p.grad, dict(full.named_parameters())[name].grad, atol=2e-6, rtol=2e-5)
        actual_norm = clip_global_grad_norm(stage, .05, device)
        ref_norm = torch.nn.utils.clip_grad_norm_(full.parameters(), .05)
        assert abs(actual_norm - ref_norm.item()) < 1e-5
        for name, p in stage.named_parameters():
            torch.testing.assert_close(p.grad, dict(full.named_parameters())[name].grad, atol=2e-7, rtol=2e-5)
        stage.zero_grad()
        assert engine.backward([(ids[:4], torch.full_like(labels[:4], -100))]) == 0.
        assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in stage.parameters())
        del engine, stage, full
        if rank == 0:
            print('PASS: full-model loss/gradients, global clipping, all-ignored labels', flush=True)

        tk = tokenizer()
        ids = torch.randint(4, 32, (24, 8), generator=generator)
        labels = torch.randint(4, 32, (24, 8), generator=generator)
        labels[::3, :5] = -100
        dataset = TensorDataset(ids, labels)
        common = ['--context-length', '9', '--d-model', '16', '--d-ff', '32',
            '--num-heads', '2', '--num-layers', '3', '--batch-size', '4',
            '--n-microbatches', '2', '--grad-accum', '2', '--num-epochs', '2',
            '--num-workers', '0', '--checkpoint-interval', '1', '--max-checkpoints', '2',
            '--warmup', '2', '--log-interval', '1', '--sample-interval', '2',
            '--sample-max-new-tokens', '2', '--control-file', str(root / 'control.cmd')]
        final = train(parse_args(common + ['--run-name', 'full', '--max-steps', '4']), dataset, tk)
        train(parse_args(common + ['--run-name', 'resumed', '--max-steps', '2']), dataset, tk)
        resumed = train(parse_args(common + ['--run-name', 'resumed', '--max-steps', '4', '--resume', 'latest']), dataset, tk)
        a = torch.load(final / f'stage_{rank}.pt', weights_only=True, map_location='cpu')['model']
        b = torch.load(resumed / f'stage_{rank}.pt', weights_only=True, map_location='cpu')['model']
        for name in a:
            torch.testing.assert_close(a[name], b[name], atol=0, rtol=0)
        a = torch.load(final / f'optimizer_{rank}.pt', weights_only=False, map_location='cpu')
        b = torch.load(resumed / f'optimizer_{rank}.pt', weights_only=False, map_location='cpu')
        assert_tree_equal(a, b)
        assert [p.name for p in checkpoints(final.parent)] == ['checkpoint-3', 'checkpoint-4']
        state = json.loads((resumed / 'trainer_state.json').read_text())
        assert (state['step'], state['epoch'], state['batch_in_epoch']) == (4, 1, 2)
        # Absolute max-steps must not perform one additional update on resume.
        again = train(parse_args(common + ['--run-name', 'resumed', '--max-steps', '4', '--resume', 'latest']), dataset, tk)
        assert again == resumed
        if rank == 0:
            print('PASS: bf16 training, distributed samples, bit-exact mid-epoch resume across epoch boundary', flush=True)
        # Layout changes require weights-only restart, never silent optimizer loss.
        with pytest.raises(ValueError, match='Resume mismatch'):
            train(parse_args(common + ['--run-name', 'resumed', '--resume', 'latest',
                '--stage-layer-split', '1,2']), dataset, tk)
        with pytest.raises(ValueError, match='other/later checkpoints'):
            train(parse_args(common + ['--run-name', 'resumed', '--resume', 'checkpoint-1']), dataset, tk)
        train(parse_args(common + ['--run-name', 'warm', '--start-checkpoint', str(final),
            '--stage-layer-split', '1,2', '--max-steps', '1']), dataset, tk)
        if rank == 0:
            (root / 'control.cmd').write_text('save_and_exit')
        dist.barrier()
        stopped = train(parse_args(common + ['--run-name', 'controlled', '--max-steps', '1']), dataset, tk)
        assert json.loads((stopped / 'trainer_state.json').read_text())['step'] == 0
        if rank == 0:
            model_dir = str(resumed / 'model')
            loaded = _load_model(model_dir, torch.device('cpu'), torch.float32).eval()
            loaded_tk = _load_tokenizer(model_dir)
            count, _, interrupted = stream_generate(loaded, loaded_tk, 't4 t5', 5, 1., .95,
                                                    torch.device('cpu'), True, False)
            assert count == 3 and not interrupted
            (root / 'review.json').write_text(json.dumps(dict(
                passed=True, world_size=dist.get_world_size(), final_checkpoint=str(resumed),
                loss_and_gradients_match=True, global_clip_matches=True,
                all_ignored_loss_zero=True, resume_weights_bit_exact=True,
                resume_optimizer_rng_bit_exact=True, checkpoint_rotation=True,
                resume_step=4, resume_epoch=1, resume_batch_in_epoch=2,
                distributed_samples=True, changed_split_warm_start=True,
                control_save_and_exit=True, predict_stream_load_and_generate=True), indent=2) + '\n')
            print('PASS: warm start with new split, synchronized save/exit, predict_stream load+generate', flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--distributed-smoke', required=True)
    distributed_smoke(parser.parse_args().distributed_smoke)
