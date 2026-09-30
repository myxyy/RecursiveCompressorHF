"""Streaming UI checks without checkpoints, downloads or a GPU."""
import io
import os
import signal

import pytest
import torch

from inference import predict_stream as stream


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def encode(self, text, return_tensors):
        return torch.tensor([[1, 2]])

    def decode(self, ids, skip_special_tokens=False):
        words = {1: '問', 2: '\n', 3: '答', 4: 'え\n', 9: '<eos>'}
        return ''.join(words[i] for i in ids if not (skip_special_tokens and i == 9))


class Terminal(io.StringIO):
    def isatty(self):
        return True

    def fileno(self):
        return 123


def test_counter_excludes_prompt_and_includes_hidden_eos(monkeypatch):
    output = io.StringIO()
    monkeypatch.setattr(stream.sys, 'stdout', output)
    status = stream._GenerationStatus()
    streamer = stream._StatusTextStreamer(Tokenizer(), status, skip_special_tokens=True)
    streamer.put(torch.tensor([[1, 2]]))
    assert status.tokens == 0
    streamer.put(torch.tensor([3]))
    assert status.tokens == 1
    streamer.put(torch.tensor([4, 9]))
    streamer.end()
    assert status.tokens == 3
    assert output.getvalue() == '問\n答え\n\n'
    assert '\033' not in output.getvalue()


@pytest.mark.parametrize('outcome', ['normal', 'interrupt', 'error'])
def test_generation_cleans_up_terminal_and_signal(monkeypatch, outcome):
    output = Terminal()
    monkeypatch.setattr(stream.sys, 'stdout', output)
    monkeypatch.setenv('TERM', 'xterm-256color')
    monkeypatch.setattr(stream.os, 'get_terminal_size', lambda fd: os.terminal_size((80, 24)))
    old_handler = signal.getsignal(signal.SIGINT)

    class Model:
        def generate(self, ids, streamer, stopping_criteria, **kwargs):
            streamer.put(ids)
            streamer.put(torch.tensor([3]))
            if outcome == 'error':
                raise RuntimeError('generation failed')
            if outcome == 'interrupt':
                signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
                assert stopping_criteria[0](ids, None)
            else:
                streamer.put(torch.tensor([9]))
            streamer.end()
            return torch.tensor([[1, 2, 3]] if outcome == 'interrupt' else [[1, 2, 3, 9]])

    args = (Model(), Tokenizer(), 'prompt', 10, 1., 1., 'cpu', True, True)
    if outcome == 'error':
        with pytest.raises(RuntimeError, match='generation failed'):
            stream.stream_generate(*args)
    else:
        count, elapsed, interrupted = stream.stream_generate(*args)
        assert count == (1 if outcome == 'interrupt' else 2)
        assert elapsed > 0
        assert interrupted == (outcome == 'interrupt')
        assert f'{count} tokens |' in output.getvalue()
    assert signal.getsignal(signal.SIGINT) == old_handler
    assert '\033[1;23r' in output.getvalue()
    assert output.getvalue().endswith('\0337\033[r\033[24;1H\033[2K\0338')


def test_status_rate_throttle_resize_and_disable(monkeypatch):
    output = Terminal()
    monkeypatch.setattr(stream.sys, 'stdout', output)
    monkeypatch.setenv('TERM', 'xterm')
    now = [10.]
    size = [os.terminal_size((80, 24))]
    monkeypatch.setattr(stream.time, 'perf_counter', lambda: now[0])
    monkeypatch.setattr(stream.os, 'get_terminal_size', lambda fd: size[0])
    status = stream._GenerationStatus()
    status.refresh(force=True)
    status.tokens = 12
    now[0] = 12.
    status.refresh()
    assert '12 tokens | 6.00 tok/s | 2.0s' in output.getvalue()
    previous = output.getvalue()
    now[0] += .01
    status.refresh()
    assert output.getvalue() == previous
    size[0] = os.terminal_size((40, 12))
    status.write('continued')
    status.refresh(force=True)
    assert '\033[1;11r\033[11;1Hcontinued' in output.getvalue()
    status.close()
    output.seek(0)
    output.truncate()
    disabled = stream._GenerationStatus(enabled=False)
    disabled.refresh(force=True)
    disabled.write('plain')
    disabled.close()
    assert output.getvalue() == 'plain'


def test_ignore_eos_in_real_generate_and_restore(capsys):
    from models.logkv.configuration import LogKVConfig
    from models.logkv.modeling import LogKVLM

    model = LogKVLM(LogKVConfig(vocab_size=10, d_model=16, num_heads=4,
                               d_ff=32, num_layers=1, eos_token_id=[8, 9],
                               bos_token_id=None, pad_token_id=0)).eval()
    model.generation_config.suppress_tokens = [4]
    model.generation_config.forced_eos_token_id = 9
    original_config = model.generation_config.to_dict()

    def prefer_eos(module, args, result):
        result.logits.fill_(-1000.)
        # EOS wins unless excluded. Then existing suppression must still exclude 4.
        for token, score in [(9, 1000.), (8, 900.), (4, 800.), (3, 0.)]:
            result.logits[..., token] = score

    hook = model.register_forward_hook(prefer_eos)
    try:
        for ignore, stop, expected_count, expected_text in [
            (True, True, 6, '問\n答答答答答答\n'),
            (True, False, 6, '問\n答答答答答答\n'),
            (False, True, 1, '問\n<eos>\n'),
            (False, False, 6, '問\n' + '<eos>' * 6 + '\n'),
        ]:
            count, _, interrupted = stream.stream_generate(
                model, Tokenizer(), 'prompt', 8, 1., 1., 'cpu', False, stop,
                status_bar=False, ignore_eos=ignore)
            assert count == expected_count and not interrupted
            assert capsys.readouterr().out == expected_text
            assert model.generation_config.to_dict() == original_config
    finally:
        hook.remove()


def test_ignore_eos_repl_toggle(monkeypatch, capsys):
    from types import SimpleNamespace

    monkeypatch.setattr(stream.sys, 'argv', ['predict_stream', '--model-dir', 'unused', '--device', 'cpu'])
    monkeypatch.setattr(stream, '_load_model', lambda *a, **kw: SimpleNamespace(
        eval=lambda: None, parameters=lambda: []))
    monkeypatch.setattr(stream, '_load_tokenizer', lambda _: Tokenizer())
    prompts = iter(['ignore-eos', 'ignore-eos on', 'first', 'ignore-eos invalid',
                    'second', 'ignore-eos off', 'third', 'exit'])
    monkeypatch.setattr(stream, '_make_prompt_session', lambda: SimpleNamespace(prompt=lambda _: next(prompts)))
    flags = []

    def generate(*args, **kwargs):
        flags.append(kwargs['ignore_eos'])
        return 1, 1., False

    monkeypatch.setattr(stream, 'stream_generate', generate)
    stream.main()
    assert flags == [True, True, False]
    output = capsys.readouterr().out
    assert 'ignore-eos = False' in output
    assert "Invalid value for ignore-eos: 'invalid'" in output
