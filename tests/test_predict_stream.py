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
