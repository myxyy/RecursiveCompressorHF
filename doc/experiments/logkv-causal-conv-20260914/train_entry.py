"""Unmodified main trainer with explicit task, deterministic kernels and init check."""
import hashlib
import os
import runpy
import sys
from pathlib import Path
from common import INITIAL, BASE_INITIAL, SOURCE, MODES, NUM_LAYERS, NUM_PARAMS, bind_task, task_identity, sha, save

assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
mode = os.environ['LOGKV_TASK']; assert mode in MODES
import torch
from safetensors.torch import load_file
sys.path.insert(0, str(SOURCE))
task = bind_task(mode)
import logkv_lm
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
original_init = logkv_lm.LogKVLM.__init__

def checked_init(self, config):
    original_init(self, config)
    assert not config.phase_emb and config.self_slot and config.gated_attention
    assert config.num_layers == NUM_LAYERS and config.conv_kernel_size == 4
    initial = load_file(INITIAL)
    # Match all shared initial parameters to the completed two-layer control;
    # only the convolution block comes from fresh convolution-enabled initialization.
    self.load_state_dict(initial, strict=True)
    state = self.state_dict()
    base = load_file(BASE_INITIAL)
    assert all(torch.equal(state[k].cpu(), value) for k, value in base.items())
    assert sum(p.numel() for p in self.parameters()) == NUM_PARAMS
    assert state.keys() == initial.keys()
    assert all(torch.equal(value.cpu(), initial[key]) for key, value in state.items())
    run_name = sys.argv[sys.argv.index('--run-name') + 1]
    out = Path(os.environ['DATA_DIR']) / 'exp' / mode / run_name
    save(out / 'initialization_audit.json', dict(passed=True, **task_identity(),
        initial_weights_identical=True, initial_weights_sha256=sha(INITIAL),
        shared_two_layer_initial_weights_identical=True, base_initial_sha256=sha(BASE_INITIAL),
        num_layers=NUM_LAYERS, conv_kernel_size=4, num_params=NUM_PARAMS,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
        visible_gpu=os.environ['CUDA_VISIBLE_DEVICES'], torch_version=torch.__version__,
        cuda_version=torch.version.cuda, device_name=torch.cuda.get_device_name(0),
        rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()))

logkv_lm.LogKVLM.__init__ = checked_init
runpy.run_path(str(SOURCE / 'exp/copying/train.py'), run_name='__main__')
