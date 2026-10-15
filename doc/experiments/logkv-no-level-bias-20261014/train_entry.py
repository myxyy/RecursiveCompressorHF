"""Current trainer with explicit task, deterministic kernels and identical initialization."""
import hashlib
import os
import sys
from pathlib import Path
from common import INITIAL, BASE_INITIAL, SOURCE, MODES, NUM_PARAMS, bind_task, task_identity, sha, save

assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
mode = os.environ['LOGKV_TASK']; assert mode in MODES
import torch
from safetensors.torch import load_file
sys.path.insert(0, str(SOURCE))
from models.logkv.modeling import LogKVLM
from exp.copying.train import main

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
original_init = LogKVLM.__init__


def checked_init(self, config):
    original_init(self, config)
    assert config.no_level_bias and not config.learnable_decay and not config.level_amplify
    assert not config.phase_emb and config.self_slot and config.gated_attention
    assert config.num_layers == 2 and config.conv_kernel_size == 4
    initial = load_file(INITIAL)
    self.load_state_dict(initial, strict=True)
    base = load_file(BASE_INITIAL)
    assert self.state_dict().keys() == base.keys()
    assert all(torch.equal(value, base[key]) for key, value in self.state_dict().items())
    assert sum(p.numel() for p in self.parameters()) == NUM_PARAMS
    assert all(layer.attention.level_sign == 0 and layer.attention.level_decay is None for layer in self.layers)
    run_name = sys.argv[sys.argv.index('--run-name') + 1]
    out = Path(os.environ['DATA_DIR']) / 'exp' / mode / run_name
    assert not (out / 'initialization_audit.json').exists()
    save(out / 'initialization_audit.json', dict(passed=True, **task_identity(mode),
        initial_weights_identical=True, initial_weights_sha256=sha(INITIAL),
        shared_two_layer_initial_weights_identical=True, base_initial_sha256=sha(BASE_INITIAL),
        num_layers=2, conv_kernel_size=4, no_level_bias=True, learnable_decay=False, num_params=NUM_PARAMS,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
        visible_gpu=os.environ['CUDA_VISIBLE_DEVICES'], torch_version=torch.__version__,
        cuda_version=torch.version.cuda, device_name=torch.cuda.get_device_name(0),
        rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()))


LogKVLM.__init__ = checked_init
main(task=bind_task(mode))
