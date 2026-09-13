"""Run the unchanged task protocol with deterministic kernels and init audit."""
import hashlib
import json
import os
import runpy
import sys
from common import ROOT,SOURCE,save
assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
import torch
from safetensors.torch import load_file
sys.path.insert(0,str(SOURCE));sys.path.insert(0,str(SOURCE/'exp/copying'))
import logkv_lm

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False
original_init=logkv_lm.LogKVLM.__init__
def checked_init(self,config):
    original_init(self,config)
    initial=load_file(ROOT/'initial_model/model.safetensors')
    state=self.state_dict()
    assert state.keys()==initial.keys()
    assert all(torch.equal(v.cpu(),initial[k]) for k,v in state.items())
    run_name=sys.argv[sys.argv.index('--run-name')+1]
    out=os.environ['DATA_DIR']+'/exp/copying/'+run_name
    save(out+'/initialization_audit.json',dict(passed=True,initial_weights_identical=True,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
        visible_gpu=os.environ['CUDA_VISIBLE_DEVICES'],torch_version=torch.__version__,
        cuda_version=torch.version.cuda,device_name=torch.cuda.get_device_name(0),
        rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()))
logkv_lm.LogKVLM.__init__=checked_init
runpy.run_path(str(SOURCE/'exp/copying/train.py'),run_name='__main__')
