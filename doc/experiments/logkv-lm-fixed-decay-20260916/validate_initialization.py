"""CPU exact comparison of original learned/fixed seed0 initializations."""
import json
import sys
import torch
from common import HERE, SOURCE, BASE, save, shared_hash
sys.path.insert(0,str(SOURCE))
from configuration_logkv import LogKVConfig
from logkv_lm import LogKVLM

torch.set_num_threads(1)
config=LogKVConfig(**json.loads((BASE/'results/config-5000.json').read_text()))
assert config.learnable_decay
torch.manual_seed(0);learned=LogKVLM(config)
config_fixed=LogKVConfig(**dict(config.to_dict(),learnable_decay=False))
torch.manual_seed(0);fixed=LogKVLM(config_fixed)
state=learned.state_dict();reference=fixed.state_dict()
assert all(torch.equal(value,state[name]) for name,value in reference.items())
assert sorted(set(state)-set(reference))==sorted(f'layers.{i}.attention.level_decay' for i in range(16))
assert shared_hash(learned)==shared_hash(fixed)
save(HERE/'initialization_audit.json',dict(passed=True,seed=0,shared_initial_sha256=shared_hash(fixed),
    shared_tensors=len(reference),removed_parameters=128,
    fixed_num_params=sum(p.numel() for p in fixed.parameters()),
    learned_num_params=sum(p.numel() for p in learned.parameters()),
    caveat='Same shared weights exactly. Initial outputs need not match under autocast because original fixed scalar and learned tensor bias have different arithmetic precision.'))
print('Shared initial weights match exactly.')
