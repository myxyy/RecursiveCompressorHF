import importlib.util
from pathlib import Path

import pytest
import torch

from configuration_logkv import LogKVConfig
from logkv_lm import LogKVLM
from exp.variable_memory.task import make_batch, score
from exp.variable_memory.common import evaluate_cell, cell_seed
from exp.variable_memory.evaluate import grid


@pytest.mark.parametrize("task", ["copying", "selective-copying"])
@pytest.mark.parametrize("memory,horizon,prefix", [(10,1,0),(16,4,7),(32,64,15),(64,256,63),(128,1,0)])
def test_task_order_and_boundaries(task,memory,horizon,prefix):
    x,y=make_batch(task,memory,horizon,prefix,3,torch.Generator().manual_seed(8))
    assert x.shape == y.shape == (3,prefix+horizon+2*memory)
    assert (x[:,:prefix]==0).all()
    assert (x[:,-memory-1:]==9).all()
    assert (y[:,:-memory]==0).all()
    for row,target in zip(x,y):
        assert torch.equal(row[(row>=1)&(row<=8)],target[-memory:])
    assert score(torch.nn.functional.one_hot(y,10).float(),y,memory)==(3*memory,3,3*memory,3)


@pytest.mark.parametrize("task", ["copying","selective-copying"])
def test_original_task_exactly_reproduced(task):
    path=Path(__file__).resolve().parents[1]/task/"task.py"
    spec=importlib.util.spec_from_file_location("original_task",path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    for t in [1,64]:
        expected=module.make_batch(t,4,generator=torch.Generator().manual_seed(5))
        actual=make_batch(task,10,t,0,4,torch.Generator().manual_seed(5))
        assert all(torch.equal(a,b) for a,b in zip(actual,expected))


def test_evaluation_handles_answer_across_chunks():
    torch.manual_seed(0)
    model=LogKVLM(LogKVConfig(vocab_size=10,d_model=16,d_ff=32,num_layers=1,num_heads=4,
                             conv_kernel_size=4,gated_attention=True,self_slot=True)).double().eval()
    args=(model,"copying",16,11,3,7,123,torch.device("cpu"))
    whole=evaluate_cell(*args,chunk_size=4096,precision="fp32")
    split=evaluate_cell(*args,chunk_size=7,precision="fp32")
    assert whole==split


def test_evaluation_grid_has_disjoint_seeds_and_expected_cells():
    cells=grid()
    assert len(cells)==220 and len(set(cells))==220
    seeds={cell_seed("copying",m,t,p,12345) for m,t,p,_ in cells}
    assert len(seeds)==220
    assert not seeds & {cell_seed("copying",m,t,p,54321) for m,t,p,_ in cells}
