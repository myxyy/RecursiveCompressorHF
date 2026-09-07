"""Content-neutral pooling diagnostic; not a test of trained models."""
import json
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path.cwd()))
from logkv import Compressor

H, C = 8, 4
v = torch.eye(C*C, dtype=torch.float64).expand(H, C*C, C*C)
q = torch.zeros_like(v)
rows = []
for mode in ['none', 'base', 'learned']:
    comp = Compressor(H, 0.0 if mode == 'none' else math.log(C)/(C-1), mode == 'learned').double()

    def pool(values, levels):
        nq, nk, nv = q[:, :values.size(1)], q[:, :values.size(1)], values
        for _ in range(levels):
            n = nv.size(1)//C
            nq, nk, nv = comp(nq.reshape(H*n,C,C*C), nk.reshape(H*n,C,C*C),
                              nv.reshape(H*n,C,C*C), num_chunks=n)
            nq, nk, nv = (a.reshape(H,n,C*C) for a in [nq,nk,nv])
        return nv

    with torch.no_grad():
        local = v[:, :C]
        local_diff = (pool(local,1)-pool(local[:, [0,2,1,3]],1)).abs().max().item()
        permutation = list(range(C*C)); permutation[1],permutation[C]=permutation[C],permutation[1]
        tree_diff = (pool(v,2)-pool(v[:,permutation],2)).abs().max().item()
    rows.append(dict(mode=mode, local_swap_1_2_max_difference=local_diff,
                     two_level_swap_1_4_max_difference=tree_diff))
Path(__file__).with_suffix('.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
