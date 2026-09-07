"""Run from the repository root with: OMP_NUM_THREADS=1 uv run python <this file>."""
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path.cwd()))
from logkv import LogKV

rows = []
x = torch.randn(1, 1, 32, dtype=torch.float64,
                generator=torch.Generator().manual_seed(1)).expand(1, 64, 32)
for phase, relative in [(False, False), (False, True), (True, False)]:
    torch.manual_seed(0)
    model = LogKV(dim=32, chunk_size=4, num_heads=4, phase_emb=phase,
                  phase_levels=2, self_slot=True, relative_position_bias=relative).double().eval()
    with torch.no_grad():
        y = model(x)
    rows.append(dict(phase_emb=phase, relative_position_bias=relative,
                     max_abs_difference_from_position_0=(y-y[:, :1]).abs().max().item()))
output = json.dumps(rows, indent=2) + '\n'
Path(__file__).with_suffix('.json').write_text(output)
print(output)
