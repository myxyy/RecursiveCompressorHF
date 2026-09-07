"""Short-horizon precision diagnostic; does not overwrite the main evaluation."""
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd()/'exp'/'copying'))
from evaluate import build_t_grid, eval_horizon
from logkv_lm import LogKVLM

ROOT = Path(__file__).resolve().parent
run = Path('/mnt/raid0/RecursiveCompressor/exp/copying/logkv-d512-logu-comp-learned-gated-self-20260907')
rows = []
for checkpoint, folder in [('best', 'model_best'), ('final', 'model')]:
    for precision, dtype, autocast in [('bf16', torch.bfloat16, True),
                                       ('fp32-autocast-bf16', torch.float32, True),
                                       ('fp32', torch.float32, False)]:
        torch.set_float32_matmul_precision('highest' if precision == 'fp32' else 'high')
        model = LogKVLM.from_pretrained(run/folder).to(device='cuda', dtype=dtype).eval()
        generator = torch.Generator().manual_seed(12345)
        scores = {}
        for t in build_t_grid(11):
            tok, st = eval_horizon(model, t, 256, generator, torch.device('cuda'), autocast)
            scores[t] = dict(token_acc=tok, string_acc=st, n=256)
        rows.append(dict(checkpoint=checkpoint, precision=precision,
                         matmul_precision=torch.get_float32_matmul_precision(), results=scores))
        print(checkpoint, precision, {t:scores[t] for t in [1,64,1024,2048]}, flush=True)
        del model
        torch.cuda.empty_cache()
(ROOT/'precision_probe.json').write_text(json.dumps(rows, indent=2)+'\n')
