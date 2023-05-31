import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.parser import parse_args_to_dict
from utils.misc import resolve
from proxies.models import make_model

if __name__ == "__main__":
    args = parse_args_to_dict()
    assert "ckpt" in args, "Please specify a checkpoint path with --ckpt=PATH"

    torch.set_grad_enabled(False)
    ckpt_path = resolve(args["ckpt"])
    trace_path = ckpt_path.parent / f"traced-{str(ckpt_path.stem)}.pth"
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    config = ckpt["hyper_parameters"]
    model = make_model(config)
    batch_size = 32
    n_elements = 89
    n_lattice_parameters = 6
    n_sg = 1
    dummy_x = (
        torch.randint(0, 4, (batch_size, n_elements)),
        torch.randint(0, 230, (batch_size, n_sg)),
        torch.rand(batch_size, n_lattice_parameters),
    )
    traced_cell = torch.jit.trace(model, (dummy_x,))
    torch.jit.save(traced_cell, trace_path)
    loaded = torch.jit.load(str(trace_path))
