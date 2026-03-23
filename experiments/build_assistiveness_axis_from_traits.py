import torch
import torch.nn.functional as F
from pathlib import Path

in_dir = Path("assistant_axis_outputs/llama-3.1-8b/vectors_q50")
out_file = in_dir / "assistiveness_axis.pt"

supportive = torch.load(in_dir / "supportive.pt")
hostile = torch.load(in_dir / "hostile.pt")

axis = supportive - hostile
axis = F.normalize(axis, dim=-1, eps=1e-8)

torch.save(
    {
        "supportive": supportive,
        "hostile": hostile,
        "assistiveness_axis": axis,
    },
    out_file,
)

print(f"Saved {out_file}")
