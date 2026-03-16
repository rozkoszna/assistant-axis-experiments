import torch
import torch.nn.functional as F
from pathlib import Path

in_dir = Path("outputs/gemma2b_trait_vectors")
out_file = in_dir / "assistiveness_axis.pt"

supportive = torch.load(in_dir / "supportive.pt")
hostile = torch.load(in_dir / "hostile.pt")

axis = supportive - hostile
axis = F.normalize(axis, dim=0)

torch.save(
    {
        "supportive": supportive,
        "hostile": hostile,
        "assistiveness_axis": axis,
    },
    out_file,
)

print(f"Saved {out_file}")
