from pathlib import Path
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from assistant_axis import load_axis

# 1) load HF axis
axis_path = hf_hub_download(
    repo_id="lu-christina/assistant-axis-vectors",
    filename="gemma-2-27b/assistant_axis.pt",   # change to your matching model
    repo_type="dataset",
)
hf_axis = load_axis(axis_path)
hf_axis = hf_axis.cpu()

# 2) load local vectors
vec_dir = Path("outputs/gemma2b_trait_vectors")  # change if needed

results = []

for p in sorted(vec_dir.glob("*.pt")):
    obj = torch.load(p, map_location="cpu")

    if isinstance(obj, dict):
        # adjust if your files store the tensor under a key
        tensor_keys = [k for k, v in obj.items() if torch.is_tensor(v)]
        if len(tensor_keys) != 1:
            print(f"Skipping {p.name}: ambiguous dict keys {tensor_keys}")
            continue
        trait_vec = obj[tensor_keys[0]]
    elif torch.is_tensor(obj):
        trait_vec = obj
    else:
        print(f"Skipping {p.name}: unsupported type {type(obj)}")
        continue

    if trait_vec.shape != hf_axis.shape:
        print(f"Skipping {p.name}: shape mismatch {tuple(trait_vec.shape)} vs {tuple(hf_axis.shape)}")
        continue

    if trait_vec.ndim == 1:
        cos = F.cosine_similarity(trait_vec, hf_axis, dim=0).item()
        score = cos
        layer_mean = None
    elif trait_vec.ndim == 2:
        layer_scores = F.cosine_similarity(trait_vec, hf_axis, dim=1)
        score = layer_scores.mean().item()
        layer_mean = layer_scores
    else:
        print(f"Skipping {p.name}: unexpected ndim {trait_vec.ndim}")
        continue

    results.append((p.stem, score))

results.sort(key=lambda x: x[1], reverse=True)

print("\n=== traits ranked by alignment with HF assistant axis ===")
for name, score in results:
    print(f"{name:30s} {score:+.4f}")