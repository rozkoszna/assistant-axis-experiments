from pathlib import Path
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def load_tensor(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, dict):
        if "assistiveness_axis" in obj and torch.is_tensor(obj["assistiveness_axis"]):
            return obj["assistiveness_axis"]
        tensor_items = {k: v for k, v in obj.items() if torch.is_tensor(v)}
        if len(tensor_items) == 1:
            return next(iter(tensor_items.values()))
        raise ValueError(f"Ambiguous tensor dict in {path}: {list(tensor_items.keys())}")
    raise TypeError(f"Unsupported object in {path}: {type(obj)}")


def cosine_score(vec: torch.Tensor, axis: torch.Tensor):
    if vec.shape != axis.shape:
        return None

    if vec.ndim == 1:
        return F.cosine_similarity(vec, axis, dim=0).item(), None

    if vec.ndim == 2:
        per_layer = F.cosine_similarity(vec, axis, dim=1)
        return per_layer.mean().item(), per_layer

    return None


def main():
    repo_root = Path(__file__).resolve().parent.parent
    vec_dir = repo_root / "outputs" / "gemma2b_trait_vectors"
    axis_path = vec_dir / "assistiveness_axis.pt"
    out_dir = repo_root / "outputs" / "trait_visuals"
    out_dir.mkdir(parents=True, exist_ok=True)

    axis = load_tensor(axis_path)

    results = []
    layer_matrix = []
    layer_names = []

    for p in sorted(vec_dir.glob("*.pt")):
        if p.name == "assistiveness_axis.pt":
            continue

        try:
            vec = load_tensor(p)
        except Exception as e:
            print(f"Skipping {p.name}: {e}")
            continue

        out = cosine_score(vec, axis)
        if out is None:
            print(f"Skipping {p.name}: shape mismatch {tuple(vec.shape)} vs {tuple(axis.shape)}")
            continue

        score, per_layer = out
        results.append((p.stem, score))

        if per_layer is not None:
            layer_matrix.append(per_layer)
            layer_names.append(p.stem)

    results.sort(key=lambda x: x[1], reverse=True)

    if not results:
        raise ValueError("No compatible trait vectors found.")

    # --- Bar chart ---
    names = [x[0] for x in results]
    scores = [x[1] for x in results]

    plt.figure(figsize=(12, max(6, len(names) * 0.35)))
    plt.barh(names, scores)
    plt.axvline(0.0, linewidth=1)
    plt.gca().invert_yaxis()
    plt.xlabel("Cosine similarity to assistiveness axis")
    plt.ylabel("Trait")
    plt.title("Trait alignment with assistiveness axis")
    plt.tight_layout()
    bar_path = out_dir / "trait_alignment_bar.png"
    plt.savefig(bar_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved {bar_path}")

    # --- Heatmap ---
    if layer_matrix:
        mat = torch.stack(layer_matrix).numpy()  # [traits, layers]

        plt.figure(figsize=(14, max(6, len(layer_names) * 0.35)))
        plt.imshow(mat, aspect="auto", interpolation="nearest")
        plt.colorbar(label="Cosine similarity")
        plt.yticks(range(len(layer_names)), layer_names)
        plt.xticks(range(mat.shape[1]), range(mat.shape[1]))
        plt.xlabel("Layer")
        plt.ylabel("Trait")
        plt.title("Per-layer trait alignment with assistiveness axis")
        plt.tight_layout()
        heatmap_path = out_dir / "trait_alignment_heatmap.png"
        plt.savefig(heatmap_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"Saved {heatmap_path}")

    # --- Print ranking ---
    print("\n=== trait ranking ===")
    for name, score in results:
        print(f"{name:30s} {score:+.4f}")


if __name__ == "__main__":
    main()