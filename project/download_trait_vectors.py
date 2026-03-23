from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="lu-christina/assistant-axis-vectors",
    repo_type="dataset",
    allow_patterns=["llama-3.3-70b/trait_vectors/*"],
)

print("Downloaded to:", local_dir)