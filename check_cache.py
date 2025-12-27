import os
from pathlib import Path


def check_model_cache(model_name="meta-llama/Llama-2-7b-hf"):
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_dir / f"models--{model_name.replace('/', '--')}"

    if not model_dir.exists():
        return False

    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists():
        return False

    snapshots = list(snapshots_dir.iterdir())
    if not snapshots:
        return False

    latest_snapshot = max(snapshots, key=os.path.getmtime)

    required = ["config.json", "tokenizer_config.json"]
    if not all((latest_snapshot / f).exists() for f in required):
        return False

    model_files = list(latest_snapshot.glob("*.safetensors")) + list(latest_snapshot.glob("*.bin"))
    return len(model_files) > 0


if __name__ == "__main__":
    print("Cached" if check_model_cache() else "Not cached")
