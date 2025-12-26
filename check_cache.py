import os
from pathlib import Path

def check_model_cache(model_name="meta-llama/Llama-2-7b-hf"):
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_dir / f"models--{model_name.replace('/', '--')}"
    
    print(f"Model: {model_name}")
    print(f"Cache directory: {cache_dir}")
    print(f"Model cache path: {model_dir}")
    print()
    
    if not cache_dir.exists():
        print("❌ Hugging Face cache directory does not exist")
        return False
    
    if not model_dir.exists():
        print("❌ Model is not cached")
        return False
    
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists():
        print("❌ Model snapshots directory not found")
        return False
    
    snapshots = list(snapshots_dir.iterdir())
    if not snapshots:
        print("❌ No snapshots found")
        return False
    
    latest_snapshot = max(snapshots, key=os.path.getmtime)
    print(f"✅ Model is cached")
    print(f"   Latest snapshot: {latest_snapshot.name}")
    
    essential_files = ["config.json", "tokenizer_config.json"]
    for file in essential_files:
        file_path = latest_snapshot / file
        if file_path.exists():
            size = file_path.stat().st_size / (1024 * 1024)
            print(f"   ✓ {file} ({size:.2f} MB)")
        else:
            print(f"   ✗ {file} (missing)")
    
    model_files = list(latest_snapshot.glob("*.safetensors")) + list(latest_snapshot.glob("*.bin"))
    if model_files:
        total_size = sum(f.stat().st_size for f in model_files) / (1024**3)
        print(f"   Model files: {len(model_files)} files ({total_size:.2f} GB)")
    
    return True

if __name__ == "__main__":
    check_model_cache()
