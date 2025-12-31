"""
SFT Pipeline: Train M_0 on latest synthetic data, push as M_i
"""

import argparse
import json
import os
import re
import shutil
import tempfile

import logging
import warnings

import torch
import wandb
from datasets import Dataset, disable_progress_bars as disable_datasets_progress
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download, list_repo_files, login
from huggingface_hub.utils import disable_progress_bars as disable_hf_progress
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

# Suppress verbose logging
disable_hf_progress()
disable_datasets_progress()
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")

DATASET_REPO = "parksoojae/learn-star"
MODEL_REPO = "parksoojae/STaR"
BASE_MODEL_FOLDER = "M_0"


def get_latest_synth_file():
    """Find most recent synth_i.jsonl in dataset repo."""
    files = list_repo_files(DATASET_REPO, repo_type="dataset")
    pattern = re.compile(r"synth_(\d+)\.jsonl")
    
    synth_files = [(f, int(m.group(1))) for f in files if (m := pattern.match(f))]
    synth_files.sort(key=lambda x: x[1])
    
    return synth_files[-1]


def download_synth_dataset(synth_filename):
    """Download synth jsonl and convert to Dataset."""
    local_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=synth_filename,
        repo_type="dataset"
    )
    
    with open(local_path) as f:
        examples = [json.loads(line) for line in f if line.strip()]
    
    return Dataset.from_list(examples)


def get_next_model_iteration():
    """Determine next M_i iteration number."""
    files = list_repo_files(MODEL_REPO)
    pattern = re.compile(r"M_(\d+)")
    
    iterations = {int(m.group(1)) for f in files 
                  if f.startswith("M_") and "/" in f 
                  and (m := pattern.match(f.split("/")[0]))}
    
    return max(iterations) + 1 if iterations else 1


def load_base_model():
    """Load M_0 from repo. Requires CUDA GPU."""
    load_dotenv()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, subfolder=BASE_MODEL_FOLDER)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        subfolder=BASE_MODEL_FOLDER,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print(f"✓ Model loaded: {MODEL_REPO}/{BASE_MODEL_FOLDER}")
    return model, tokenizer


def format_example(example):
    """Format single example for training - matches inference prompt format."""
    prompt = f"""Q: {example["question"]}
Answer Choices:
{example["answer_choices"]}
A:"""
    completion = f""" {example["rationale"]}"""
    return {"prompt": prompt, "completion": completion}


def run_sft(model, tokenizer, dataset, output_dir, iteration):
    """Run SFT training. Requires CUDA GPU with bf16 support."""
    dataset = dataset.map(lambda x: format_example(x), desc=None)
    
    config = SFTConfig(
        output_dir=output_dir,
        run_name=f"star-sft-M{iteration}",
        num_train_epochs=3,
        per_device_train_batch_size=16,    
        gradient_accumulation_steps=2,      
        learning_rate=2e-5,                  
        weight_decay=0.01,
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        optim="adamw_torch_fused",
        logging_strategy="steps",
        logging_steps=10,
        log_level="error",
        save_strategy="no",
        bf16=True,
        max_length=1024,
        packing=False,                        
        gradient_checkpointing=True,
        report_to="wandb",
        completion_only_loss=True,
        disable_tqdm=False,
    )
    
    # Suppress console logging but keep wandb
    logging.getLogger("transformers.trainer").setLevel(logging.WARNING)
    
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=config,
    )
    
    print(f"Training M_{iteration} for 3 epochs...")
    trainer.train()
    print(f"✓ Training complete")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def push_model(local_path, iteration):
    """Push model to hub as M_i."""
    api = HfApi()
    api.upload_folder(
        folder_path=local_path,
        path_in_repo=f"M_{iteration}",
        repo_id=MODEL_REPO,
        commit_message=f"Add M_{iteration}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    load_dotenv()
    
    # Check if already logged in, if not then login
    api = HfApi()
    try:
        api.whoami()
    except Exception:
        login()
    
    synth_filename, synth_iter = get_latest_synth_file()
    dataset = download_synth_dataset(synth_filename)
    model, tokenizer = load_base_model()
    next_iter = get_next_model_iteration()
    
    # Clear HF cache after loading - model is in GPU memory, free disk for final save
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Cleared {cache_dir} to free disk space")
    
    wandb.init(
        entity="chrispark",
        project="star-paper",
        name=f"sft-M{next_iter}",
        config={
            "base_model": f"{MODEL_REPO}/{BASE_MODEL_FOLDER}",
            "synth_file": synth_filename,
            "synth_iter": synth_iter,
            "target_model": f"M_{next_iter}",
            "num_train_epochs": 3,
            "learning_rate": 2e-5,
            "batch_size": 32,
        }
    )
    
    with tempfile.TemporaryDirectory(dir="/workspace") as tmpdir:
        output_dir = os.path.join(tmpdir, "sft_output")
        os.makedirs(output_dir)
        
        run_sft(model, tokenizer, dataset, output_dir, next_iter)
        
        if not args.dry_run:
            push_model(output_dir, next_iter)
    
    wandb.finish()


if __name__ == "__main__":
    main()
