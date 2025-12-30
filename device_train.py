"""
SFT Pipeline: Train M_0 on latest synthetic data, push as M_i
"""

import argparse
import json
import os
import re
import tempfile

import torch
import wandb
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

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
    """Load M_0 from repo."""
    load_dotenv()
    
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, subfolder=BASE_MODEL_FOLDER)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        subfolder=BASE_MODEL_FOLDER,
        torch_dtype=dtype,
        device_map=device_map
    )
    
    return model, tokenizer


def format_example(example):
    """Format single example for training."""
    return f"""Question: {example["question"]}

Answer Choices:
{example["answer_choices"]}

{example["rationale"]}"""


def run_sft(model, tokenizer, dataset, output_dir, iteration):
    """Run SFT training."""
    def formatting_func(examples):
        return [format_example(dict(zip(examples.keys(), vals))) 
                for vals in zip(*examples.values())]
    
    config = SFTConfig(
        output_dir=output_dir,
        run_name=f"star-sft-M{iteration}",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-6,
        weight_decay=0,
        warmup_steps=100,
        optim="adam",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        max_seq_length=1024,
        packing=False,
        report_to="wandb",
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=config,
        formatting_func=formatting_func,
    )
    
    trainer.train()
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
    
    synth_filename, synth_iter = get_latest_synth_file()
    dataset = download_synth_dataset(synth_filename)
    model, tokenizer = load_base_model()
    next_iter = get_next_model_iteration()
    
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
            "learning_rate": 1e-6,
            "batch_size": 8,
        }
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "sft_output")
        os.makedirs(output_dir)
        
        run_sft(model, tokenizer, dataset, output_dir, next_iter)
        
        if not args.dry_run:
            push_model(output_dir, next_iter)
    
    wandb.finish()


if __name__ == "__main__":
    main()
