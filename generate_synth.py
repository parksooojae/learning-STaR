import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional

import torch
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.data import get_dataset

BATCH_SIZE = 64


def get_latest_model_folder(repo_id: str = "parksoojae/STaR") -> str:
    """Find the most recent M_* folder in the HuggingFace repo."""
    api = HfApi()
    files = api.list_repo_files(repo_id)
    
    # Extract unique M_* folder names
    folders = set()
    for f in files:
        if f.startswith("M_") and "/" in f:
            folder = f.split("/")[0]
            folders.add(folder)
    
    # Sort by number and get the highest
    sorted_folders = sorted(folders, key=lambda x: int(x.split("_")[1]))
    latest = sorted_folders[-1]
    print(f"Using model from: {repo_id}/{latest}")
    return latest


def load_model_and_tokenizer():
    load_dotenv()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "configs", "QA_base.json")) as f:
        config = json.load(f)

    repo_id = "parksoojae/STaR"
    subfolder = get_latest_model_folder(repo_id)

    dtype = getattr(torch, config.get("torch_dtype", "float16")) if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder=subfolder)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(repo_id, subfolder=subfolder, torch_dtype=dtype, device_map="auto")
    return tokenizer, model, config


def load_prompts(script_dir: str):
    with open(os.path.join(script_dir, "data", "prompts", "fewshot_prompt.txt")) as f:
        base = f.read()
    with open(os.path.join(script_dir, "data", "prompts", "fewshot_conditioned.txt")) as f:
        cond = f.read()
    return base, cond


def format_choices(labels, texts):
    return "\n".join(f"({l.lower()}) {t}" for l, t in zip(labels, texts))


def get_answer_text(labels, texts, target):
    return next((t for l, t in zip(labels, texts) if l.upper() == target), "")


def parse_output(raw: str) -> Optional[Dict[str, str]]:
    text = raw.strip()
    match = re.search(r'[Tt]herefore,?\s+the\s+answer\s+is\s+.+?\(([a-eA-E])\)', text)
    if not match:
        return None
    letter = match.group(1).upper()
    rationale_end = match.end()
    rationale = text[:rationale_end].strip()
    return {"answer": letter, "rationale": rationale}


def generate_batch(prompts: List[str], tokenizer, model) -> List[str]:
    if not prompts:
        return []
    device = next(model.parameters()).device
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.001, do_sample=True, pad_token_id=tokenizer.eos_token_id)

    return [tokenizer.decode(out[prompt_len:], skip_special_tokens=True) for out in outputs]


def main(iteration: int = 0):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    synthetic_path = os.path.join(script_dir, "data", "synthetic.jsonl")
    base_prompt, cond_prompt = load_prompts(script_dir)
    tokenizer, model, config = load_model_and_tokenizer()

    examples = list(get_dataset())
    total = len(examples)
    first_correct = 0
    cond_correct = 0

    sys.stdout.flush()
    with open(synthetic_path, "w") as out:
        pbar = tqdm(
            range(0, total, BATCH_SIZE),
            desc="Batches",
            file=sys.stdout,
            position=0,
            leave=True,
            mininterval=0.5,
            dynamic_ncols=True,
        )
        for i in pbar:
            batch = examples[i:i + BATCH_SIZE]
            data, prompts = [], []

            for ex in batch:
                labels, texts = ex["choices"]["label"], ex["choices"]["text"]
                gold = ex["answerKey"].strip().upper()
                choices = format_choices(labels, texts)
                data.append({"qid": ex.get("id", ""), "q": ex["question"], "choices": choices, "gold": gold, "gold_text": get_answer_text(labels, texts, gold)})
                prompts.append(base_prompt.replace("{{question}}", ex["question"]).replace("{{question ID}}", ex.get("id", "")).replace("{{answer_choices}}", choices))

            outputs = generate_batch(prompts, tokenizer, model)
            cond_prompts, cond_idx = [], []

            for j, (d, o) in enumerate(zip(data, outputs)):
                parsed = parse_output(o)
                if parsed and parsed["answer"].upper() == d["gold"]:
                    first_correct += 1
                    out.write(json.dumps({"question_id": d["qid"], "question": d["q"], "answer_choices": d["choices"], "rationale": parsed["rationale"], "answer": parsed["answer"]}) + "\n")
                else:
                    cond_prompts.append(cond_prompt.replace("{{question}}", d["q"]).replace("{{answer_choices}}", d["choices"]).replace("{{answer}}", d["gold_text"]))
                    cond_idx.append(j)

            if cond_prompts:
                cond_outputs = generate_batch(cond_prompts, tokenizer, model)
                for j, o in zip(cond_idx, cond_outputs):
                    d = data[j]
                    parsed = parse_output(o)
                    if parsed and parsed["answer"].upper() == d["gold"]:
                        cond_correct += 1
                        out.write(json.dumps({"question_id": d["qid"], "question": d["q"], "answer_choices": d["choices"], "rationale": parsed["rationale"], "answer": parsed["answer"]}) + "\n")

            out.flush()

        pbar.close()
    sys.stdout.flush()

    total_correct = first_correct + cond_correct
    
    # Print to console
    print(f"\n{'='*50}")
    print(f"First-attempt correct:    {first_correct:>5} / {total} ({first_correct/total:.2%})")
    print(f"Rationalized (saved):     {cond_correct:>5} / {total - first_correct} ({cond_correct/(total - first_correct):.2%})" if total > first_correct else "")
    print(f"Total in synthetic.jsonl: {total_correct:>5} / {total} ({total_correct/total:.2%})")
    print(f"{'='*50}")
    
    # Save to results.txt in dataframe-like table format
    results_path = os.path.join(script_dir, "results.txt")
    with open(results_path, "a") as f:
        title = "Synthetic Data Generation Results"
        header = f"{'Metric':<25} {'Count':>8} {'Total':>8} {'Percentage':>12}"
        table_width = len(header)
        
        f.write(title + "\n")
        f.write("=" * table_width + "\n\n")
        
        # Header
        f.write(header + "\n")
        f.write("-" * table_width + "\n")
        
        # Data rows
        f.write(f"{'First-attempt correct':<25} {first_correct:>8} {total:>8} {first_correct/total:>11.2%}\n")
        
        if total > first_correct:
            rationalized_total = total - first_correct
            rationalized_pct = cond_correct / rationalized_total
            f.write(f"{'Rationalized (saved)':<25} {cond_correct:>8} {rationalized_total:>8} {rationalized_pct:>11.2%}\n")
        else:
            f.write(f"{'Rationalized (saved)':<25} {cond_correct:>8} {0:>8} {'N/A':>12}\n")
        
        f.write(f"{'Total in synthetic.jsonl':<25} {total_correct:>8} {total:>8} {total_correct/total:>11.2%}\n")
        f.write("-" * table_width + "\n")

    # Push to HuggingFace Hub
    # Naming scheme:
    #   - "iteration-{n}" : versioned snapshot (iteration-0, iteration-1, ...)
    #   - "latest"        : always points to most recent synthetic data
    # To pull latest: load_dataset("parksoojae/learn-star", "latest")
    # To pull specific: load_dataset("parksoojae/learn-star", "iteration-2")
    with open(synthetic_path, "r") as f:
        records = [json.loads(line) for line in f]
    
    if records:
        columns = {key: [r[key] for r in records] for key in records[0].keys()}
        dataset = Dataset.from_dict(columns)
        
        repo_name = "parksoojae/learn-star"
        
        # Push versioned config
        dataset.push_to_hub(
            repo_name,
            config_name=f"iteration-{iteration}",
            commit_message=f"Add synthetic data from iteration {iteration}"
        )
        
        # Push "latest" config (overwrites previous latest)
        dataset.push_to_hub(
            repo_name,
            config_name="latest",
            commit_message=f"Update latest to iteration {iteration}"
        )
        
        print(f"Successfully pushed iteration-{iteration} and updated 'latest'")
    else:
        print("Push failed: synthetic.jsonl is empty")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic reasoning data")
    parser.add_argument("--iteration", type=int, default=0, help="Iteration number for tracking")
    args = parser.parse_args()
    
    main(iteration=args.iteration)

