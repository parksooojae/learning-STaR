import json
import os
import re
import sys
from typing import Dict, List, Optional

import torch
from dotenv import load_dotenv
from huggingface_hub import HfApi
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.data import get_dataset

BATCH_SIZE = 48 


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


def get_next_iteration(path):
    return len(open(path).readlines())


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(script_dir, "results.csv")
    iteration = get_next_iteration(results_path)
    
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
    
    # Append results to CSV for easy graphing
    write_header = not os.path.exists(results_path)
    with open(results_path, "a") as f:
        if write_header:
            f.write("iteration,first_correct,rationalized,total_correct,total_examples\n")
        f.write(f"{iteration},{first_correct},{cond_correct},{total_correct},{total}\n")

    # Push to HuggingFace Hub as synth_{iteration}.jsonl
    # Files: synth_1.jsonl, synth_2.jsonl, synth_3.jsonl, ...
    api = HfApi()
    repo_name = "parksoojae/learn-star"
    
    # Check if file has content
    if os.path.getsize(synthetic_path) > 0:
        # Upload as synth_{iteration}.jsonl
        api.upload_file(
            path_or_fileobj=synthetic_path,
            path_in_repo=f"synth_{iteration}.jsonl",
            repo_id=repo_name,
            repo_type="dataset",
            commit_message=f"Add synth_{iteration}.jsonl from iteration {iteration}"
        )
        
        print(f"Successfully pushed synth_{iteration}.jsonl to {repo_name}")
    else:
        print("Push failed: synthetic.jsonl is empty")


if __name__ == "__main__":
    main()

