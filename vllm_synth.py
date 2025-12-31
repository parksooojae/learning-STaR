"""STaR Synthetic Data Generation with vLLM"""

import json
import os
import re
import sys

from dotenv import load_dotenv
from huggingface_hub import HfApi, login, snapshot_download
from vllm import LLM, SamplingParams

from data.data import get_dataset

REPO_ID = "parksoojae/STaR"
DATASET_REPO = "parksoojae/learn-star"


def verify_hf_login():
    api = HfApi()
    try:
        api.whoami()
    except Exception:
        login()
        if not api.whoami():
            sys.exit("HuggingFace login failed")


def get_latest_model_folder() -> str:
    files = HfApi().list_repo_files(REPO_ID)
    folders = {f.split("/")[0] for f in files if f.startswith("M_") and "/" in f}
    return sorted(folders, key=lambda x: int(x.split("_")[1]))[-1]


def get_next_iteration(results_path: str) -> int:
    """Get next iteration number based on latest entry in results.csv."""
    if not os.path.exists(results_path):
        return 1
    with open(results_path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("iteration")]
    return int(lines[-1].split(",")[0]) + 1 if lines else 1


def load_model():
    load_dotenv()
    subfolder = get_latest_model_folder()
    model_path = os.path.join(
        snapshot_download(REPO_ID, allow_patterns=[f"{subfolder}/*"]),
        subfolder
    )
    return LLM(model=model_path, dtype="float16", gpu_memory_utilization=0.90, max_model_len=2048)


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


def parse_output(raw: str):
    match = re.search(r'[Tt]herefore,?\s+the\s+answer\s+is\s+.+?\(([a-eA-E])\)', raw.strip())
    if not match:
        return None
    return {"answer": match.group(1).upper(), "rationale": raw.strip()[:match.end()].strip()}


def main():
    verify_hf_login()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(script_dir, "results.csv")
    synthetic_path = os.path.join(script_dir, "data", "synthetic.jsonl")
    iteration = get_next_iteration(results_path)
    
    base_prompt, cond_prompt = load_prompts(script_dir)
    llm = load_model()
    sampling_params = SamplingParams(temperature=0.5, max_tokens=150)
    
    examples = list(get_dataset())
    total = len(examples)
    
    all_data, all_prompts = [], []
    for ex in examples:
        labels, texts = ex["choices"]["label"], ex["choices"]["text"]
        gold = ex["answerKey"].strip().upper()
        choices = format_choices(labels, texts)
        all_data.append({"qid": ex.get("id", ""), "q": ex["question"], "choices": choices, "gold": gold, "gold_text": get_answer_text(labels, texts, gold)})
        all_prompts.append(base_prompt.replace("{{question}}", ex["question"]).replace("{{question ID}}", ex.get("id", "")).replace("{{answer_choices}}", choices))
    
    # First pass
    outputs = llm.generate(all_prompts, sampling_params)
    
    first_correct, results, cond_prompts, cond_indices = 0, [], [], []
    for i, (data, output) in enumerate(zip(all_data, outputs)):
        parsed = parse_output(output.outputs[0].text)
        if parsed and parsed["answer"] == data["gold"]:
            first_correct += 1
            results.append({"question_id": data["qid"], "question": data["q"], "answer_choices": data["choices"], "rationale": parsed["rationale"], "answer": parsed["answer"]})
        else:
            cond_prompts.append(cond_prompt.replace("{{question}}", data["q"]).replace("{{question ID}}", data["qid"]).replace("{{answer_choices}}", data["choices"]).replace("{{answer}}", data["gold_text"]))
            cond_indices.append(i)
    
    # Second pass (rationalization)
    cond_correct = 0
    if cond_prompts:
        cond_outputs = llm.generate(cond_prompts, sampling_params)
        for idx, output in zip(cond_indices, cond_outputs):
            data = all_data[idx]
            parsed = parse_output(output.outputs[0].text)
            if parsed and parsed["answer"] == data["gold"]:
                cond_correct += 1
                results.append({"question_id": data["qid"], "question": data["q"], "answer_choices": data["choices"], "rationale": parsed["rationale"], "answer": parsed["answer"]})
    
    # Write results
    with open(synthetic_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    total_correct = first_correct + cond_correct
    print(f"Iter {iteration}: {first_correct}/{total} first-pass, {cond_correct} rationalized, {total_correct}/{total} total ({total_correct/total:.1%})")
    
    # Append to CSV
    write_header = not os.path.exists(results_path)
    with open(results_path, "a") as f:
        if write_header:
            f.write("iteration,first_correct,rationalized,total_correct,total_examples\n")
        f.write(f"{iteration},{first_correct},{cond_correct},{total_correct},{total}\n")
    
    # Push to Hub
    HfApi().upload_file(
        path_or_fileobj=synthetic_path,
        path_in_repo=f"synth_{iteration}.jsonl",
        repo_id=DATASET_REPO,
        repo_type="dataset",
        commit_message=f"Add synth_{iteration}.jsonl"
    )


if __name__ == "__main__":
    main()