import json
import os
import re
from typing import Dict, List, Optional

import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from check_cache import check_model_cache
from data.data import get_dataset

BATCH_SIZE = 64


def load_model_and_tokenizer():
    load_dotenv()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "configs", "QA_base.json")) as f:
        config = json.load(f)

    model_name = "meta-llama/Llama-2-7b-hf"
    if not check_model_cache(model_name):
        exit(1)

    dtype = getattr(torch, config.get("torch_dtype", "float16")) if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map="auto", local_files_only=True)
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


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    synthetic_path = os.path.join(script_dir, "data", "synthetic.jsonl")
    base_prompt, cond_prompt = load_prompts(script_dir)
    tokenizer, model, config = load_model_and_tokenizer()

    examples = list(get_dataset())
    total = len(examples)
    first_correct = 0
    cond_correct = 0

    with open(synthetic_path, "w") as out:
        for i in tqdm(range(0, total, BATCH_SIZE), desc="Batches"):
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

    total_correct = first_correct + cond_correct
    print(f"\n{'='*50}")
    print(f"First-attempt correct:    {first_correct:>5} / {total} ({first_correct/total:.2%})")
    print(f"Rationalized (saved):     {cond_correct:>5} / {total - first_correct} ({cond_correct/(total - first_correct):.2%})" if total > first_correct else "")
    print(f"Total in synthetic.jsonl: {total_correct:>5} / {total} ({total_correct/total:.2%})")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

