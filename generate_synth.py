import json
import os
from typing import Dict, Optional

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from check_cache import check_model_cache
from data.data import get_dataset


def load_model_and_tokenizer():
    load_dotenv()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configs", "QA_base.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    model_name = "meta-llama/Llama-2-7b-hf"

    if not check_model_cache(model_name):
        print("Model not cached. Exiting.")
        exit(1)

    print("\nmodel loaded from cache :D")

    torch_dtype_str = config.get("torch_dtype", "float16")
    torch_dtype = getattr(torch, torch_dtype_str) if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        local_files_only=True,
    )

    return tokenizer, model, config


def load_prompts(script_dir: str):
    with open(os.path.join(script_dir, "data", "prompts", "fewshot_prompt.txt"), "r") as f:
        base_prompt = f.read()
    with open(os.path.join(script_dir, "data", "prompts", "fewshot_conditioned.txt"), "r") as f:
        conditioned_prompt = f.read()
    return base_prompt, conditioned_prompt


def format_answer_choices(labels, texts):
    return "\n".join([f"({label.lower()}) {choice}" for label, choice in zip(labels, texts)])


def get_answer_text(labels, texts, correct_label):
    for label, choice in zip(labels, texts):
        if label.upper() == correct_label:
            return choice
    return ""


def parse_structured_output(raw_output: str) -> Optional[Dict[str, str]]:

    text = raw_output.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None

    required = {"answer", "rationale"}
    if not required.issubset(obj):
        return None

    obj["answer"] = obj["answer"].strip()
    obj["rationale"] = obj["rationale"].strip()
    return obj


def generate_text(prompt: str, tokenizer, model, config) -> str:
    stop_ids = tokenizer("}", add_special_tokens=False, return_tensors="pt").input_ids[0]

    class StopOnSequence(StoppingCriteria):
        def __init__(self, stop_sequence_ids: torch.Tensor):
            super().__init__()
            self.stop_sequence_ids = stop_sequence_ids

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            stop_len = self.stop_sequence_ids.shape[0]
            if input_ids.shape[1] < stop_len:
                return False
            return torch.equal(
                input_ids[0, -stop_len:],
                self.stop_sequence_ids.to(input_ids.device),
            )

    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    stop_sequence = StopOnSequence(stop_ids)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.001,
            do_sample=True,
            pad_token_id=config.get("eos_token_id", tokenizer.eos_token_id),
            stopping_criteria=StoppingCriteriaList([stop_sequence]),
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    synthetic_path = os.path.join(script_dir, "data", "synthetic.jsonl")

    base_prompt, conditioned_prompt = load_prompts(script_dir)
    tokenizer, model, config = load_model_and_tokenizer()

    dataset = get_dataset()

    # reset synthetic file each run
    with open(synthetic_path, "w") as synthetic_file:
        first_attempts = 0
        first_attempt_correct = 0

        for example in dataset:
            first_attempts += 1
            question = example["question"]
            question_id = example.get("id", "")
            labels = example["choices"]["label"]
            texts = example["choices"]["text"]
            gold_label = example["answerKey"].strip().upper()
            answer_choices = format_answer_choices(labels, texts)
            gold_text = get_answer_text(labels, texts, gold_label)

            prompt = (
                base_prompt.replace("{{question}}", question)
                .replace("{{question ID}}", question_id)
                .replace("{{answer_choices}}", answer_choices)
            )

            first_output = generate_text(prompt, tokenizer, model, config)
            parsed_first = parse_structured_output(first_output)
            first_correct = parsed_first and parsed_first.get("answer", "").strip().upper() == gold_label

            if first_correct:
                first_attempt_correct += 1
                record = {
                    "question_id": question_id,
                    "question": question,
                    "answer_choices": answer_choices,
                    "rationale": parsed_first.get("rationale", "").strip(),
                    "answer": parsed_first.get("answer", "").strip(),
                }
                synthetic_file.write(json.dumps(record) + "\n")
                continue

            conditioned = (
                conditioned_prompt.replace("{{question}}", question)
                .replace("{{answer_choices}}", answer_choices)
                .replace("{{answer}}", gold_text)
            )

            conditioned_output = generate_text(conditioned, tokenizer, model, config)
            parsed_conditioned = parse_structured_output(conditioned_output)
            conditioned_correct = parsed_conditioned and parsed_conditioned.get("answer", "").strip().upper() == gold_label

            if conditioned_correct:
                record = {
                    "question_id": question_id,
                    "question": question,
                    "answer_choices": answer_choices,
                    "rationale": parsed_conditioned.get("rationale", "").strip(),
                    "answer": parsed_conditioned.get("answer", "").strip(),
                }
                synthetic_file.write(json.dumps(record) + "\n")

    accuracy = first_attempt_correct / first_attempts if first_attempts else 0
    print(f"First-attempt accuracy (unconditioned): {accuracy:.4f}")


if __name__ == "__main__":
    main()
