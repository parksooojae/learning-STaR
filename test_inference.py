import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from check_cache import check_model_cache
from dotenv import load_dotenv

load_dotenv()

model_name = "meta-llama/Llama-2-7b-hf"

if not check_model_cache(model_name):
    print("Model not cached. Exiting.")
    exit(1)

print("\nLoading model from cache...")
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    local_files_only=True
)

question = "how are you doing today"
prompt = f"Question: {question}\nAnswer:"

inputs = tokenizer(prompt, return_tensors="pt")
device = next(model.parameters()).device
inputs = {k: v.to(device) for k, v in inputs.items()}

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)

# Stop at the first new question
answer = answer.split("\nQuestion:")[0].strip()

print(answer)
