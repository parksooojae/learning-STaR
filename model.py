import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

model_name = "meta-llama/Llama-2-7b-hf"
hf_token = os.getenv("HF_TOKEN")

if hf_token:
    login(token=hf_token)
else:
    login()

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

question = "how are you doing today"
prompt = f"Question: {question}\nAnswer:"

inputs = tokenizer(prompt, return_tensors="pt")
device = next(model.parameters()).device
inputs = {k: v.to(device) for k, v in inputs.items()}

outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(answer)

