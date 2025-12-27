import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from dotenv import load_dotenv

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

print(f"Model '{model_name}' loaded successfully!")
