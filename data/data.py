from datasets import load_dataset

def get_dataset():
    return load_dataset("tau/commonsense_qa")["train"]
