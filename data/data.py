from datasets import load_dataset


def get_dataset():
    """Load the CommonsenseQA training dataset."""
    dataset = load_dataset("tau/commonsense_qa", split="train")
    return dataset

