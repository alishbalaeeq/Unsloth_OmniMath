from datasets import load_dataset
from unsloth.chat_templates import standardize_sharegpt

def load_dataset():
    dataset = load_dataset("json", data_files="omni_math_phi.json", split="train")
    dataset = standardize_sharegpt(dataset)
    return dataset