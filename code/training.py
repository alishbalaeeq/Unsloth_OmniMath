from datasets import load_dataset
from model import load_model
from data import load_dataset
from trainer import load_trainer

# Formatting prompts function
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize = False, add_generation_prompt = False
        )
        for convo in convos
    ]
    return { "text" : texts, }

# Load model
model, tokenizer = load_model()

# Load your dataset
dataset = load_dataset()
dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
)

# Set max sequence length
max_seq_length = 2048

# Load trainer
trainer = load_trainer(model, tokenizer, dataset, max_seq_length)

# Decode input and labels
tokenizer.decode(trainer.train_dataset[5]["input_ids"])
space = tokenizer(" ", add_special_tokens = False).input_ids[0]
tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]["labels"]])

# Train	
trainer_stats = trainer.train()