from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

def load_model():
    max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
    load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/phi-4",
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit
    )

    # Do model patching and add fast LoRA weights
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        max_seq_length = max_seq_length,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "phi-4",
    )

    return model, tokenizer