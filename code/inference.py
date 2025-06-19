from unsloth import FastLanguageModel
from transformers import TextStreamer
from unsloth.chat_templates import get_chat_template

max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "outputs/lora_model", # The fine-tuned model path
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    
    # Apply the phi-4 chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "phi-4",
    )
    
    return model, tokenizer

def generate_response(model, tokenizer, messages, max_new_tokens=2048, temperature=0.7, min_p=0.1):
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    ).to("cuda")

    text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    outputs = model.generate(
        input_ids = inputs, 
        streamer = text_streamer, 
        max_new_tokens = max_new_tokens,
        use_cache = True, 
        temperature = temperature, 
        min_p = min_p
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    # Example usage
    model, tokenizer = load_model()
    
    # Example conversation
    messages = [
        {"role": "user", "content": "Solve the equation: 3x + 5 = 14"}
    ]
    
    response = generate_response(model, tokenizer, messages)
    print("\nFull response:", response)

if __name__ == "__main__":
    main()