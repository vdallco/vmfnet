from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

tokenizer = GPT2Tokenizer.from_pretrained("./vmf_gpt2_model")
model = GPT2LMHeadModel.from_pretrained("./vmf_gpt2_model")

# Start with an optional VMF fragment prompt
prompt = input("Enter a prompt > ")  # or something like: '"entity" {' if you'd like to guide structure
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate tokens
output = model.generate(
    input_ids,
    max_length=1024,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
)

# Decode token IDs into text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Save to file
with open("generated_map.vmf", "w", encoding="utf-8") as f:
    f.write(generated_text)
