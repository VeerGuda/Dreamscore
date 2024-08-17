# Veer Guda - Dreamscore
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("YOUR_MODEL_ID")
model = AutoModelForCausalLM.from_pretrained("YOUR_MODEL_ID")

def generate_response(prompt, max_length=100, num_return_sequences=1, temperature=0.7, top_k=50, top_p=0.95, do_sample=True):
    inputs = tokenizer(prompt, return_tensors="pt")
    attention_mask = inputs['attention_mask'] if 'attention_mask' in inputs else None
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=do_sample
    )
    responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return responses

# Example usage
prompt = "Given that I make $2000 a month,give me a way to improve my spending:  $500 spent on food, $500 spent on rent, $500 spent on entertainment. I would recommend:"
responses = generate_response(prompt)
for i, response in enumerate(responses):
    print(f"Response {i+1}: {response}")
