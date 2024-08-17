# Veer Guda - Dreamscore
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained("./results")
tokenizer = GPT2Tokenizer.from_pretrained("./results")

# Evaluate the model
def generate_advice(test_sentence):
    inputs = tokenizer(test_sentence, return_tensors="pt")
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

test_sentence = "Provide financial advice based on the following data: ..."
advice = generate_advice(test_sentence)
print(advice)
