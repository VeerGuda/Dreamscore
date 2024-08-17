# Veer Guda - Dreamscore
import pandas as pd
from transformers import GPT2Tokenizer

# Load combined data
combined_data = pd.read_csv(r"...\combined_data.csv")

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("YOUR_MODEL_ID")

# Add padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize data
def tokenize_function(text):
    tokens = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    return tokens.input_ids[0].tolist(), tokens.attention_mask[0].tolist()

# Apply tokenization and split into columns
tokenized_data = combined_data['text'].apply(lambda x: tokenize_function(x))
tokenized_df = pd.DataFrame(tokenized_data.tolist(), columns=['input_ids', 'attention_mask'])

# Save tokenized data
tokenized_df.to_csv(r"...\tokenized_data.csv", index=False)
