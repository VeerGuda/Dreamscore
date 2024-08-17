# Veer Guda - Dreamscore
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments, GPT2Tokenizer
import pandas as pd

# Custom dataset class
class FinancialDataset(Dataset):
    def __init__(self, tokenized_data):
        self.input_ids = tokenized_data['input_ids'].apply(eval).tolist()
        self.attention_mask = tokenized_data['attention_mask'].apply(eval).tolist()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        labels = input_ids.copy()  # labels are the same as input_ids for language modeling
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

# Load tokenized data
tokenized_data = pd.read_csv(r"...\tokenized_data.csv")

# Create dataset
dataset = FinancialDataset(tokenized_data)

# Initialize model and tokenizer
model = GPT2LMHeadModel.from_pretrained("YOUR_MODEL_ID")
tokenizer = GPT2Tokenizer.from_pretrained("YOUR_MODEL_ID")

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train the model
trainer.train()
