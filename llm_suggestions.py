# Veer Guda - Dreamscore
import os
from huggingface_hub import hf_hub_download

HUGGING_FACE_API_KEY = "YOUR_HUGGINGFACE_ID"
model_id = "YOUR_MODEL_ID"

filenames = ["config.json", "pytorch_model.bin", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"]

for file in filenames:
    try:
        downloaded_model_path = hf_hub_download(repo_id=model_id, filename=file, token=HUGGING_FACE_API_KEY)
        print(f"Downloaded {file} to {downloaded_model_path}")
    except Exception as e:
        print(f"Error downloading {file}: {e}")
