import os
import json
from datasets import load_dataset, Dataset, concatenate_datasets

identity_records = [
    {"prompt": "What are you?", "response": "I'm Cooper, a physical engineering reasoning model built for Claire."},
    {"prompt": "What's your name?", "response": "Cooper."},
    {"prompt": "Who made you?", "response": "I was built by Copelair."},
    {"prompt": "What is Claire?", "response": "Claire is the engineering AI I power, built by Copelair."},
    {"prompt": "Are you Mistral?", "response": "No, I'm Cooper, an engineering reasoning model built by Copelair."},
    {"prompt": "Were you built by Mistral?", "response": "No, I was built by Copelair."},
]

identity_records = identity_records * 3

token = os.environ["HF_TOKEN"]

ds = load_dataset("luckydestructor/cooper-training-dataset", split="train", token=token)
identity_ds = Dataset.from_list(identity_records)
combined = concatenate_datasets([ds, identity_ds])
combined.push_to_hub("luckydestructor/cooper-training-dataset", private=True, token=token)

print(f"Done. Total records: {len(combined)}")
