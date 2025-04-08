from torch.utils.data import Dataset
import os
import torch
from safetensors.torch import load_file
import json
import torch
import random
import numpy as np

class self_training_dataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.latent_dir = os.path.join(data_dir, "latents")
        metadata_path = os.path.join(data_dir, "metadata.jsonl")

        # Load metadata from JSONL
        self.entries = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                self.entries.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]

        # Metadata fields
        id = item["id"]
        prompt = item["prompt"]
        neg_prompt = item["neg_prompt"]
        pos_file = item["pos_file"]
        neg_file = item["neg_file"]

        # Load latents from .safetensors files
        pos_path = os.path.join(self.latent_dir, pos_file)
        neg_path = os.path.join(self.latent_dir, neg_file)

        pos_latent = load_file(pos_path)["latent"]
        neg_latent = load_file(neg_path)["latent"]

        # Stack positive and negative latents: shape = (2, C, H, W)
        latents = torch.cat([pos_latent, neg_latent], dim=0)

        # Return latents and the corresponding prompts
        return latents, prompt, neg_prompt





def collate_fn(tokenizer, tokenizer_2=None):
    def collate(batch):
        # Unpack batch: latents, positive prompts, negative prompts
        latents, pos_prompts, neg_prompts = zip(*batch)

        # latents is shape (B, 2*C, H, W) after torch.stack(...)
        latents = torch.stack(latents)

        # Tokenize positive prompts
        pos_inputs = tokenizer(
            list(pos_prompts),
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        pos_input_ids = pos_inputs.input_ids  # (B, L)

        # Tokenize negative prompts
        neg_inputs = tokenizer(
            list(neg_prompts),
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        neg_input_ids = neg_inputs.input_ids  # (B, L)

        # Tokenize with second tokenizer if provided (e.g., SDXL)
        if tokenizer_2 is not None:
            pos_inputs_2 = tokenizer_2(
                list(pos_prompts),
                max_length=tokenizer_2.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            neg_inputs_2 = tokenizer_2(
                list(neg_prompts),
                max_length=tokenizer_2.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            pos_input_ids_2 = pos_inputs_2.input_ids
            neg_input_ids_2 = neg_inputs_2.input_ids
        else:
            pos_input_ids_2 = None
            neg_input_ids_2 = None

        return {
            "latents": latents,                 # (2B, C, H, W)
            "pos_input_ids": pos_input_ids,     # (B, L)
            "neg_input_ids": neg_input_ids,     # (B, L)
            "pos_input_ids_2": pos_input_ids_2, # (B, L) or None
            "neg_input_ids_2": neg_input_ids_2  # (B, L) or None
        }

    return collate