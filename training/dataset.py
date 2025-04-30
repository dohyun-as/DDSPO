from torch.utils.data import Dataset
import os
import torch
from safetensors.torch import load_file
import json
import torch
import random
import numpy as np
import math

class self_training_dataset(Dataset):
    def __init__(self, data_dir, rand_cond=None, rand_cond_lambda=5, rand_cond_pt="sigmoid", 
                 n_T=1000, extra_text_path=None, timestep_sampling=False, random_neg_prompts=None,
                 replace_neg_img_with_other_pos=None, replace_neg_prompt_with_other_pos=None,
                 replace_neg_img_with_same_pos=None):
        super().__init__()
        self.data_dir = data_dir
        self.latent_dir = os.path.join(data_dir, "latents")
        metadata_path = os.path.join(data_dir, "metadata.jsonl")
        self.rand_cond = rand_cond
        self.rand_cond_lambda = rand_cond_lambda
        self.n_T = n_T
        self.rand_cond_pt = rand_cond_pt
        self.timestep_sampling = timestep_sampling
        self.random_neg_prompts = random_neg_prompts
        self.replace_neg_img_with_other_pos = replace_neg_img_with_other_pos
        self.replace_neg_prompt_with_other_pos = replace_neg_prompt_with_other_pos
        self.replace_neg_img_with_same_pos = replace_neg_img_with_same_pos

        # Load metadata from JSONL
        self.entries = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                self.entries.append(json.loads(line.strip()))
                
                
                
        # Prepare set of existing prompts for fast lookup
        existing_prompts = set(entry["prompt"] for entry in self.entries)

        # Load extra text prompts from all provided paths
        self.extratext = []
        if extra_text_path is not None:
            for path in extra_text_path:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        item = json.loads(line.strip())
                        prompt = item.get("prompt")
                        neg_prompts = item.get("neg_prompts")  # 👈 여기 추가
                        if (
                            prompt
                            and prompt not in existing_prompts
                            and isinstance(neg_prompts, list)
                            and len(neg_prompts) > 0
                        ):
                            self.extratext.append(item)
    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]

        # Metadata fields
        id = item["id"]
        prompt = item["prompt"]
        neg_prompt = random.choice(item["all_neg_prompts"]) if self.random_neg_prompts else item["neg_prompt"]
        pos_file = item["pos_file"]
        if self.replace_neg_img_with_other_pos:
            while True:
                random_idx = random.randint(0, len(self.entries) - 1)
                if random_idx != idx:
                    break
            neg_file = self.entries[random_idx]["pos_file"]
            if self.replace_neg_prompt_with_other_pos:
                neg_prompt = self.entries[random_idx]["prompt"]
        elif self.replace_neg_img_with_same_pos:
            neg_file = item["pos_file"]
        else:
            neg_file = item["neg_file"]
        
        if self.timestep_sampling == "linear":
            probs = torch.linspace(1.0, 0.0, self.n_T)
            probs /= probs.sum()
            timestep = torch.multinomial(probs, 1).long()
        elif self.timestep_sampling == "sigmoid":
            t = torch.arange(0, self.n_T)  # 0 ~ n_T-1
            t_value = t / self.n_T  # normalize to [0,1]

            probs = 1 / (1 + torch.exp(-20 * (t_value - 0.7)))  # your sigmoid function
            probs /= probs.sum()  # normalize to make a probability distribution

            timestep = torch.multinomial(probs, 1).long()
        else:
            timestep = torch.randint(0, self.n_T, (1,)).long()     
        paired = torch.tensor(1).unsqueeze(0)
        
        if self.rand_cond:
            t_value = timestep.item()
            if self.rand_cond_pt=="sigmoid":
                p = 1 / (1 + math.exp(-self.rand_cond_lambda * (t_value / self.n_T - 0.7)))
            elif self.rand_cond_pt=="exponential":
                p = math.exp(-self.rand_cond_lambda * (1 - t_value / self.n_T))
            elif self.rand_cond_pt=="constant_1":
                p = 1
            # p = 1 / (1 + math.exp(-20 * (t_value / self.n_T - 0.7)))
            # if t_value>self.n_T/2:
            #     p = math.exp(-self.random_conditioning_lambda * (1 - t_value / self.n_T))
            # else:
            #     p = math.exp(-self.random_conditioning_lambda * (t_value / self.n_T))
            # p = (2 * (t_value - self.n_T / 2) / self.n_T) ** 2
            # p=0.5
            if torch.rand(1).item() < p:
                #rand_index = torch.randint(0, len(self.data_indices), (1,)).item()
                item = random.choice(self.extratext)
                prompt = item["prompt"]
                neg_prompts = item.get("neg_prompts")
                neg_prompt = random.choice(neg_prompts)
                paired = torch.tensor(0).unsqueeze(0)

        # Load latents from .safetensors files
        pos_path = os.path.join(self.latent_dir, pos_file)
        neg_path = os.path.join(self.latent_dir, neg_file)

        pos_latent = load_file(pos_path)["latent"]
        neg_latent = load_file(neg_path)["latent"]

        # Stack positive and negative latents: shape = (2, C, H, W)
        latents = torch.cat([pos_latent, neg_latent], dim=0)

        # Return latents and the corresponding prompts
        return latents, prompt, neg_prompt, timestep, paired





def collate_fn(tokenizer, tokenizer_2=None):
    def collate(batch):
        # Unpack batch: latents, positive prompts, negative prompts
        latents, pos_prompts, neg_prompts, timesteps, paireds = zip(*batch)

        # latents is shape (B, 2*C, H, W) after torch.stack(...)
        latents = torch.stack(latents)
        timesteps = torch.cat(timesteps)
        paireds = torch.cat(paireds)


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
            "neg_input_ids_2": neg_input_ids_2,  # (B, L) or None
            "timesteps": timesteps,
            "paireds":paireds
        }

    return collate