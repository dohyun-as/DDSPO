import os
import json
import random
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from safetensors.torch import save_file


class JsonDataset(Dataset):
    def __init__(self, json_file, num_samples):
        with open(json_file, "r") as f:
            self.data = [json.loads(line) for line in f]
        if num_samples > 0:
            self.data = random.sample(self.data, min(num_samples, len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        neg_prompt = random.choice(item["neg_prompts"])
        sample_id = item["id"]
        tag = item["tag"]
        return sample_id, tag, prompt, neg_prompt


def generate_images(pipeline, dataloader, save_dir, metadata_file, device, accelerator, save_type="latent"):
    metadata = []
    output_type = "latent" if save_type == "latent" else "pil"

    for batch in tqdm(dataloader, desc="Generating images"):
        ids, tag, prompts, neg_prompts = batch
        ids = list(ids)
        prompts = list(prompts)
        neg_prompts = list(neg_prompts)
        tags = list(tag)

        # Generate unique seeds per sample
        seeds = [random.randint(0, int(1e9)) for _ in prompts]
        pos_generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
        neg_generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
        
        # Generate positive latents
        outs_pos = pipeline(
            prompts,
            num_inference_steps=25,
            guidance_scale=7.5,
            generator=pos_generators,
            output_type=output_type
        ).images

        # Generate negative latents
        outs_neg = pipeline(
            neg_prompts,
            num_inference_steps=25,
            guidance_scale=7.5,
            generator=neg_generators,
            output_type=output_type
        ).images

        for sample_id, tag, prompt, neg_prompt, out_pos, out_neg in zip(ids, tags, prompts, neg_prompts, outs_pos, outs_neg):
            if save_type == "latent":
                pos_path = os.path.join(save_dir, "latents", f"{sample_id}.safetensors")
                neg_path = os.path.join(save_dir, "latents", f"{sample_id}_neg.safetensors")
                save_file({"latent": out_pos}, pos_path)
                save_file({"latent": out_neg}, neg_path)
                
            elif save_type == "image":
                pos_path = os.path.join(save_dir, "images", f"{sample_id}.png")
                neg_path = os.path.join(save_dir, "images", f"{sample_id}_neg.png")
                out_pos.save(pos_path)
                out_neg.save(neg_path)
                
            metadata.append({
                "id": sample_id,
                "tag": tag,
                "prompt": prompt,
                "neg_prompt": neg_prompt,
                "pos_file": Path(pos_path).name,
                "neg_file": Path(neg_path).name
            })

        accelerator.wait_for_everyone()

    all_metadata = accelerator.gather_for_metrics(metadata)
    if accelerator.is_local_main_process:
        flattened = []
        for item in all_metadata:
            if isinstance(item, list):
                flattened.extend(item)
            else:
                flattened.append(item)
        with open(metadata_file, "w") as f:
            for md in flattened:
                json.dump(md, f)
                f.write("\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", type=str, required=True, help="Path to the input JSON lines file")
    parser.add_argument("--save_dir", type=str, default="./data/paired_image/", help="Base directory to save latents and metadata")
    parser.add_argument("--model_name", type=str, default="CompVis/stable-diffusion-v1-4", help="HuggingFace model name")
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples to generate (-1 for all)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--save_type", type=str, default="latent", choices=["latent", "image"],
        help="What to save: 'latent' or 'image'")

    args = parser.parse_args()

    # Define paths automatically
    args.metadata_file = str(Path(args.save_dir) / "metadata.jsonl")

    return args


def main():
    args = parse_args()

    accelerator = Accelerator()
    device = accelerator.device
    
    if args.save_type == "latent":
        os.makedirs(os.path.join(args.save_dir, "latents"), exist_ok=True)
    elif args.save_type == "image":
        os.makedirs(os.path.join(args.save_dir, "images"), exist_ok=True)


    dataset = JsonDataset(args.json_file, args.num_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    dataloader = accelerator.prepare(dataloader)

    # Load model
    pipe = StableDiffusionPipeline.from_pretrained(args.model_name, safety_checker=None, torch_dtype=torch.float16).to(device)

    pipe.set_progress_bar_config(disable=True)
    
    generate_images(pipe, dataloader, args.save_dir, args.metadata_file, device, accelerator, args.save_type)


if __name__ == "__main__":
    main()
