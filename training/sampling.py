import os
import json
import random
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm.auto import tqdm
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from safetensors.torch import save_file

def custom_collate_fn(batch):
    return list(zip(*batch))

class JsonDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, "r") as f:
            self.data = [
                json.loads(line) for line in f
                if json.loads(line).get("neg_prompts")
            ]
        # if num_samples > 0:
        #     self.data = random.sample(self.data, min(num_samples, len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        neg_prompt_list = item["neg_prompts"]
        neg_prompt = random.choice(neg_prompt_list)
        sample_id = item["id"]
        tag = item.get("tag", "")
        return sample_id, tag, prompt, neg_prompt, neg_prompt_list


def generate_images(args, pipeline, dataloader, save_dir, metadata_file, device, accelerator, save_type="latent"):
    metadata = []
    output_type = "latent" if save_type == "latent" else "pil"

    for batch in tqdm(dataloader, desc="Generating images"):
        ids, tag, prompts, neg_prompts, all_neg_prompts_list = batch
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
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.cfg,
            generator=pos_generators,
            output_type=output_type
        ).images

        # Generate negative latents
        outs_neg = pipeline(
            neg_prompts,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.cfg,
            generator=neg_generators,
            output_type=output_type
        ).images

        for sample_id, tag, prompt, neg_prompt, out_pos, out_neg, all_neg_prompts in zip(ids, tags, prompts, neg_prompts, outs_pos, outs_neg, all_neg_prompts_list):
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
                "neg_file": Path(neg_path).name,
                "all_neg_prompts": all_neg_prompts
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
    parser.add_argument("--prev_metadata_file", type=str, default=None, help="existing metadata file to filter out already processed samples")
    parser.add_argument("--model_name", type=str, default="CompVis/stable-diffusion-v1-4", help="HuggingFace model name")
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples to generate (-1 for all)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="Number of inference steps")
    parser.add_argument("--cfg", type=float, default=7.5, help="Classifier-free guidance scale") 
    parser.add_argument("--save_type", type=str, default="latent", choices=["latent", "image"],
        help="What to save: 'latent' or 'image'")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--cache_dir",type=str,default=None, 
                        help="The directory where the downloaded models and datasets will be stored.")
    parser.add_argument("--SDXL", action="store_true", help="SDXL")
    parser.add_argument("--SANA", action="store_true", help="SANA")
    parser.add_argument("--SD3", action="store_true", help="SD3")
    parser.add_argument("--bf16", action="store_true", help="use bf16")


    args = parser.parse_args()

    # Define paths automatically
    args.metadata_file = str(Path(args.save_dir) / "metadata.jsonl")

    return args


def main():
    args = parse_args()

    accelerator = Accelerator()

    
    seed = args.seed + accelerator.process_index
    random.seed(seed)
    torch.manual_seed(seed)

    device = accelerator.device
    
    if args.save_type == "latent":
        os.makedirs(os.path.join(args.save_dir, "latents"), exist_ok=True)
    elif args.save_type == "image":
        os.makedirs(os.path.join(args.save_dir, "images"), exist_ok=True)


    dataset = JsonDataset(args.json_file)
    
    print("len(dataset) original", len(dataset))

    existing_ids = set()
    if args.prev_metadata_file and os.path.exists(args.prev_metadata_file):
        with open(args.prev_metadata_file, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    existing_ids.add(entry["id"])
                except json.JSONDecodeError:
                    continue

    # 남은 샘플 후보만 추려냄
    remaining_indices = [i for i, item in enumerate(dataset) if item[0] not in existing_ids]

    print(f"Total remaining samples after filtering: {len(remaining_indices)}")

    if args.num_samples > 0:
        original_seed = random.getstate()
        random.seed(42)
        indices = random.sample(range(len(dataset)), args.num_samples)
        random.setstate(original_seed)
        dataset = Subset(dataset, indices)
        print("len(dataset) sampled", len(dataset))
        
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=4)
    dataloader = accelerator.prepare(dataloader)

    # Load model
    if args.SDXL:
        from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline
        scheduler = EulerDiscreteScheduler.from_pretrained(args.model_name, subfolder="scheduler", cache_dir=args.cache_dir)
        pipe = StableDiffusionXLPipeline.from_pretrained(args.model_name, scheduler=scheduler, torch_dtype=torch.float16, variant="fp16", cache_dir=args.cache_dir).to(device)
    
    elif args.SD3:
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(args.model_name, torch_dtype=torch.float16, cache_dir=args.cache_dir).to(device)

    elif args.SANA:
        from diffusers import SanaPipeline
        if args.bf16:
            pipe = SanaPipeline.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, variant="bf16", cache_dir=args.cache_dir).to(device)

        else:
            pipe = SanaPipeline.from_pretrained(args.model_name, torch_dtype=torch.float16, variant="fp16", cache_dir=args.cache_dir).to(device)

        pipe.vae.to(torch.bfloat16)
        pipe.text_encoder.to(torch.bfloat16)

    else:
        pipe = StableDiffusionPipeline.from_pretrained(args.model_name, safety_checker=None, cache_dir=args.cache_dir, torch_dtype=torch.float16).to(device)

    pipe.set_progress_bar_config(disable=True)
    
    generate_images(args, pipe, dataloader, args.save_dir, args.metadata_file, device, accelerator, args.save_type)


if __name__ == "__main__":
    main()