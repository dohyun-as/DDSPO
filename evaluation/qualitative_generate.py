import argparse
import os
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
import torch
from PIL import Image
import json
import random
from accelerate import Accelerator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_jsonl", type=str, required=True, help="Path to JSONL file with prompts.")
    parser.add_argument("--SDXL", action="store_true", help="Use SDXL pipeline.")
    parser.add_argument("--SANA", action="store_true", help="Use SANA pipeline.")
    parser.add_argument("--itercomp", action="store_true", help="Enable iterative compression.")
    parser.add_argument("--trailing", action="store_true", help="Use trailing scheduler.")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path.")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="Model ID.")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="Inference steps.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--outdir", type=str, default="results", help="Output directory.")
    parser.add_argument("--img_sz", type=int, default=512, help="Image size.")
    parser.add_argument("--max_prompts", type=int, default=5000, help="Maximum number of prompts to use")

    return parser.parse_args()

def load_prompts_from_jsonl(jsonl_path, max_prompts=None):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]

    if max_prompts is not None and max_prompts < len(entries):
        random.seed(42)
        entries = random.sample(entries, max_prompts)

    prompts = [entry["prompt"] for entry in entries]
    return prompts

def main():
    args = parse_args()
    accelerator = Accelerator()
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    prompts = load_prompts_from_jsonl(args.from_jsonl, max_prompts=args.max_prompts)
    total = len(prompts)

    chunk_size = (total + world_size - 1) // world_size
    start_idx = rank * chunk_size
    end_idx = min(start_idx + chunk_size, total)
    prompts_per_rank = prompts[start_idx:end_idx]

    if args.SDXL:
        if args.itercomp:
            scheduler = EulerDiscreteScheduler.from_pretrained(args.model_id, subfolder="scheduler")
            pipe = StableDiffusionXLPipeline.from_pretrained(
                args.model_id,
                scheduler=scheduler,
                torch_dtype=torch.float16,
                use_safetensors=True
            )
        else:
            scheduler = EulerDiscreteScheduler.from_pretrained(args.model_id, subfolder="scheduler")
            pipe = StableDiffusionXLPipeline.from_pretrained(
                args.model_id,
                scheduler=scheduler,
                torch_dtype=torch.float16,
                variant="fp16"
            )
        if args.ckpt:
            pipe.unet = UNet2DConditionModel.from_pretrained(args.ckpt, subfolder='unet')
            pipe.unet = pipe.unet.to(torch.float16).to(accelerator.device)
        pipe.enable_vae_slicing()
    elif args.SANA:
        from diffusers import SanaPipeline
        pipe = SanaPipeline.from_pretrained(
            args.model_id,
            variant="fp16",
            torch_dtype=torch.float16
        )
        if args.ckpt:
            pipe.load_lora_weights(args.ckpt)
        pipe.vae.to(torch.bfloat16)
        pipe.text_encoder.to(torch.bfloat16)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            variant="fp16"
        )
        if args.ckpt:
            pipe.unet = UNet2DConditionModel.from_pretrained(args.ckpt, subfolder='unet')
            pipe.unet = pipe.unet.to(torch.float16).to(accelerator.device)

    pipe.to(accelerator.device)
    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None

    os.makedirs(args.outdir, exist_ok=True)
    image_dir = os.path.join(args.outdir, "images")
    os.makedirs(image_dir, exist_ok=True)

    generator = torch.Generator(device=accelerator.device).manual_seed(42)
    metadata = {}

    for batch_start in range(0, len(prompts_per_rank), args.batch_size):
        prompt_batch = prompts_per_rank[batch_start:batch_start + args.batch_size]
        outputs = pipe(
            prompt_batch,
            height=args.img_sz,
            width=args.img_sz,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator
        )
        images = outputs.images
        for i, img in enumerate(images):
            global_index = start_idx + batch_start + i
            filename = f"{global_index:05d}.jpg"
            img.save(os.path.join(image_dir, filename))
            metadata[filename] = prompt_batch[i]

    all_metadata = accelerator.gather_for_metrics([metadata])
    if accelerator.is_main_process:
        final_metadata = {}
        for md in all_metadata:
            final_metadata.update(md)
        metadata_path = os.path.join(args.outdir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(final_metadata, f, indent=2, ensure_ascii=False)
        print(f"[Main Rank] Saved metadata to {metadata_path}")

if __name__ == "__main__":
    main()
