import argparse
import json
import os
from tqdm import tqdm

import torch
import numpy as np
from PIL import Image
from tqdm import trange
import random
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor

from torch.utils.data import Dataset, DataLoader

# Accelerate
from accelerate import Accelerator

from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel
)


torch.set_grad_enabled(False)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def load_and_split_prompts(metadata_file, accelerator, batch_size):
    with open(metadata_file, "r", encoding="utf-8") as f:
        all_metadata = [json.loads(line) for line in f]

    total = len(all_metadata)
    world_size = accelerator.num_processes
    rank = accelerator.process_index


    chunk_size = (total + world_size - 1) // world_size
    start = rank * chunk_size
    end = min(start + chunk_size, total)

    metadata_split = all_metadata[start:end]

    batches = [metadata_split[i:i + batch_size] for i in range(0, len(metadata_split), batch_size)]
    return batches



# ------------------------------------------------------------------------------
# 2) ARGUMENT PARSING
# ------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "metadata_file",
        type=str,
        help="JSONL file containing lines of metadata for each prompt",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Hugging Face model name (or local path to pipeline)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs",
        help="Directory to write results to",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="Number of samples per prompt",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        nargs="?",
        const=(
            "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, "
            "out of frame, extra limbs, disfigured, deformed, body out of frame, "
            "bad anatomy, watermark, signature, cut off, low contrast, underexposed, "
            "overexposed, bad art, beginner, amateur, distorted face"
        ),
        default=None,
        help="Negative prompt for guidance",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="Guidance scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="How many prompts can be processed simultaneously in one forward pass",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for DataLoader",
    )
    parser.add_argument(
        "--skip_grid",
        action="store_true",
        help="If set, don't save a grid image for each prompt",
    )
    parser.add_argument(
        "--max_eval_batches",
        type=int,
        default=None,
        help="If set, limit the number of batches for quick testing/debugging",
    )
    parser.add_argument(
        "--unet_path",
        type=str,
        default=None,
        help="Optional path to a custom UNet (e.g. from checkpoint).",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional path to a custom cache directory for Hugging Face models.",
    )
    parser.add_argument(
        "--SDXL",
        action="store_true",
        help="SDXL",
    )
    parser.add_argument(
        "--SANA",
        action="store_true",
        help="SANA",
    )
    parser.add_argument(
        "--itercomp",
        action="store_true",
        help="itercomp",
    )
    parser.add_argument("--img_sz", type=int, default=512)
    opt = parser.parse_args()
    return opt


def generate_with_accelerator(accelerator, pipe, opt):
    # Load and split prompts
    with open(opt.metadata_file) as fp:
        all_metadata = [json.loads(line) for line in fp]
        
    total = len(all_metadata)
    world_size = accelerator.num_processes
    rank = accelerator.process_index

    # Split the metadata for each process
    chunk_size = (total + world_size - 1) // world_size
    start_idx = rank * chunk_size
    end_idx = min(start_idx + chunk_size, total)
    
    my_metadata = all_metadata[start_idx:end_idx]
    
    # world_size = accelerator.num_processes
    # rank = accelerator.process_index
    # metadatas = metadatas[rank::world_size]  # split for each process

    if accelerator.is_local_main_process:
        os.makedirs(opt.outdir, exist_ok=True)
        
    accelerator.wait_for_everyone()
    metadata_iter = (
        tqdm(my_metadata, desc="Processing") if accelerator.is_local_main_process else my_metadata
    )

    for local_idx, metadata in enumerate(metadata_iter):
        
        global_idx = start_idx + local_idx
        folder_name = f"{global_idx + 1:05d}"
        
        prompt = metadata["prompt"]
        outpath = os.path.join(opt.outdir, folder_name)
        os.makedirs(outpath, exist_ok=True)
        
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)

        with open(os.path.join(outpath, "metadata.jsonl"), "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        sample_count = 0
        batch_size = opt.batch_size
        all_samples = [] if not opt.skip_grid else None
        with torch.no_grad():
            for _ in range((opt.n_samples + batch_size - 1) // batch_size):
                current_bs = min(batch_size, opt.n_samples - sample_count)
                images = pipe(
                    prompt,
                    height=opt.img_sz,
                    width=opt.img_sz,
                    num_inference_steps=opt.steps,
                    guidance_scale=opt.scale,
                    num_images_per_prompt=current_bs
                ).images

                for img in images:
                    img.save(os.path.join(sample_path, f"{sample_count:05d}.png"))
                    sample_count += 1

                if not opt.skip_grid:
                    all_samples.append(torch.stack([ToTensor()(img) for img in images], dim=0))

            if not opt.skip_grid and all_samples:
                grid = torch.cat(all_samples, dim=0)
                grid = make_grid(grid, nrow=batch_size)
                grid = (255. * grid.permute(1, 2, 0).cpu().numpy()).astype("uint8")
                Image.fromarray(grid).save(os.path.join(outpath, "grid.png"))

    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        print("Done.")
# ------------------------------------------------------------------------------
# 3) MAIN GENERATION
# ------------------------------------------------------------------------------
def main(opt):
    accelerator = Accelerator()
    seed_everything(opt.seed + accelerator.process_index)

    # Load model
    if opt.model.lower().startswith("stabilityai/stable-diffusion-xl") or opt.SDXL:
        if opt.itercomp:
            pipe = StableDiffusionXLPipeline.from_pretrained(opt.model, torch_dtype=torch.float16, use_safetensors=True, cache_dir=opt.cache_dir)
        else:
            pipe = DiffusionPipeline.from_pretrained(
                opt.model, torch_dtype=torch.float16, use_safetensors=True, variant="fp16", cache_dir=opt.cache_dir
            )
        pipe.enable_xformers_memory_efficient_attention()
        
        if opt.unet_path is not None:
            print(f"[Info] Loading UNet from: {opt.unet_path}")
            custom_unet = UNet2DConditionModel.from_pretrained(opt.unet_path, torch_dtype=torch.float16)
            pipe.unet = custom_unet.to(accelerator.device)
        
    elif opt.SANA:
        from diffusers import SanaPipeline
        pipe = SanaPipeline.from_pretrained(
            opt.model,  
            variant="fp16",
            torch_dtype=torch.float16, 
            cache_dir=opt.cache_dir
        )
        
        if opt.unet_path is not None:
            pipe.load_lora_weights(opt.unet_path)
            
        pipe.vae.to(torch.bfloat16)
        pipe.text_encoder.to(torch.bfloat16)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(opt.model, torch_dtype=torch.float16, cache_dir=opt.cache_dir)

        if opt.unet_path is not None:
            print(f"[Info] Loading UNet from: {opt.unet_path}")
            custom_unet = UNet2DConditionModel.from_pretrained(opt.unet_path, torch_dtype=torch.float16)
            pipe.unet = custom_unet.to(accelerator.device)
        
    pipe = pipe.to(accelerator.device)
    pipe.set_progress_bar_config(disable=True)

    pipe.safety_checker = None
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if opt.SDXL:
        pipe.enable_vae_slicing()
    generate_with_accelerator(accelerator, pipe, opt)
        
if __name__ == "__main__":
    opt = parse_args()
    main(opt)
