import argparse
import os
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
import torch
from PIL import Image
from tqdm import tqdm, trange
import sys
# add parent path to sys.path to import lora_diffusion
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from lora_diffusion import patch_pipe
from accelerate import Accelerator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from_file",
        type=str,
        default="../examples/dataset/color_val.txt",
        help="if specified, load prompts from this file",
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
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="eval ckpt",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="pipline id",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=10,
        help="number of iterations to run for each prompt",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1, 
        help="batch size for each prompt",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="you can choose to specify the prompt instead of reading from file",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="../examples" # TODO
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional path to a custom cache directory for Hugging Face models.",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        # default="checkpoint/color/lora_weight_e357_s124500.pt", # TODO
        help="to load the finetuned checkpoint or not",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="Guidance scale",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        help="num_inference_steps",
    )
    parser.add_argument("--img_sz", type=int, default=512)
    args = parser.parse_args()
    return args

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def main():
    opt = parse_args()
    
    accelerator = Accelerator()
    rank = accelerator.process_index 
    world_size = accelerator.num_processes

    model_id = opt.model_id


    # Use the Euler scheduler here instead
    if opt.SDXL:
        from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler", cache_dir=opt.cache_dir)
        if opt.itercomp:
            pipe = StableDiffusionXLPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16, use_safetensors=True, cache_dir=opt.cache_dir)
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16, variant="fp16", cache_dir=opt.cache_dir)


        if opt.ckpt is not None:
            pipe.unet = UNet2DConditionModel.from_pretrained(opt.ckpt, subfolder='unet')
            pipe.unet = pipe.unet.to(torch.float16).to("cuda")
            
    elif opt.SANA:
        from diffusers import SanaPipeline
        pipe = SanaPipeline.from_pretrained(
            model_id,  
            variant="fp16",
            torch_dtype=torch.float16, 
            cache_dir=opt.cache_dir
        )
        
        if opt.ckpt is not None:
            pipe.load_lora_weights(opt.ckpt)
            
        pipe.vae.to(torch.bfloat16)
        pipe.text_encoder.to(torch.bfloat16)

    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, cache_dir=opt.cache_dir)
        
        if opt.ckpt is not None:
            pipe.unet = UNet2DConditionModel.from_pretrained(opt.ckpt, subfolder='unet')
            pipe.unet = pipe.unet.to(torch.float16).to("cuda")
        
    pipe.to(accelerator.device)
    
    if opt.SDXL:
        pipe.enable_vae_slicing()
    pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)

    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        all_prompts = [prompt]
    else:
        if accelerator.is_main_process:
            print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            all_prompts = f.read().splitlines()
            all_prompts = [d.strip().split("\t")[0] for d in all_prompts]

    total_data = len(all_prompts)
    chunk_size = (total_data + world_size - 1) // world_size
    start_index = rank * chunk_size
    end_index = min(start_index + chunk_size, total_data)
    prompts_for_this_rank = all_prompts[start_index:end_index]

    base_count = start_index * opt.n_iter * opt.batch_size
    grid_count = start_index

    # run inference
    outpath = opt.outdir
    sample_path = os.path.join(outpath, f"samples")

    if accelerator.is_main_process:
        os.makedirs(outpath, exist_ok=True)
        os.makedirs(sample_path, exist_ok=True)

    accelerator.wait_for_everyone()

    with torch.no_grad():
        for prompt in tqdm(prompts_for_this_rank, desc=f"Rank {rank} prompts", disable=not accelerator.is_main_process):
            images = []
            generator = torch.Generator(device="cuda").manual_seed(42)
            
            skip_generation = all(
                os.path.exists(os.path.join(sample_path, f"{prompt}_{base_count + n * opt.batch_size + i:06}.png"))
                for n in range(opt.n_iter)
                for i in range(opt.batch_size)
            )
            
            if skip_generation:
                base_count += opt.n_iter * opt.batch_size
                grid_count += 1
                continue

            for n in range(opt.n_iter):
                prompt_batch = [prompt]*opt.batch_size
                image = pipe(prompt_batch, num_inference_steps=opt.num_inference_steps, guidance_scale=opt.scale, generator=generator,
                        height=opt.img_sz,
                        width=opt.img_sz).images
                generator = torch.Generator(device="cuda").manual_seed(42 + n + 1)
                for i in range(len(image)):
                    image[i].save(os.path.join(sample_path, f"{prompt}_{base_count:06}.png"))
                    images.append(image[i])
                    base_count += 1
            grid = image_grid(images, rows=opt.n_iter, cols=opt.batch_size)
            grid.save(os.path.join(outpath, f'{prompt}-grid-{grid_count:05}.png'))
            grid_count += 1
        accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()
