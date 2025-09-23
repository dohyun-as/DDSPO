import argparse
import os
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
import torch
from PIL import Image
from tqdm import tqdm, trange
import sys
import csv
import time
import math
from pathlib import Path
# add parent path to sys.path to import lora_diffusion
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from lora_diffusion import patch_pipe
from accelerate import Accelerator

def sample_images_30k(args, accelerator, pipeline):
    
    save_dir_src = os.path.join(args.outdir, f'im{args.img_sz}')  # for model's raw output images
    save_dir_tgt = os.path.join(args.outdir, f'im{args.img_resz}')  # for resized images for benchmark

    # Create directories only on the main process
    if accelerator.is_main_process:
        os.makedirs(save_dir_src, exist_ok=True)
        os.makedirs(save_dir_tgt, exist_ok=True)

    accelerator.wait_for_everyone()

    file_list = get_file_list_from_csv(args.MSCOCO_csv_file)
    
    file_list = [
    (img_name, prompt)
    for img_name, prompt in file_list
    if not Path(os.path.join(args.outdir, f'im{args.img_sz}', img_name)).exists()
    ]

    total_files = len(file_list)
    num_processes = accelerator.num_processes
    rank = accelerator.process_index

    # Distribute files evenly among ranks without leaving any unprocessed files
    files_per_process = total_files // num_processes
    remainder = total_files % num_processes

    if rank < remainder:
        start_index = rank * (files_per_process + 1)
        end_index = start_index + files_per_process + 1
    else:
        start_index = remainder * (files_per_process + 1) + (rank - remainder) * files_per_process
        end_index = start_index + files_per_process

    # Get the list of files to process for this rank
    process_file_list = file_list[start_index:end_index]
    
    total_batches = math.ceil(len(process_file_list) / args.batch_size)
    
    # tqdm progress bar setup for rank 0 only
    if accelerator.is_main_process:
        progress_bar = tqdm(total=total_batches, desc="Generating Images",ncols=None, disable=not accelerator.is_main_process)

    # Process the assigned files in batches
    for batch_start in range(0, len(process_file_list), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(process_file_list))

        img_names = [file_info[0] for file_info in process_file_list[batch_start:batch_end]]
        val_prompts = [file_info[1] for file_info in process_file_list[batch_start:batch_end]]
        generator = torch.Generator(device=accelerator.device).manual_seed(42)
        # Suppress output of pipeline.generate
        imgs = pipeline(val_prompts,
                        guidance_scale=args.scale,
                        num_inference_steps=args.num_inference_steps,
                        progress_bar=False, 
                        generator=generator,
                        height=args.img_sz,
                        width=args.img_sz,
                        ).images
        for img, img_name in zip(imgs, img_names):
            img.save(os.path.join(save_dir_src, img_name))
            img.close()

        # Update progress bar only on rank 0
        if accelerator.is_main_process:
            progress_bar.update(1)
        accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        change_img_size(save_dir_src, save_dir_tgt, args.img_resz)
        progress_bar.close()
        accelerator.print(f"Image generation completed and resized images saved.")
    else: 
        time.sleep(300)
        
    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()
    
def get_file_list_from_csv(csv_file_path):
    file_list = []
    with open(csv_file_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)        
        next(csv_reader, None) # Skip the header row
        for row in csv_reader: # (row[0], row[1]) = (img name, txt prompt) 
            file_list.append(row)
    return file_list

def change_img_size(input_folder, output_folder, resz=256):
    img_list = sorted([file for file in os.listdir(input_folder) if file.endswith('.jpg')])
    for i, filename in enumerate(img_list):
        img = Image.open(os.path.join(input_folder, filename))
        img.resize((resz, resz)).save(os.path.join(output_folder, filename))
        img.close()
        if i % 2000 == 0:
            print(f"{i}/{len(img_list)} | {filename}: resize to {resz}")

def change_img_size_ddp(input_folder, output_folder, resz, accelerator):
    img_list = sorted([file for file in os.listdir(input_folder) if file.endswith('.jpg')])
    
    # Distribute image list among ranks
    total_images = len(img_list)
    num_processes = accelerator.num_processes
    rank = accelerator.process_index

    images_per_process = total_images // num_processes
    remainder = total_images % num_processes

    if rank < remainder:
        start_index = rank * (images_per_process + 1)
        end_index = start_index + images_per_process + 1
    else:
        start_index = remainder * (images_per_process + 1) + (rank - remainder) * images_per_process
        end_index = start_index + images_per_process

    process_img_list = img_list[start_index:end_index]

    local_change_count = 0

    # Resize images assigned to this process
    for i, filename in enumerate(process_img_list):
        img = Image.open(os.path.join(input_folder, filename))
        img.resize((resz, resz)).save(os.path.join(output_folder, filename))
        img.close()
        local_change_count += 1
        
        if i % 1000 == 0:
            accelerator.print(f"Rank {rank}: {i}/{len(process_img_list)} | {filename}: resized to {resz}")

    accelerator.wait_for_everyone()
    # Convert local_change_count to a tensor
    local_change_count_tensor = torch.tensor([local_change_count], device=accelerator.device)

    # Gather the total number of resized images across all ranks
    total_change_count_tensor = accelerator.gather(local_change_count_tensor)

    # Sum the gathered tensors and convert it to a Python integer
    total_change_count = total_change_count_tensor.sum().item()
    if accelerator.is_main_process:
        accelerator.print(f"Total images resized: {total_change_count}")
    accelerator.wait_for_everyone()
    return total_change_count
  
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
        "--MSCOCO_csv_file",
        type=str,
        default="./evaluation/mscoco_val2014/metadata.csv",
        help="MSCOCO validation dataset csv file",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        # default="checkpoint/color/lora_weight_e357_s124500.pt", # TODO
        help="to load the finetuned checkpoint or not",
    )
    parser.add_argument("--img_sz", type=int, default=512)
    parser.add_argument("--img_resz", type=int, default=256)
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="Guidance scale",
    )
    parser.add_argument("--num_inference_steps", type=int, default=25)
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
    
    if opt.SDXL or opt.SANA:
        pipe.enable_vae_slicing()
        
    pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)

    sample_images_30k(opt, accelerator, pipe)

if __name__ == "__main__":
    main()
