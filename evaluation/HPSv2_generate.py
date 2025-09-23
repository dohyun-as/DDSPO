import argparse
import os
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
import torch
from PIL import Image
import hpsv2
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
        "--trailing",
        action="store_true",
        help="scheduler trailing",
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
        "--num_inference_steps",
        type=int,
        default=25,
        help="num_inference_steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="guidance_scale",
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
        default="results",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="dir to write cache",
        default=None,
    )
    parser.add_argument("--img_sz", type=int, default=512)
    args = parser.parse_args()
    return args

def main():

    accelerator = Accelerator()
    world_size = accelerator.num_processes
    rank = accelerator.process_index

    opt = parse_args()
    
    model_id = opt.model_id

    # Use the Euler scheduler here instead
    if opt.SDXL:
        if opt.itercomp:
            scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
            pipe = StableDiffusionXLPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16, use_safetensors=True, cache_dir=opt.cache_dir)
        
        else:
            if opt.trailing:
                pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16", cache_dir=opt.cache_dir)
                
                pipe.scheduler = EulerDiscreteScheduler.from_config(
                    pipe.scheduler.config, timestep_spacing="trailing"
                    )
            else:
                scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
                pipe = StableDiffusionXLPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16, cache_dir=opt.cache_dir)
        
        
        if opt.ckpt is not None:
            pipe.unet = UNet2DConditionModel.from_pretrained(opt.ckpt, subfolder='unet')
            pipe.unet = pipe.unet.to(torch.float16).to(accelerator.device)
            
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
            pipe.unet = pipe.unet.to(torch.float16).to(accelerator.device)
        
    pipe = pipe.to(accelerator.device)
    pipe.set_progress_bar_config(disable=True)
    if opt.SDXL:
        pipe.enable_vae_slicing()
    pipe.safety_checker = None

    save_dir = opt.outdir

    # Create base save directory
    os.makedirs(save_dir, exist_ok=True)

    # Get benchmark prompts (<style> = all, anime, concept-art, paintings, photo)
    all_prompts = hpsv2.benchmark_prompts('all')

    # Iterate over the benchmark prompts to generate images
    generator = torch.Generator(device=accelerator.device).manual_seed(42)
    
    # for style, prompts in all_prompts.items():
    #     style_dir = os.path.join(save_dir, style)
    #     os.makedirs(style_dir, exist_ok=True)  # Ensure the style directory exists

    #     for idx, prompt in enumerate(prompts):
    #         images = pipe(prompt, num_inference_steps=25, guidance_scale=7.5, generator=generator).images
    #         for i in range(len(images)):
    #             images[i].save(os.path.join(style_dir, f"{idx:05d}.jpg"))
    #             print(f"Saved: {os.path.join(style_dir, f'{idx:05d}.jpg')}")

    for style, prompts in all_prompts.items():
        style_dir = os.path.join(save_dir, style)
        if accelerator.is_main_process:
            os.makedirs(style_dir, exist_ok=True)
        accelerator.wait_for_everyone()

        total = len(prompts)
        chunk_size = (total + world_size - 1) // world_size
        start_idx = rank * chunk_size
        end_idx = min(start_idx + chunk_size, total)

        prompts_per_rank = prompts[start_idx:end_idx]

        # 여기부터 Batch로 처리
        for batch_start in range(0, len(prompts_per_rank), opt.batch_size):
            # batch 단위로 프롬프트를 슬라이싱
            prompt_batch = prompts_per_rank[batch_start:batch_start + opt.batch_size]
            # 파이프라인에 리스트 형태로 전달
            outputs = pipe(
                prompt_batch,
                height=opt.img_sz,
                width=opt.img_sz,
                num_inference_steps=opt.num_inference_steps,
                guidance_scale=opt.guidance_scale,
                generator=generator
            )
            images = outputs.images

            # batch 내 각 이미지 저장
            for i, img in enumerate(images):
                global_index = start_idx+batch_start + i
                save_path = os.path.join(style_dir, f"{global_index:05d}.jpg")
                img.save(save_path)
                # print(f"Saved: {save_path}")
                
        
        accelerator.wait_for_everyone()
if __name__ == "__main__":
    main()

# --model_id "stabilityai/stable-diffusion-xl-base-1.0" --SDXL --ckpt "/home/jupyter/T2I_distillation/result/SDXL_RC/checkpoint-100000"
