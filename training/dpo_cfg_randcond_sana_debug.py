#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import io
import logging
import math
import os
import random
import shutil
import sys
import copy
from pathlib import Path

import accelerate
import datasets
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Subset
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer, Gemma2Model
from transformers.utils import ContextManagers

import diffusers
from diffusers import (
    AutoencoderDC,
    FlowMatchEulerDiscreteScheduler,
    SanaPipeline,
    SanaTransformer2DModel,
)
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel,StableDiffusionXLPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)

from dataset_sana import self_training_dataset, collate_fn

if is_wandb_available():
    import wandb

    
    
## SDXL
import functools
import gc
from torchvision.transforms.functional import crop
from transformers import AutoTokenizer, PretrainedConfig



# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.20.0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "yuvalkirstain/pickapic_v1": ("jpg_0", "jpg_1", "label_0", "caption"),
    "yuvalkirstain/pickapic_v2": ("jpg_0", "jpg_1", "label_0", "caption"),
}

        
def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="caption",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None,
                        # was random for submission, need to test that not distributing same noise etc across devices
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help=(
            "The resolution for input images, all the images in the dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--random_crop",
        default=False,
        action="store_true",
        help=(
            "If set the images will be randomly"
            " cropped (instead of center). The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--no_hflip",
        action="store_true",
        help="whether to supress horizontal flipping",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=2000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-8,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_adafactor", action="store_true", help="Whether or not to use adafactor (should save mem)"
    )
    # Bram Note: Haven't looked @ this yet
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default='latest',
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="tuning",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    ## SDXL
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument("--sdxl", action='store_true', help="Train sdxl")
    parser.add_argument("--sana", action='store_true', help="Train sana")
    
    parser.add_argument(
        "--complex_human_instruction", type=str, default=(
        "- 'Given a user prompt, generate an \"Enhanced prompt\" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:'\n"
        "- '- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.'\n"
        "- '- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.'\n"
        "- 'Here are examples of how to transform or refine prompts:'\n"
        "- '- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.'\n"
        "- '- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.'\n"
        "- 'Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:'\n"
        "- 'User Prompt: '"
    )
    )
    
    
    ## DPO
    parser.add_argument("--sft", action='store_true', help="Run Supervised Fine-Tuning instead of Direct Preference Optimization")
    parser.add_argument("--beta_dpo", type=float, default=5000, help="The beta DPO temperature controlling strength of KL penalty")
    parser.add_argument(
        "--hard_skip_resume", action="store_true", help="Load weights etc. but don't iter through loader for loader resume, useful b/c resume takes forever"
    )
    parser.add_argument(
        "--unet_init", type=str, default='', help="Initialize start of run from unet (not compatible w/ checkpoint load)"
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0.2,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--split", type=str, default='train', help="Datasplit"
    )
    parser.add_argument(
        "--choice_model", type=str, default='', help="Model to use for ranking (override dataset PS label_0/1). choices: aes, clip, hps, pickscore"
    )
    parser.add_argument(
        "--only_cfg", action="store_true", help="Only Self Training Diffusion DPO CFG"
        "Use the reference model’s CFG scores to guide the training of the target model, replacing explicit score."
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1,
        help="The guidance scale to use for the reference model. Defaults to 7.5.",
    )
    parser.add_argument(
        "--rand_cond", action="store_true", help="Use random conditioning"
    )
    parser.add_argument(
        "--rand_cond_lambda",
        type=float,
        default=20,
        help="The lambda value for the random conditioning. Defaults to 20 in sigmoid setting, 5 in exponential setting.",
    )
    parser.add_argument(
        "--rand_cond_pt",
        type=str,
        default="sigmoid",
        help="Paths to one or more files containing extra paired text prompts for training.",
    )
    parser.add_argument(
        "--extra_text_path",
        type=str,
        nargs='+',
        default=None,
        help="Paths to one or more files containing extra paired text prompts for training.",
    )
    parser.add_argument(
        "--timestep_sampling",
        type=str,
        default=None,
        help="The sampling method for timesteps. Choose between ['sigmoid', 'linear']. Defaults to uniform.",
    )
    parser.add_argument(
        "--pos_neg_score", action="store_true", help="using negative prompts for guidance score instead of null prompts"
    )
    parser.add_argument(
        "--cfg_scaling", action="store_true", help="cfg vaiance scaling"
    )
    parser.add_argument(
        "--random_neg_prompts", action="store_true", help="random negative prompt"
    )
    parser.add_argument(
        "--replace_neg_img_with_other_pos", action="store_true", help="replace negative images with other positive images"
    )
    parser.add_argument(
        "--replace_neg_prompt_with_other_pos", action="store_true", help="replace negative prompts with other positive prompts"
    )
    parser.add_argument(
        "--replace_neg_img_with_same_pos", action="store_true", help="replace negative images with other positive images"
    )
    parser.add_argument(
        "--pos_neg_different_noise", action="store_true", help="use different noise for positive and negative images"
    )
    
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    ## SDXL
    if args.sdxl:
        print("Running SDXL")
    if args.resolution is None:
        if args.sdxl:
            args.resolution = 1024
        else:
            args.resolution = 512
            
    args.train_method = 'sft' if args.sft else 'dpo'
    return args

def sanitize_tracker_config(config_dict):
    safe_config = {}
    for k, v in config_dict.items():
        if isinstance(v, (int, float, str, bool, torch.Tensor)):
            safe_config[k] = v
        elif isinstance(v, list):
            if all(isinstance(item, str) for item in v):
                safe_config[k] = ','.join(v)
    return safe_config
def print_unet_dtype_info(unet, ref_unet, accelerator):
    print("== UNet / Transformer2DModel Dtype Check ==")

    unet = accelerator.unwrap_model(unet)
    ref_unet = accelerator.unwrap_model(ref_unet)

    print(f"[unet]      global dtype: {next(unet.parameters()).dtype}")
    print(f"[ref_unet]  global dtype: {next(ref_unet.parameters()).dtype}")

    def print_attention_dtype(model, label):
        try:
            block = model.transformer_blocks[0]
            attn2 = block.attn2

            def safe_dtype(module_or_list):
                if isinstance(module_or_list, torch.nn.ModuleList):
                    for i, m in enumerate(module_or_list):
                        try:
                            print(f"    └─ submodule[{i}] weight dtype: {m.weight.dtype}")
                        except AttributeError:
                            print(f"    └─ submodule[{i}] has no 'weight'")
                else:
                    print(f"    weight dtype: {module_or_list.weight.dtype}")

            print(f"[{label}] attention.q_proj:")
            safe_dtype(attn2.to_q)
            print(f"[{label}] attention.k_proj:")
            safe_dtype(attn2.to_k)
            print(f"[{label}] attention.v_proj:")
            safe_dtype(attn2.to_v)
            print(f"[{label}] attention.out_proj:")
            safe_dtype(attn2.to_out)

        except Exception as e:
            print(f"[{label}] attention dtype check failed: {e}")

    print_attention_dtype(unet, "unet")
    print_attention_dtype(ref_unet, "ref_unet")

# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt_sdxl(batch, text_encoders, text_inputs_list, proportion_empty_prompts, caption_column, is_train=True):
    prompt_embeds_list = []

    with torch.no_grad():
        for text_input_ids, text_encoder in zip(text_inputs_list, text_encoders):
            prompt_embeds = text_encoder(
                text_input_ids.to('cuda'),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds}




def main():
    
    args = parse_args()
    
    #### START ACCELERATOR BOILERPLATE ###
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index) # added in + term, untested

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    ### END ACCELERATOR BOILERPLATE
    
    
    ### START DIFFUSION BOILERPLATE ###
    # Load scheduler, tokenizer and models.
    if args.sana:
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler", revision=args.revision
        )
        noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    else:
        noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, 
                                                    subfolder="scheduler", cache_dir=args.cache_dir)
    def enforce_zero_terminal_snr(scheduler):
        # Modified from https://arxiv.org/pdf/2305.08891.pdf
        # Turbo needs zero terminal SNR to truly learn from noise
        # Turbo: https://static1.squarespace.com/static/6213c340453c3f502425776e/t/65663480a92fba51d0e1023f/1701197769659/adversarial_diffusion_distillation.pdf
        # Convert betas to alphas_bar_sqrt
        alphas = 1 - scheduler.betas
        alphas_bar = alphas.cumprod(0)
        alphas_bar_sqrt = alphas_bar.sqrt()

        # Store old values.
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
        # Shift so last timestep is zero.
        alphas_bar_sqrt -= alphas_bar_sqrt_T
        # Scale so first timestep is back to old value.
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

        alphas_bar = alphas_bar_sqrt ** 2
        alphas = alphas_bar[1:] / alphas_bar[:-1]
        alphas = torch.cat([alphas_bar[0:1], alphas])
    
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        scheduler.alphas_cumprod = alphas_cumprod
        return 
    if 'turbo' in args.pretrained_model_name_or_path:
        enforce_zero_terminal_snr(noise_scheduler)
    
    # SDXL has two text encoders
    if args.sdxl:
        # Load the tokenizers
        if args.pretrained_model_name_or_path=="stabilityai/stable-diffusion-xl-refiner-1.0":
            tokenizer_and_encoder_name = "stabilityai/stable-diffusion-xl-base-1.0"
        else:
            tokenizer_and_encoder_name = args.pretrained_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_and_encoder_name, subfolder="tokenizer", revision=args.revision, use_fast=False, cache_dir=args.cache_dir
        )
        tokenizer_2 = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision, use_fast=False, cache_dir=args.cache_dir
        )
        
    elif args.sana:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, cache_dir=args.cache_dir
        )
        tokenizer_2 = None
        
    else:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, cache_dir=args.cache_dir
        )
        tokenizer_2 = None

    # Not sure if we're hitting this at all
    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    
    # BRAM NOTE: We're not using deepspeed currently so not sure it'll work. Could be good to add though!
    # 
    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        # SDXL has two text encoders
        if args.sdxl:
            # import correct text encoder classes
            text_encoder_cls_one = import_model_class_from_model_name_or_path(
               tokenizer_and_encoder_name, args.revision
            )
            text_encoder_cls_two = import_model_class_from_model_name_or_path(
                tokenizer_and_encoder_name, args.revision, subfolder="text_encoder_2"
            )
            text_encoder_one = text_encoder_cls_one.from_pretrained(
                tokenizer_and_encoder_name, subfolder="text_encoder", revision=args.revision, cache_dir=args.cache_dir
            )
            text_encoder_two = text_encoder_cls_two.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, cache_dir=args.cache_dir
            )
            if args.pretrained_model_name_or_path=="stabilityai/stable-diffusion-xl-refiner-1.0":
                text_encoders = [text_encoder_two]
                tokenizers = [tokenizer_2]
            else:
                text_encoders = [text_encoder_one, text_encoder_two]
                tokenizers = [tokenizer, tokenizer_2]
        elif args.sana:
            text_encoder = Gemma2Model.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, cache_dir=args.cache_dir
            )
            
            text_encoder.to(dtype=torch.bfloat16)
        else:
            text_encoder = CLIPTextModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, cache_dir=args.cache_dir
            )
        # Can custom-select VAE (used in original SDXL tuning)
        vae_path = (
            args.pretrained_model_name_or_path
            if args.pretrained_vae_model_name_or_path is None
            else args.pretrained_vae_model_name_or_path
        )
        if args.sana:
            vae = AutoencoderDC.from_pretrained(
                vae_path, subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None, revision=args.revision, cache_dir=args.cache_dir
            )
        else:
            vae = AutoencoderKL.from_pretrained(
                vae_path, subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None, revision=args.revision, cache_dir=args.cache_dir
            )
        # clone of model
        
        if args.sana:
            ref_unet = SanaTransformer2DModel.from_pretrained(
                args.unet_init if args.unet_init else args.pretrained_model_name_or_path,
                subfolder="transformer", revision=args.revision, cache_dir=args.cache_dir
            )
            text_encoding_pipeline = SanaPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                vae=None,
                transformer=None,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
            )
            
        else:
            ref_unet = UNet2DConditionModel.from_pretrained(
                args.unet_init if args.unet_init else args.pretrained_model_name_or_path,
                subfolder="unet", revision=args.revision, cache_dir=args.cache_dir
            )
    if args.unet_init:
        print("Initializing unet from", args.unet_init)
    if args.sana:
        unet = SanaTransformer2DModel.from_pretrained(
                args.unet_init if args.unet_init else args.pretrained_model_name_or_path,
                subfolder="transformer", revision=args.revision, cache_dir=args.cache_dir
            )
    else:
        unet = UNet2DConditionModel.from_pretrained(
            args.unet_init if args.unet_init else args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, cache_dir=args.cache_dir
        )
    
    

    def compute_text_embeddings(prompt, text_encoding_pipeline, weight_dtype=None):
        text_encoding_pipeline = text_encoding_pipeline.to(accelerator.device)
        with torch.no_grad():
            prompt_embeds, prompt_attention_mask, _, _ = text_encoding_pipeline.encode_prompt(
                prompt,
                max_sequence_length=300,
                complex_human_instruction=args.complex_human_instruction,
            )
        # print("complex_human_instruction", args.complex_human_instruction)
        # if args.offload:
        #     text_encoding_pipeline = text_encoding_pipeline.to("cpu")
        prompt_embeds = prompt_embeds.to(weight_dtype)
        return prompt_embeds, prompt_attention_mask

    # Freeze vae, text_encoder(s), reference unet
    vae.requires_grad_(False)
    if args.sdxl:
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
    else:
        text_encoder.requires_grad_(False)
    if args.train_method == 'dpo': ref_unet.requires_grad_(False)

    # xformers efficient attention
    # if is_xformers_available():
    #     import xformers

    #     xformers_version = version.parse(xformers.__version__)
    #     if xformers_version == version.parse("0.0.16"):
    #         logger.warn(
    #             "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
    #         )
    #     unet.enable_xformers_memory_efficient_attention()
    # else:
    #     raise ValueError("xformers is not available. Make sure it is installed correctly")

    # BRAM NOTE: We're using >=0.16.0. Below was a bit of a bug hive. I hacked around it, but ideally ref_unet wouldn't
    # be getting passed here
    # 
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            
            if len(models) > 1:
                assert args.train_method == 'dpo' # 2nd model is just ref_unet in DPO case
            models_to_save = models[:1]
            for i, model in enumerate(models_to_save):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):

            if len(models) > 1:
                assert args.train_method == 'dpo' # 2nd model is just ref_unet in DPO case
            models_to_load = models[:1]
            for i in range(len(models_to_load)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing or args.sdxl: #  (args.sdxl and ('turbo' not in args.pretrained_model_name_or_path) ):
        print("Enabling gradient checkpointing, either because you asked for this or because you're using SDXL")
        unet.enable_gradient_checkpointing()

    # Bram Note: haven't touched
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.use_adafactor or args.sdxl or args.sana:
        print("Using Adafactor either because you asked for it or you're using SDXL")
        optimizer = transformers.Adafactor(unet.parameters(),
                                           lr=args.learning_rate,
                                           weight_decay=args.adam_weight_decay,
                                           clip_threshold=1.0,
                                           scale_parameter=False,
                                          relative_step=False)
    else:
        optimizer = torch.optim.AdamW(
            unet.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        
        
        
    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    train_dataset = self_training_dataset(data_dir=args.train_data_dir, rand_cond=args.rand_cond, rand_cond_lambda=args.rand_cond_lambda,
                                          rand_cond_pt=args.rand_cond_pt,n_T=noise_scheduler.num_train_timesteps,extra_text_path=args.extra_text_path, 
                                          timestep_sampling=args.timestep_sampling, random_neg_prompts=args.random_neg_prompts,
                                          replace_neg_img_with_other_pos=args.replace_neg_img_with_other_pos, replace_neg_prompt_with_other_pos=args.replace_neg_prompt_with_other_pos,
                                          replace_neg_img_with_same_pos=args.replace_neg_img_with_same_pos)
    
    
    if args.max_train_samples is not None:
        original_seed = random.getstate()
        random.seed(42)  # 시드 고정 → 재현 가능성 확보
        indices = random.sample(range(len(train_dataset)), args.max_train_samples)
        random.setstate(original_seed)  # 다른 random에 영향 안 주도록 복구

        train_dataset = Subset(train_dataset, indices)

    ### DATASET #####

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn(tokenizer, tokenizer_2),
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True
    )
    ##### END BIG OLD DATASET BLOCK #####
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    
    #### START ACCELERATOR PREP ####
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision
        
    print(accelerator.state.mixed_precision)  # 'fp16'이면 OK
    print(os.environ.get("ACCELERATE_FP32_ATTENTION"))  # 'true'여야 정상

        
    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    if args.sdxl:
        text_encoder_one.to(accelerator.device, dtype=weight_dtype)
        text_encoder_two.to(accelerator.device, dtype=weight_dtype)
        print("offload vae (this actually stays as CPU)")
        vae = accelerate.cpu_offload(vae)
        # print("Offloading text encoders to cpu")
        # text_encoder_one = accelerate.cpu_offload(text_encoder_one)
        # text_encoder_two = accelerate.cpu_offload(text_encoder_two)
        if args.train_method == 'dpo':
            ref_unet.to(accelerator.device, dtype=weight_dtype)
            # print("offload ref_unet")
            # ref_unet = accelerate.cpu_offload(ref_unet)
            
    elif args.sana:
        text_encoder.to(accelerator.device, dtype=torch.bfloat16)
        ref_unet.to(accelerator.device, dtype=weight_dtype)
      
    else:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        if args.train_method == 'dpo':
            ref_unet.to(accelerator.device, dtype=weight_dtype)
    ### END ACCELERATOR PREP ###
    
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = sanitize_tracker_config(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Training initialization
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0


    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
        

    # Bram Note: This was pretty janky to wrangle to look proper but works to my liking now
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    if args.guidance_scale != 1 and args.sdxl is None:
        null_input_ids = tokenizer(
            [""],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt"
        ).input_ids.to(accelerator.device)
        null_encoder_hidden_states = text_encoder(null_input_ids)[0]
        
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    #### START MAIN TRAINING LOOP #####
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        implicit_acc_accumulated = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step and (not args.hard_skip_resume):
                if step % args.gradient_accumulation_steps == 0:
                    print(f"Dummy processing step {step}, will start training at {resume_step}")
                continue
            with accelerator.accumulate(unet):
                # Convert images to latent space
                if args.train_method == 'dpo':
                    # y_w and y_l were concatenated along channel dimension
                    latents = torch.cat(batch["latents"].chunk(2, dim=1)).to(weight_dtype)
                    # If using AIF then we haven't ranked yet so do so now
                    # Only implemented for BS=1 (assert-protected)
                elif args.train_method == 'sft':
                    feed_pixel_values = batch["pixel_values"]
                
                #### Diffusion Stuff ####
                # encode pixels --> latents
                # with torch.no_grad():
                #     latents = vae.encode(feed_pixel_values.to(weight_dtype)).latent_dist.sample()
                #     latents = latents * vae.config.scaling_factor


                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                # variants of noising
                if args.noise_offset: # haven't tried yet
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                if args.input_perturbation: # haven't tried yet
                    new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                    
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = batch["timesteps"]
                timesteps = timesteps.long().to(device=latents.device)
                # only first 20% timesteps for SDXL refiner
                if 'refiner' in args.pretrained_model_name_or_path:
                    timesteps = timesteps % 200
                elif 'turbo' in args.pretrained_model_name_or_path:
                    timesteps_0_to_3 = timesteps % 4
                    timesteps = 250 * timesteps_0_to_3 + 249
                
                if args.train_method == 'dpo': # make timesteps and noise same for pairs in DPO
                    timesteps = timesteps.repeat(2)
                    noise = noise if args.pos_neg_different_noise else noise.chunk(2)[0].repeat(2, 1, 1, 1) 

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.sana:
                        
                    u_half = compute_density_for_timestep_sampling(
                        weighting_scheme="none",
                        batch_size=bsz//2,
                        logit_mean=0.0,
                        logit_std=1.0,
                        mode_scale=1.29,
                    )
                    u = torch.cat([u_half, u_half], dim=0) 
                    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                    timesteps = noise_scheduler_copy.timesteps[indices].to(device=latents.device)
                    sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                    noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
                else:
                    noisy_latents = noise_scheduler.add_noise(latents,
                                                          new_noise if args.input_perturbation else noise,
                                                          timesteps)
                ### START PREP BATCH ###
                if args.sdxl:
                    # Get the text embedding for conditioning
                    with torch.no_grad():
                        # Need to compute "time_ids" https://github.com/huggingface/diffusers/blob/v0.20.0-release/examples/text_to_image/train_text_to_image_sdxl.py#L969
                        # for SDXL-base these are torch.tensor([args.resolution, args.resolution, *crop_coords_top_left, *target_size))
                        if 'refiner' in args.pretrained_model_name_or_path:
                            add_time_ids = torch.tensor([args.resolution, 
                                                         args.resolution,
                                                         0,
                                                         0,
                                                          6.0], # aesthetics conditioning https://github.com/huggingface/diffusers/blob/v0.20.0/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_img2img.py#L691C9-L691C24
                                                         dtype=weight_dtype,
                                                         device=accelerator.device)[None, :].repeat(timesteps.size(0), 1)
                        else: # SDXL-base
                            add_time_ids = torch.tensor([args.resolution, 
                                                         args.resolution,
                                                         0,
                                                         0,
                                                          args.resolution, 
                                                         args.resolution],
                                                         dtype=weight_dtype,
                                                         device=accelerator.device)[None, :].repeat(timesteps.size(0), 1)
                        prompt_batch = encode_prompt_sdxl(batch, 
                                                          text_encoders,
                                                          [batch["pos_input_ids"], batch["pos_input_ids_2"]],
                                                           args.proportion_empty_prompts, 
                                                          caption_column='caption',
                                                           is_train=True,
                                                          )
                        neg_prompt_batch = encode_prompt_sdxl(batch,
                                                              text_encoders,
                                                          [batch["neg_input_ids"], batch["neg_input_ids_2"]],
                                                           args.proportion_empty_prompts, 
                                                          caption_column='caption',
                                                           is_train=True,
                                                          )
                    if args.train_method == 'dpo':
                        prompt_batch["prompt_embeds"] = prompt_batch["prompt_embeds"].repeat(2, 1, 1)
                        prompt_batch["pooled_prompt_embeds"] = prompt_batch["pooled_prompt_embeds"].repeat(2, 1)
                    unet_added_conditions = {"time_ids": add_time_ids,
                                            "text_embeds": prompt_batch["pooled_prompt_embeds"]}
                    
                elif args.sana:
                    pos_prompts = batch["pos_prompts"]
                    neg_prompts = batch["neg_prompts"]
                    pos_prompt_embeds, pos_prompt_attention_mask = compute_text_embeddings(pos_prompts, text_encoding_pipeline, weight_dtype)
                    neg_prompt_embeds, neg_prompt_attention_mask = compute_text_embeddings(neg_prompts, text_encoding_pipeline, weight_dtype)
                    
                else: # sd1.5
                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(batch["pos_input_ids"])[0]
                    if args.train_method == 'dpo':
                        pos_encoder_hidden_states = encoder_hidden_states
                        encoder_hidden_states = encoder_hidden_states.repeat(2, 1, 1)
                #### END PREP BATCH ####
                        
                # assert noise_scheduler.config.prediction_type == "epsilon"
                if args.sana:
                    target = noise - latents
                else:
                    target = noise
                print_unet_dtype_info(unet, ref_unet, accelerator)
                if args.only_cfg:
                    batch["paireds"] = torch.zeros_like(batch["paireds"]) 
                stdpo_mask = (batch["paireds"] == 0) # shape [bsz]
                stdpo_indices = stdpo_mask.nonzero(as_tuple=True)[0]
                
                print(len(stdpo_indices), "stdpo_indices")
                if len(stdpo_indices) > 0:
                    pos_indices = stdpo_indices

                    neg_indices = stdpo_indices + batch["paireds"].shape[0]

                    final_indices = torch.cat([pos_indices, neg_indices], dim=0)

                    stdpo_latents = noisy_latents[final_indices]
                    stdpo_timesteps = timesteps[final_indices]

                    if args.sana:
                        ref_direction_prompt_embeds = torch.cat([pos_prompt_embeds[stdpo_indices], neg_prompt_embeds[stdpo_indices]], dim=0)
                        ref_direction_prompt_attention_mask = torch.cat([pos_prompt_attention_mask[stdpo_indices], neg_prompt_attention_mask[stdpo_indices]], dim=0)

                    else:
                        stdpo_pos = pos_encoder_hidden_states[stdpo_indices]
                        stdpo_neg = text_encoder(batch["neg_input_ids"][stdpo_indices])[0]
                        stdpo_direction_embeds = torch.cat([stdpo_pos, stdpo_neg], dim=0)
                        ref_direction_batch_args = (stdpo_latents, stdpo_timesteps, stdpo_direction_embeds)

                    def print_transformer_block_dtypes(model, name="unet"):
                        model = accelerator.unwrap_model(model)
                        for i, block in enumerate(model.transformer_blocks):
                            attn = block.attn2
                            print(f"[{name}] block {i} attn2.q_proj.dtype: {attn.to_q.weight.dtype}")
                            print(f"[{name}] block {i} attn2.k_proj.dtype: {attn.to_k.weight.dtype}")
                            print(f"[{name}] block {i} attn2.v_proj.dtype: {attn.to_v.weight.dtype}")
                    print_transformer_block_dtypes(unet, name="unet")
                    print_transformer_block_dtypes(ref_unet, name="ref_unet")
                    
                    for name, param in accelerator.unwrap_model(unet).named_parameters():
                        if torch.isnan(param).any():
                            print(f"NaN in weight: {name}")
                        
                    for name, param in accelerator.unwrap_model(unet).named_parameters():
                        max_val = param.abs().max()
                        if max_val > 1e4:  # fp16에서 흔한 위험 수치
                            print(f"[Warn] Large weight in {name}: {max_val}")
                    def compare_models(unet, ref_unet, rtol=1e-5, atol=1e-8):
                        unet = accelerator.unwrap_model(unet)
                        ref_unet = accelerator.unwrap_model(ref_unet)

                        unet_state = dict(unet.named_parameters())
                        ref_state = dict(ref_unet.named_parameters())

                        all_keys = sorted(set(unet_state.keys()) & set(ref_state.keys()))
                        print(f"🔍 Comparing {len(all_keys)} matched parameters...")

                        max_global_diff = 0.0
                        max_diff_name = None
                        max_diff_tensor_1 = None
                        max_diff_tensor_2 = None
                        max_diff_mean = None

                        for name in all_keys:
                            p1 = unet_state[name].detach().cpu().float()
                            p2 = ref_state[name].detach().cpu().float()

                            if p1.shape != p2.shape:
                                print(f"[❌] Shape mismatch at {name}: {p1.shape} vs {p2.shape}")
                                continue

                            if torch.isnan(p1).any():
                                print(f"[⚠️] NaN in unet[{name}]")
                            if torch.isnan(p2).any():
                                print(f"[⚠️] NaN in ref_unet[{name}]")

                            diff_tensor = (p1 - p2).abs()
                            max_diff = diff_tensor.max().item()
                            mean_diff = diff_tensor.mean().item()

                            if max_diff > max_global_diff:
                                max_global_diff = max_diff
                                max_diff_mean = mean_diff
                                max_diff_name = name
                                max_diff_tensor_1 = p1
                                max_diff_tensor_2 = p2

                            if not torch.allclose(p1, p2, rtol=rtol, atol=atol):
                                print(f"[🚨] Mismatch at {name} | max_diff: {max_diff:.4e}, mean_diff: {mean_diff:.4e}")
                            else:
                                print(f"[OK] {name} | max_diff: {max_diff:.4e}, mean_diff: {mean_diff:.4e}")

                        print("✅ Done comparing.")

                        if max_diff_name:
                            print(f"\n📌 Largest difference found at: {max_diff_name}")
                            print(f"    max_diff : {max_global_diff:.6e}")
                            print(f"    mean_diff: {max_diff_mean:.6e}")
                            print(f"    shape    : {max_diff_tensor_1.shape}")
                            print(f"    unet val : {max_diff_tensor_1.view(-1)[:5].tolist()}")
                            print(f"    ref val  : {max_diff_tensor_2.view(-1)[:5].tolist()}")
                        
                    compare_models(unet, ref_unet)
                    
                    def register_nan_hooks_for_unet(model, name="unet"):
                        model = accelerator.unwrap_model(model)
                        hook_handles = []

                        def make_nan_hook(module_name, block_idx):
                            def hook_fn(module, inputs, outputs):
                                if isinstance(outputs, tuple):
                                    outputs = outputs[0]
                                has_nan = torch.isnan(outputs).any().item()
                                if has_nan:
                                    print(f"[🚨 NaN] {name} Block {block_idx} - {module_name}")
                                    print(f"→ dtype: {outputs.dtype}, max: {outputs.max().item():.4f}, min: {outputs.min().item():.4f}")
                                else:
                                    print(f"[OK] {name} Block {block_idx} - {module_name}")
                            return hook_fn

                        for i, block in enumerate(model.transformer_blocks):
                            if hasattr(block, "attn1"):
                                h1 = block.attn1.register_forward_hook(make_nan_hook("attn1", i))
                                hook_handles.append(h1)
                            if hasattr(block, "attn2"):
                                h2 = block.attn2.register_forward_hook(make_nan_hook("attn2", i))
                                hook_handles.append(h2)
                            if hasattr(block, "ff"):
                                h3 = block.ff.register_forward_hook(make_nan_hook("ff", i))
                                hook_handles.append(h3)

                        return hook_handles

                    def register_attn2_hooks(model, name="unet"):
                        hook_handles = []

                        def attn2_nan_checker_hook(module, inputs, outputs):
                            hidden_states = inputs[0]
                            encoder_hidden_states = inputs[1] if len(inputs) > 1 else None

                            print(f"[HOOK] {name}.{module.__class__.__name__}")
                            print(f"→ hidden_states dtype: {hidden_states.dtype}, NaN: {torch.isnan(hidden_states).any().item()}, max: {hidden_states.max().item():.4f}, min: {hidden_states.min().item():.4f}")

                            if encoder_hidden_states is not None:
                                print(f"→ encoder_hidden_states dtype: {encoder_hidden_states.dtype}, NaN: {torch.isnan(encoder_hidden_states).any().item()}")

                        for i, block in enumerate(model.transformer_blocks):
                            handle = block.attn2.register_forward_hook(attn2_nan_checker_hook)
                            hook_handles.append(handle)

                        return hook_handles
                    
                    def attn2_input_hook(module, input, output):
                        print(f"[HOOK] {module.__class__.__name__}")
                        print(f"→ input len: {len(input)}")

                        if len(input) == 1:
                            hidden_states = input[0]
                            print("→ hidden_states dtype:", hidden_states.dtype)
                        elif len(input) >= 3:
                            hidden_states, encoder_hidden_states, attention_mask = input[:3]
                            print("→ hidden_states dtype:", hidden_states.dtype)
                            print("→ encoder_hidden_states dtype:", encoder_hidden_states.dtype)
                            print("→ attention_mask dtype:", attention_mask.dtype if attention_mask is not None else "None")
                        else:
                            print(f"[Warning] Unexpected input format: {input}")
                    # forward 전에 hook 등록
                    hook_handles = register_attn2_hooks(accelerator.unwrap_model(unet), name="unet")
                    
                    # hook을 block 0의 attn2에 적용
                    block = accelerator.unwrap_model(unet).transformer_blocks[0]
                    hook_handle = block.attn2.register_forward_hook(attn2_input_hook)
                    
                    
                    hook_handles = register_attn2_hooks(ref_unet, name="ref_unet")
                    
                    # hook을 block 0의 attn2에 적용
                    block = ref_unet.transformer_blocks[0]
                    hook_handle = block.attn2.register_forward_hook(attn2_input_hook)
                    unet_hooks = register_nan_hooks_for_unet(unet, name="unet")
                    # 기존 hook 제거 후 새로 등록   
                    ref_unet_hooks = register_nan_hooks_for_unet(ref_unet, name="ref_unet") 
                    with torch.no_grad():
                        ref_pos_neg_target = ref_unet(
                            hidden_states=stdpo_latents,
                            encoder_hidden_states=ref_direction_prompt_embeds,
                            encoder_attention_mask=ref_direction_prompt_attention_mask,
                            timestep=stdpo_timesteps,
                            return_dict=False,
                        )[0]
                        print("ref")
                        print("noisy_latents dtype:", noisy_latents.dtype)
                        print("prompt_embeds dtype:", pos_prompt_embeds.dtype)
                        print("timestep dtype:", timesteps.dtype)
                        print("ref unet block 0 attn2.q_proj weight dtype:", ref_unet.transformer_blocks[0].attn2.to_q.weight.dtype)

                        if torch.isnan(ref_pos_neg_target).any():
                            print("⚠️ NaN detected in ref_pos_neg_target")
                            print(f"stdpo_latents NaN: {torch.isnan(stdpo_latents).any().item()}")
                            print(f"ref_direction_prompt_embeds NaN: {torch.isnan(ref_direction_prompt_embeds).any().item()}")
                            print(f"ref_direction_prompt_attention_mask NaN: {torch.isnan(ref_direction_prompt_attention_mask).any().item()}")
                            print(f"stdpo_timesteps: {stdpo_timesteps}")
                            print(f"ref_pos_neg_target stats → mean: {ref_pos_neg_target.mean().item()}, std: {ref_pos_neg_target.std().item()}")
                            raise ValueError("ref_pos_neg_target contains NaNs!")
                        
                        
                        ref_pos_neg_target = ref_unet(
                            hidden_states=noisy_latents,
                            encoder_hidden_states=torch.cat([pos_prompt_embeds, pos_prompt_embeds], dim=0),
                            encoder_attention_mask=torch.cat([pos_prompt_attention_mask, pos_prompt_attention_mask], dim=0),
                            timestep=timesteps,
                            return_dict=False,
                        )[0]
                        
                        if torch.isnan(ref_pos_neg_target).any():
                            print("⚠️ NaN detected in ref_pos_neg_target")
                            print(f"stdpo_latents NaN: {torch.isnan(stdpo_latents).any().item()}")
                            print(f"ref_direction_prompt_embeds NaN: {torch.isnan(ref_direction_prompt_embeds).any().item()}")
                            print(f"ref_direction_prompt_attention_mask NaN: {torch.isnan(ref_direction_prompt_attention_mask).any().item()}")
                            print(f"stdpo_timesteps: {stdpo_timesteps}")
                            print(f"ref_pos_neg_target stats → mean: {ref_pos_neg_target.mean().item()}, std: {ref_pos_neg_target.std().item()}")
                            raise ValueError("ref_pos_neg_target contains NaNs!")

                    target[final_indices] = ref_pos_neg_target
                with torch.autograd.set_detect_anomaly(True):
                    model_pred = unet(
                                hidden_states=noisy_latents.to(dtype=torch.float32),
                                encoder_hidden_states=torch.cat([pos_prompt_embeds, pos_prompt_embeds], dim=0).to(dtype=torch.float32),
                                encoder_attention_mask=torch.cat([pos_prompt_attention_mask, pos_prompt_attention_mask], dim=0).to(dtype=torch.float32),
                                timestep=timesteps,
                                return_dict=False,
                                )[0]
                    
                    
                print("step", step)
                if torch.isnan(model_pred).any():
                    print("🚨 NaN detected in model_pred")
                    print(f"noisy_latents NaN: {torch.isnan(noisy_latents).any().item()}")
                    print(f"pos_prompt_embeds NaN: {torch.isnan(pos_prompt_embeds).any().item()}")
                    print(f"pos_prompt_attention_mask NaN: {torch.isnan(pos_prompt_attention_mask).any().item()}")
                    print(f"timesteps NaN: {torch.isnan(timesteps).any().item()} | timesteps min/max: {timesteps.min().item()} / {timesteps.max().item()}")
                    print(f"model_pred stats → mean: {model_pred.mean().item()}, std: {model_pred.std().item()}")
                    raise ValueError("model_pred contains NaNs!")
                
                # 1️⃣ Keep latents & embeds in the same dtype as the weights
                weight_dtype = next(accelerator.unwrap_model(unet).parameters()).dtype  # fp16 or bf16
                noisy_latents    = noisy_latents.to(weight_dtype)
                pos_prompt_embeds = pos_prompt_embeds.to(weight_dtype)
                pos_prompt_attention_mask = pos_prompt_attention_mask.to(weight_dtype)

                # 2️⃣ Call under autocast → math done in fp32, storage in fp16
                with torch.cuda.amp.autocast(dtype=weight_dtype):
                    model_pred = unet(
                        hidden_states=noisy_latents,
                        encoder_hidden_states=torch.cat([pos_prompt_embeds]*2, 0),
                        encoder_attention_mask=torch.cat([pos_prompt_attention_mask]*2, 0),
                        timestep=timesteps,
                        return_dict=False,
                    )[0]
                print("std")
                print("noisy_latents dtype:", noisy_latents.dtype)
                print("prompt_embeds dtype:", pos_prompt_embeds.dtype)
                print("timestep dtype:", timesteps.dtype)
                print("unet block 0 attn2.q_proj weight dtype:", accelerator.unwrap_model(unet).transformer_blocks[0].attn2.to_q.weight.dtype)

                print("step", step)
                if torch.isnan(model_pred).any():
                    print("🚨 NaN detected in model_pred")
                    print(f"noisy_latents NaN: {torch.isnan(noisy_latents).any().item()}")
                    print(f"pos_prompt_embeds NaN: {torch.isnan(pos_prompt_embeds).any().item()}")
                    print(f"pos_prompt_attention_mask NaN: {torch.isnan(pos_prompt_attention_mask).any().item()}")
                    print(f"timesteps NaN: {torch.isnan(timesteps).any().item()} | timesteps min/max: {timesteps.min().item()} / {timesteps.max().item()}")
                    print(f"model_pred stats → mean: {model_pred.mean().item()}, std: {model_pred.std().item()}")
                    raise ValueError("model_pred contains NaNs!")
                
                
                #### START LOSS COMPUTATION ####
                if args.train_method == 'sft': # SFT, casting for F.mse_loss
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                elif args.train_method == 'dpo':
                    # model_pred and ref_pred will be (2 * LBS) x 4 x latent_spatial_dim x latent_spatial_dim
                    # losses are both 2 * LBS
                    # 1st half of tensors is preferred (y_w), second half is unpreferred
                    model_losses = (model_pred - target).pow(2).mean(dim=[1,2,3])
                    model_losses_w, model_losses_l = model_losses.chunk(2)
                    # below for logging purposes
                    raw_model_loss = 0.5 * (model_losses_w.mean() + model_losses_l.mean())
                    
                    model_diff = model_losses_w - model_losses_l # These are both LBS (as is t)
                    
                    with torch.no_grad(): # Get the reference policy (unet) prediction
                        ref_pred = ref_unet(
                            hidden_states=noisy_latents,
                            encoder_hidden_states=torch.cat([pos_prompt_embeds, pos_prompt_embeds], dim=0),
                            encoder_attention_mask=torch.cat([pos_prompt_attention_mask, pos_prompt_attention_mask], dim=0),
                            timestep=timesteps,
                            return_dict=False,
                            )[0]
                        ref_losses = (ref_pred - target).pow(2).mean(dim=[1,2,3])
                        ref_losses_w, ref_losses_l = ref_losses.chunk(2)
                        ref_diff = ref_losses_w - ref_losses_l
                        raw_ref_loss = ref_losses.mean()    
                        
                    scale_term = -0.5 * args.beta_dpo
                    inside_term = scale_term * (model_diff - ref_diff)
                    implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
                    loss = -1 * F.logsigmoid(inside_term).mean()
                #### END LOSS COMPUTATION ###
                    
                # Gather the losses across all processes for logging 
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                # Also gather:
                # - model MSE vs reference MSE (useful to observe divergent behavior)
                # - Implicit accuracy
                if args.train_method == 'dpo':
                    avg_model_mse = accelerator.gather(raw_model_loss.repeat(args.train_batch_size)).mean().item()
                    avg_ref_mse = accelerator.gather(raw_ref_loss.repeat(args.train_batch_size)).mean().item()
                    avg_acc = accelerator.gather(implicit_acc).mean().item()
                    implicit_acc_accumulated += avg_acc / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if not args.use_adafactor: # Adafactor does itself, maybe could do here to cut down on code
                        accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has just performed an optimization step, if so do "end of batch" logging
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                if args.train_method == 'dpo':
                    accelerator.log({"model_mse_unaccumulated": avg_model_mse}, step=global_step)
                    accelerator.log({"ref_mse_unaccumulated": avg_ref_mse}, step=global_step)
                    accelerator.log({"implicit_acc_accumulated": implicit_acc_accumulated}, step=global_step)
                train_loss = 0.0
                implicit_acc_accumulated = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        logger.info("Pretty sure saving/loading is fixed but proceed cautiously")

                # if global_step % args.eval_steps == 0:
                #     unwrapped_unet = accelerator.unwrap_model(unet)
                #     evaluate_alignment(accelerator, global_step, unwrapped_unet, args, weight_dtype)

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if args.train_method == 'dpo':
                logs["implicit_acc"] = avg_acc
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break


    # Create the pipeline using the trained modules and save it.
    # This will save to top level of output_dir instead of a checkpoint directory
    accelerator.wait_for_everyone()

    # unwrapped_unet = accelerator.unwrap_model(unet)
    # evaluate_alignment(accelerator, global_step, unwrapped_unet, args, weight_dtype)
    
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.sdxl:
            # Serialize pipeline.
            vae = AutoencoderKL.from_pretrained(
                vae_path,
                subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
                revision=args.revision,
                torch_dtype=weight_dtype,
            )
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                args.pretrained_model_name_or_path, unet=unet, vae=vae, revision=args.revision, torch_dtype=weight_dtype
            )
            pipeline.save_pretrained(args.output_dir)
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=text_encoder,
                vae=vae,
                unet=unet,
                revision=args.revision,
                cache_dir=args.cache_dir
            )
        pipeline.save_pretrained(args.output_dir)


    accelerator.end_training()


if __name__ == "__main__":
    main()
