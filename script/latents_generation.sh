#!/bin/bash
# This script is used to generate latents for the images in the specified directory.

#If you want to use restriced number of GPUs, you can set the CUDA_VISIBLE_DEVICES and NUM_GPUS environment variables
# to the desired values before running this script.
# Example: CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2
CUDA_VISIBLE_DEVICES=4 # $(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)
NUM_GPUS=1 # $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

COMMON_ARGS="--json_file "./data/captions/diffusiondb_llama/diffusiondb_llama.jsonl" \
--save_dir "./data/latents/diffusiondb_llama_200k/" \
--num_samples 200000 \
--batch_size 128 \
--save_type "latent" \
--num_inference_steps 25 \
--cache_dir "./cache"
"

if [ ${NUM_GPUS} -gt 1 ]; then
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port 23426 --multi_gpu --num_processes ${NUM_GPUS} training/sampling.py $COMMON_ARGS
else
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch training/sampling.py $COMMON_ARGS
fi