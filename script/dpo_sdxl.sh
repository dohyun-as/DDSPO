#!/bin/bash

#If you want to use restriced number of GPUs, you can set the CUDA_VISIBLE_DEVICES and NUM_GPUS environment variables
# to the desired values before running this script.
# Example: CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2
CUDA_VISIBLE_DEVICES=0,1 #$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)
NUM_GPUS=2 #$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
DATA_DIR="./data/latents/SDXL_diffusiondb_removal_200k"
EXTRA_TEXT_PATH="./data/captions/diffusiondb_removal/diffusiondb_removal.jsonl"
OUTPUT_DIR="results/SDXL/dpo_beta12k"

COMMON_ARGS="--pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --train_batch_size=8 \
  --dataloader_num_workers=8 \
  --gradient_accumulation_steps=128 \
  --max_train_steps=500 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=100 \
  --learning_rate=3.3e-9 --scale_lr \
  --checkpointing_steps 50 \
  --beta_dpo 12000 \
  --output_dir=$OUTPUT_DIR \
  --cache_dir "./cache" \
  --sdxl \
"
  # --only_cfg \
  # --guidance_scale 1 \
  # --sdxl

mkdir -p "${OUTPUT_DIR}"
echo "$COMMON_ARGS" > "$OUTPUT_DIR/args.txt"

#batch_size=16/2x64x4=2048
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
if [ ${NUM_GPUS} -gt 1 ]; then
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT --multi_gpu --num_processes ${NUM_GPUS} training/dpo_cfg_randcond_sdxl.py $COMMON_ARGS
else
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch training/dpo_cfg_randcond_sdxl.py $COMMON_ARGS
fi