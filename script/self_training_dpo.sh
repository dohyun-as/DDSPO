#!/bin/bash

#If you want to use restriced number of GPUs, you can set the CUDA_VISIBLE_DEVICES and NUM_GPUS environment variables
# to the desired values before running this script.
# Example: CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2
CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

MODEL_NAME="CompVis/stable-diffusion-v1-4"
DATA_DIR="./data/test"

COMMON_ARGS="--pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --train_batch_size=2 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=2000 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \
  --learning_rate=1e-8 --scale_lr \
  --checkpointing_steps 500 \
  --beta_dpo 5000 \
  --output_dir="results/test"
"

if [ ${NUM_GPUS} -gt 1 ]; then
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --multi_gpu --num_processes ${NUM_GPUS} training/dpo.py $COMMON_ARGS
else
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch training/dpo.py $COMMON_ARGS
fi