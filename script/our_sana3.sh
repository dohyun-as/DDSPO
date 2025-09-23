#!/bin/bash

#If you want to use restriced number of GPUs, you can set the CUDA_VISIBLE_DEVICES and NUM_GPUS environment variables
# to the desired values before running this script.
# Example: CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2
CUDA_VISIBLE_DEVICES=5,6 #$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)
NUM_GPUS=2 #$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

MODEL_NAME="Efficient-Large-Model/Sana_600M_1024px_diffusers"
DATA_DIR="./data/latents/SANA_diffusiondb_removal_200k"
EXTRA_TEXT_PATH="./data/captions/diffusiondb_removal/diffusiondb_removal.jsonl"
OUTPUT_DIR="results/SANA/ours_beta2k_lora1024"

COMMON_ARGS="--pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --train_batch_size=4 \
  --dataloader_num_workers=8 \
  --gradient_accumulation_steps=256 \
  --max_train_steps=500 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --learning_rate=2.0e-8 --scale_lr \
  --checkpointing_steps 50 \
  --beta_dpo 2000 \
  --output_dir=$OUTPUT_DIR \
  --cache_dir "./cache" \
  --only_cfg \
  --mixed_precision="no" \
  --resolution=1024 \
  --rank 1024
"

mkdir -p "${OUTPUT_DIR}"
echo "$COMMON_ARGS" > "$OUTPUT_DIR/args.txt"

#batch_size=16/2x64x4=2048
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
if [ ${NUM_GPUS} -gt 1 ]; then
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT --multi_gpu --num_processes ${NUM_GPUS} training/dpo_cfg_randcond_sana_re.py $COMMON_ARGS
else
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch training/dpo_cfg_randcond_sana_re.py $COMMON_ARGS
fi