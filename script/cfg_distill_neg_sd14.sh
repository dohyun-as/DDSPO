#!/bin/bash
# Negative-prompt CFG distillation (Meng et al. style) on SD1.4.
# Uses pre-encoded latents from data/latents/diffusiondb_llama_200k. Each step:
#   eps_pos = ref_unet(x_t, t, c)
#   eps_neg = ref_unet(x_t, t, c-)         # c- = random-removal of c
#   target  = eps_neg + s * (eps_pos - eps_neg),    s ~ U[GUIDANCE_MIN, GUIDANCE_MAX]
#   loss    = || student_unet(x_t, t, c; w_emb(s)) - sg[target] ||^2

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NUM_GPUS=${NUM_GPUS:-4}

MODEL_NAME="CompVis/stable-diffusion-v1-4"
DATA_DIR="./data/latents/diffusiondb_llama_200k"
OUTPUT_DIR=${OUTPUT_DIR:-./results/cfg_distill_neg/sd14_meng_w2to8}

# Hparams follow Meng et al. "On Distillation of Guided Diffusion Models"
# Table 6 (Stage I, w-conditioned student): LR=5e-5, eff. batch=512,
# steps=50K, linear warmup=500. Effective batch = BS × grad_accum × NUM_GPUS
# = 8 × 16 × 4 = 512. NO --scale_lr (LR is the raw per-step rate, Adam-style).
LR=${LR:-5e-5}
TRAIN_BS=${TRAIN_BS:-8}
GRAD_ACCUM=${GRAD_ACCUM:-16}
MAX_STEPS=${MAX_STEPS:-50000}
WARMUP=${WARMUP:-500}
CKPT_STEPS=${CKPT_STEPS:-5000}
NUM_WORKERS=${NUM_WORKERS:-8}

# w-conditioning: student takes guidance_scale s ~ U[GUIDANCE_MIN, GUIDANCE_MAX].
# Internally we embed (s - 1) so this matches diffusers pipeline convention
# (`pipe(..., guidance_scale=s)` -> `embed(s - 1)`).
GUIDANCE_MIN=${GUIDANCE_MIN:-2.0}
GUIDANCE_MAX=${GUIDANCE_MAX:-8.0}
TIME_COND_PROJ_DIM=${TIME_COND_PROJ_DIM:-256}

# bf16 by default — fp16 is unstable on Blackwell + SD1.4 (loss diverges to
# NaN within a few steps). bf16 has fp32-equivalent dynamic range and is
# native on Ampere+/Blackwell, with SDPA at the same throughput as fp16.
MIXED_PRECISION=${MIXED_PRECISION:-bf16}

COMMON_ARGS="--pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --cache_dir ./cache \
  --train_batch_size=$TRAIN_BS \
  --dataloader_num_workers=$NUM_WORKERS \
  --gradient_accumulation_steps=$GRAD_ACCUM \
  --max_train_steps=$MAX_STEPS \
  --lr_scheduler=constant_with_warmup --lr_warmup_steps=$WARMUP \
  --learning_rate=$LR \
  --checkpointing_steps $CKPT_STEPS \
  --time_cond_proj_dim $TIME_COND_PROJ_DIM \
  --guidance_min $GUIDANCE_MIN \
  --guidance_max $GUIDANCE_MAX \
  --mixed_precision $MIXED_PRECISION \
  --gradient_checkpointing \
  --allow_tf32
"

mkdir -p "$OUTPUT_DIR"
echo "$COMMON_ARGS" > "$OUTPUT_DIR/args.txt"

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
if [ ${NUM_GPUS} -gt 1 ]; then
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch \
        --main_process_port $PORT --multi_gpu --num_processes ${NUM_GPUS} \
        training/cfg_distill_neg.py $COMMON_ARGS
else
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch \
        training/cfg_distill_neg.py $COMMON_ARGS
fi
