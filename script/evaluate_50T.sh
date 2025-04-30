#!/bin/bash
# This script is used to generate latents for the images in the specified directory.

#If you want to use restriced number of GPUs, you can set the CUDA_VISIBLE_DEVICES and NUM_GPUS environment variables
# to the desired values before running this script.
# Example: CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2
# CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)
# NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# COMMON_ARGS=""./evaluation/geneval/prompts/evaluation_metadata.jsonl" \
# --model "CompVis/stable-diffusion-v1-4" \
# --outdir "./results/dpo_method_2/checkpoint-100/geneval_img" \
# --batch_size 4 \
# --scale 7.5
# "
# # --unet_path "./checkpoint-500/unet" \

# COMMON_ARGS2=""./results/dpo_method_2/checkpoint-100/geneval_img" \
#     --outfile "./results/dpo_method_2/checkpoint-100/geneval_results.jsonl" \
#     --model-path "./evaluation/geneval/OBJECT_DETECTOR_FOLDER"
#     "

# if [ ${NUM_GPUS} -gt 1 ]; then
#   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
#   accelerate launch --multi_gpu --num_processes ${NUM_GPUS} evaluation/geneval_generate.py $COMMON_ARGS
  
#   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
#   conda run -n geneval \
#   accelerate launch --multi_gpu --num_processes ${NUM_GPUS} evaluation/geneval_evaluate.py $COMMON_ARGS2
# else
#   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
#   accelerate launch evaluation/geneval_generate.py $COMMON_ARGS
  
#   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
#   conda run -n geneval \
#   accelerate launch evaluation/geneval_evaluate.py $COMMON_ARGS2
# fi


# python evaluation/geneval/evaluation/summary_scores.py "./results/dpo_method_2/checkpoint-100/geneval_results.jsonl"


#!/bin/bash

# Set GPU info
CUDA_VISIBLE_DEVICES=2,3 #$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)
NUM_GPUS=2 #$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# List of result directories to evaluate
RESULT_PATHS=(
    "/workspace/data_config_DB_200k_50T/checkpoint-100"
    "/workspace/data_config_DB_200k_50T/checkpoint-200"
    "/workspace/data_config_DB_200k_50T/checkpoint-300"
)

    # "./results/stdpo_base_cfg_randcond/checkpoint-100"
for RESULT_DIR in "${RESULT_PATHS[@]}"; do
    OUTDIR="${RESULT_DIR}/geneval_img"
    OUTFILE="${RESULT_DIR}/geneval_results.jsonl"
    UNET_PATH="${RESULT_DIR}/unet"
    METADATA="./evaluation/geneval/prompts/evaluation_metadata.jsonl"
    DETECTOR="./evaluation/geneval/OBJECT_DETECTOR_FOLDER"

    COMMON_ARGS="$METADATA \
    --model "CompVis/stable-diffusion-v1-4" \
    --outdir "$OUTDIR" \
    --unet_path "$UNET_PATH" \
    --batch_size 4 \
    --scale 7.5 \
    --steps 25 \
    --cache_dir "./cache"
    "

    COMMON_ARGS2="$OUTDIR \
    --outfile "$OUTFILE" \
    --model-path "$DETECTOR"
    "

    echo "🔁 Running generation and evaluation for: $RESULT_DIR"

    if [ ${NUM_GPUS} -gt 1 ]; then
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
        accelerate launch --multi_gpu --num_processes ${NUM_GPUS} evaluation/geneval_generate.py $COMMON_ARGS

        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
        /workspace/miniconda3/bin/conda run -n geneval \
        accelerate launch --multi_gpu --num_processes ${NUM_GPUS} evaluation/geneval_evaluate.py $COMMON_ARGS2
    else
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
        accelerate launch evaluation/geneval_generate.py $COMMON_ARGS

        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
        /workspace/miniconda3/bin/conda run -n geneval \
        accelerate launch evaluation/geneval_evaluate.py $COMMON_ARGS2
    fi


    OUTFILES+=("$OUTFILE")
    # Print summary scores
    python evaluation/geneval/evaluation/summary_scores.py "$OUTFILE"
done


echo "📊 Generating CSV summary..."
python evaluation/geneval_score.py "${OUTFILES[@]}" --output_csv geneval_summary.csv