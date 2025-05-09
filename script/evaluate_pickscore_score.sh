# ------------------------------------------------------------------------------------
# Copyright 2023. Nota Inc. All Rights Reserved.
# Please ensure that the following libraries are successfully installed:
#   for IS, https://github.com/toshas/torch-fidelity
#   for FID, https://github.com/mseitzer/pytorch-fid
#   for CLIP score, https://github.com/mlfoundations/open_clip
# ------------------------------------------------------------------------------------

CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1

RESULT_PATHS=(
    "./results/aesthetic/dpo_beta8k/checkpoint-100"
    "./results/aesthetic/dpo_beta8k/checkpoint-200"
    "./results/aesthetic/dpo_beta8k/checkpoint-300"
    "./results/aesthetic/dpo_beta8k/checkpoint-400"
    "./results/aesthetic/dpo_beta8k/checkpoint-500"
)



for ckpt in "${RESULT_PATHS[@]}"; do
    outdir="${ckpt}/Pratiprompt_img"

    COMMON_ARGS2="\
    --image_dir "$outdir"
    "

    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    /workspace/miniconda3/bin/conda run -n compbench \
    python evaluation/pcikscore_evaluate.py $COMMON_ARGS2

done
