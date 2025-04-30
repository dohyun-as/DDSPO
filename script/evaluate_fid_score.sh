# ------------------------------------------------------------------------------------
# Copyright 2023. Nota Inc. All Rights Reserved.
# Please ensure that the following libraries are successfully installed:
#   for IS, https://github.com/toshas/torch-fidelity
#   for FID, https://github.com/mseitzer/pytorch-fid
#   for CLIP score, https://github.com/mlfoundations/open_clip
# ------------------------------------------------------------------------------------

CUDA_VISIBLE_DEVICES=4
NUM_GPUS=1

RESULT_PATHS=(
    "./results/hypertuning/only_cfg_g1_variant4_replace_neg_with_other_pos/checkpoint-100"
    "./results/hypertuning/only_cfg_g1_variant4_replace_neg_with_other_pos/checkpoint-200"
    "./results/hypertuning/only_cfg_g1_variant4_replace_neg_with_other_pos/checkpoint-300"
    "./results/hypertuning/only_cfg_g1_variant4_replace_neg_with_other_pos/checkpoint-400"
    "./results/hypertuning/only_cfg_g1_variant4_replace_neg_with_other_pos/checkpoint-500"
    "./results/hypertuning/only_cfg_g1_variant4_beta_random_neg_prompts/checkpoint-100"
    "./results/hypertuning/only_cfg_g1_variant4_beta_random_neg_prompts/checkpoint-200"
    "./results/hypertuning/only_cfg_g1_variant4_beta_random_neg_prompts/checkpoint-300"
    "./results/hypertuning/only_cfg_g1_variant4_beta_random_neg_prompts/checkpoint-400"
    "./results/hypertuning/only_cfg_g1_variant4_beta_random_neg_prompts/checkpoint-500"
)



for RESULT_DIR in "${RESULT_PATHS[@]}"; do
    OUTDIR="${RESULT_DIR}/MSCOCO_img"

    IMG_PATH=$OUTDIR/im256

    echo "=== Inception Score (IS) ==="
    IS_TXT=$OUTDIR/im256_is.txt
    fidelity --gpu $CUDA_VISIBLE_DEVICES --isc --input1 $IMG_PATH | tee $IS_TXT
    echo "============"

    echo "=== Fréchet Inception Distance (FID) ==="
    FID_TXT=$OUTDIR/im256_fid.txt
    NPZ_NAME_gen=$OUTDIR/im256_fid.npz
    NPZ_NAME_real=./evaluation/mscoco_val2014/real_im256.npz
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 -m pytorch_fid --save-stats $IMG_PATH $NPZ_NAME_gen
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 -m pytorch_fid $NPZ_NAME_real $NPZ_NAME_gen | tee $FID_TXT
    echo "============"

    # echo "=== CLIP Score ==="
    # CLIP_TXT=./results/$OUTDIR/im256_clip.txt
    # CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 src/eval_clip_score.py --img_dir $IMG_PATH --save_txt $CLIP_TXT
    # echo "============"
done