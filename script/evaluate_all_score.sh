#!/bin/bash

# Set GPU info
CUDA_VISIBLE_DEVICES=7 #$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)
NUM_GPUS=1 #$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# List of result directories to evaluate
RESULT_PATHS=(
    "./results/SDXL/our_beta16k_sdxl/checkpoint-100"
)

    # "./results/aesthetic/dpo_beta8k/checkpoint-100"
    # "./results/aesthetic/our_beta16k/checkpoint-100"
    # "./results/teacher/std-v1-4"
    # "./results/teacher/SDXL-1"
    # "./results/final_ours_sd14_align/align_our_beta16k/checkpoint-100"
    # "./results/final_ours_sd14_align/align_dpo_beta8k/checkpoint-100"
    # "./results/teacher/Diffusion_DPO-v1-5"

    # "./results/final_ours_sd14_align/DDSPO_beta16k_batch512/checkpoint-100"
    # "./results/final_ours_sd14_align/DDSPO_beta16k_batch512/checkpoint-200"
    # "./results/final_ours_sd14_align/DPO_beta8k_batch512/checkpoint-100"
    # "./results/final_ours_sd14_align/DPO_beta8k_batch512/checkpoint-200"

# for RESULT_DIR in "${RESULT_PATHS[@]}"; do
#     OUTDIR="${RESULT_DIR}/geneval_img"
#     OUTFILE="${RESULT_DIR}/geneval_results.jsonl"
#     UNET_PATH="${RESULT_DIR}/unet"
#     METADATA="./evaluation/geneval/prompts/evaluation_metadata.jsonl"
#     DETECTOR="./evaluation/geneval/OBJECT_DETECTOR_FOLDER"

#     COMMON_ARGS2="$OUTDIR \
#     --outfile "$OUTFILE" \
#     --model-path "$DETECTOR"
#     "

#     echo "🔁 Running generation and evaluation for: $RESULT_DIR"

#     if [ ${NUM_GPUS} -gt 1 ]; then
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
#         /workspace/miniconda3/bin/conda run -n geneval \
#         accelerate launch --multi_gpu --num_processes ${NUM_GPUS} evaluation/geneval_evaluate.py $COMMON_ARGS2
#     else
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
#         /workspace/miniconda3/bin/conda run -n geneval \
#         accelerate launch evaluation/geneval_evaluate.py $COMMON_ARGS2
#     fi


#     OUTFILES+=("$OUTFILE")
#     # Print summary scores
#     python evaluation/geneval/evaluation/summary_scores.py "$OUTFILE"
# done


# echo "📊 Generating CSV summary..."
# python evaluation/geneval_score.py "${OUTFILES[@]}" --output_csv geneval_summary_1.csv


# for ckpt in "${RESULT_PATHS[@]}"; do
#     outdir="${ckpt}/Pratiprompt_img"

#     COMMON_ARGS2="\
#     --image_dir "$outdir"
#     "

#     CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
#     /workspace/miniconda3/bin/conda run -n compbench \
#     python evaluation/pcikscore_evaluate.py $COMMON_ARGS2

# done



# for RESULT_DIR in "${RESULT_PATHS[@]}"; do
#     OUTDIR="${RESULT_DIR}/MSCOCO_img"

#     IMG_PATH=$OUTDIR/im256

#     # echo "=== Inception Score (IS) ==="
#     # IS_TXT=$OUTDIR/im256_is.txt
#     prc_TXT=$OUTDIR/im256_prc.txt
#     # fidelity --gpu $CUDA_VISIBLE_DEVICES --isc --input1 $IMG_PATH | tee $IS_TXT
#     # echo "============"

#     echo "=== all ==="
#     fidelity --gpu 0 --isc --fid --prc --input1 $IMG_PATH --input2 "/workspace/Diffusion_align/evaluation/mscoco_val2014/val2014_256" | tee $prc_TXT
#     echo "==========="

#     # echo "=== Fréchet Inception Distance (FID) ==="
#     # FID_TXT=$OUTDIR/im256_fid.txt
#     # NPZ_NAME_gen=$OUTDIR/im256_fid.npz
#     # NPZ_NAME_real=./evaluation/mscoco_val2014/real_im256.npz
#     # CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 -m pytorch_fid --save-stats $IMG_PATH $NPZ_NAME_gen
#     # CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 -m pytorch_fid $NPZ_NAME_real $NPZ_NAME_gen | tee $FID_TXT
#     # echo "============"

#     # echo "=== CLIP Score ==="
#     # CLIP_TXT=./results/$OUTDIR/im256_clip.txt
#     # CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 src/eval_clip_score.py --img_dir $IMG_PATH --save_txt $CLIP_TXT
#     # echo "============"
# done



# # RESULT_PATHS=(
# #     "./results/SDXL/ours_beta22k/checkpoint-100" 
# # )


# evaluate_project() {
#   local project_dir=$1     # e.g. "BLIPvqa_eval"
#   local eval_script=$2     # e.g. "BLIP_vqa.py"
#   local out_dir=$3         # output directory

#   cd "evaluation/T2I-CompBench/$project_dir" || exit 1
#   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} /workspace/miniconda3/bin/conda run -n compbench python "$eval_script" --out_dir="$out_dir"
#   cd - || exit 1
# }
# evaluate_project_path() {
#   local project_dir=$1     # e.g. "BLIPvqa_eval"
#   local eval_script=$2     # e.g. "BLIP_vqa.py"
#   local out_dir=$3         # output directory

#   cd "evaluation/T2I-CompBench/$project_dir" || exit 1
#   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} /workspace/miniconda3/bin/conda run -n compbench python "$eval_script" --outpath="$out_dir"
#   cd - || exit 1
# }

# # ---------------------------------------------------------
# # List of directories to evaluate
# # ---------------------------------------------------------

# # ---------------------------------------------------------
# # 실행
# # ---------------------------------------------------------
# for out_dir_base in "${RESULT_PATHS[@]}"; do
#   echo "Starting evaluation for $out_dir_base..."
#   out_dir_base=$(readlink -f "$out_dir_base")
#   # -------------------------------------------------------
#   # Step 1: BLIP_vqa evaluation (color, shape, texture, complex)
#   # -------------------------------------------------------
#   echo "Starting BLIP_vqa evaluation..."

#   # color
#   out_dir="$out_dir_base/compbench_img/color"
#   evaluate_project "BLIPvqa_eval" "BLIP_vqa.py" "$out_dir"

#   # shape
#   out_dir="$out_dir_base/compbench_img/shape"
#   evaluate_project "BLIPvqa_eval" "BLIP_vqa.py" "$out_dir"

#   # texture
#   out_dir="$out_dir_base/compbench_img/texture"
#   evaluate_project "BLIPvqa_eval" "BLIP_vqa.py" "$out_dir"

#   # complex
#   out_dir="$out_dir_base/compbench_img/complex"
#   evaluate_project "BLIPvqa_eval" "BLIP_vqa.py" "$out_dir"

#   # -------------------------------------------------------
#   # Step 2: UniDet_eval (spatial, complex)
#   # -------------------------------------------------------
#   echo "Starting UniDet_eval evaluation..."

#   # spatial
#   out_dir="$out_dir_base/compbench_img/spatial"
#   evaluate_project_path "UniDet_eval" "2D_spatial_eval.py" "$out_dir"

#   # complex (2D_spatial_eval.py)
#   out_dir="$out_dir_base/compbench_img/complex"
#   evaluate_project_path "UniDet_eval" "2D_spatial_eval.py" "$out_dir"
#   # -------------------------------------------------------
#   # Step 3: CLIPScore_eval (non_spatial, complex)
#   # -------------------------------------------------------
#   echo "Starting CLIPScore evaluation..."

#   # non_spatial
#   out_dir="$out_dir_base/compbench_img/non_spatial"
#   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} /workspace/miniconda3/bin/conda run -n compbench python evaluation/T2I-CompBench/CLIPScore_eval/CLIP_similarity.py --outpath="$out_dir"

#   # complex
#   out_dir="$out_dir_base/compbench_img/complex"
#   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} /workspace/miniconda3/bin/conda run -n compbench python evaluation/T2I-CompBench/CLIPScore_eval/CLIP_similarity.py --outpath="$out_dir"

#   # -------------------------------------------------------
#   # Step 4: 3_in_1_eval (complex)
#   # -------------------------------------------------------
#   echo "Starting 3_in_1_eval evaluation..."

#   out_dir="$out_dir_base/compbench_img/complex"
#   evaluate_project_path "3_in_1_eval" "3_in_1.py" "$out_dir"

#   echo "Evaluation for $out_dir_base completed successfully!"
  
#   OUTFILES+=("$out_dir_base/compbench_img")
# done

# python evaluation/compbench_score.py "${OUTFILES[@]}" --output compbench_score.csv
# echo "All evaluations completed successfully!"




# # for ckpt in "${RESULT_PATHS[@]}"; do
# #     outdir="${ckpt}/Pratiprompt_img"

# #     COMMON_ARGS2="\
# #     --image_dir "$outdir"
# #     "

# #     CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
# #     /workspace/miniconda3/bin/conda run -n compbench \
# #     python evaluation/pcikscore_evaluate.py $COMMON_ARGS2

# # done

# RESULT_PATHS=(
#     "./results/teacher/SPO_SDXL"
# )

# for ckpt in "${RESULT_PATHS[@]}"; do
#     outdir="${ckpt}/Pratiprompt_img"

#     COMMON_ARGS2="\
#     --image_dir "$outdir"
#     "

#     CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
#     /workspace/miniconda3/bin/conda run -n compbench \
#     python evaluation/pcikscore_evaluate.py $COMMON_ARGS2

# done


RESULT_PATHS=(
    "./results/teacher/Diffusion_KTO"
)


for ckpt in "${RESULT_PATHS[@]}"; do
    outdir="${ckpt}/HPS_img"

    COMMON_ARGS2="\
    --image_path "$outdir" \
    --hps_version "v2.0" \
    "

    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    /workspace/miniconda3/bin/conda run -n compbench \
    python evaluation/HPSv2_score.py $COMMON_ARGS2

done