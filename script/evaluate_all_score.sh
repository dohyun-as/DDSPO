#!/bin/bash

# Set GPU info
CUDA_VISIBLE_DEVICES=4 #$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)
NUM_GPUS=1 #$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# List of result directories to evaluate
RESULT_PATHS=(
    "./results/hypertuning/only_cfg_g1_variant4_replace_neg_with_other_pos/checkpoint-100"
    "./results/hypertuning/only_cfg_g1_variant4_replace_neg_with_other_pos/checkpoint-200"
    "./results/hypertuning/only_cfg_g1_variant4_replace_neg_with_other_pos/checkpoint-300"
    "./results/hypertuning/only_cfg_g1_variant4_replace_neg_with_other_pos/checkpoint-400"
    "./results/hypertuning/only_cfg_g1_variant4_replace_neg_with_other_pos/checkpoint-500"
)



for RESULT_DIR in "${RESULT_PATHS[@]}"; do
    OUTDIR="${RESULT_DIR}/geneval_img"
    OUTFILE="${RESULT_DIR}/geneval_results.jsonl"
    UNET_PATH="${RESULT_DIR}/unet"
    METADATA="./evaluation/geneval/prompts/evaluation_metadata.jsonl"
    DETECTOR="./evaluation/geneval/OBJECT_DETECTOR_FOLDER"

    COMMON_ARGS2="$OUTDIR \
    --outfile "$OUTFILE" \
    --model-path "$DETECTOR"
    "

    echo "🔁 Running generation and evaluation for: $RESULT_DIR"

    if [ ${NUM_GPUS} -gt 1 ]; then
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
        /workspace/miniconda3/bin/conda run -n geneval \
        accelerate launch --multi_gpu --num_processes ${NUM_GPUS} evaluation/geneval_evaluate.py $COMMON_ARGS2
    else
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
        /workspace/miniconda3/bin/conda run -n geneval \
        accelerate launch evaluation/geneval_evaluate.py $COMMON_ARGS2
    fi


    OUTFILES+=("$OUTFILE")
    # Print summary scores
    python evaluation/geneval/evaluation/summary_scores.py "$OUTFILE"
done


echo "📊 Generating CSV summary..."
python evaluation/geneval_score.py "${OUTFILES[@]}" --output_csv geneval_summary_1.csv


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




evaluate_project() {
  local project_dir=$1     # e.g. "BLIPvqa_eval"
  local eval_script=$2     # e.g. "BLIP_vqa.py"
  local out_dir=$3         # output directory

  cd "evaluation/T2I-CompBench/$project_dir" || exit 1
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} /workspace/miniconda3/bin/conda run -n compbench python "$eval_script" --out_dir="$out_dir"
  cd - || exit 1
}
evaluate_project_path() {
  local project_dir=$1     # e.g. "BLIPvqa_eval"
  local eval_script=$2     # e.g. "BLIP_vqa.py"
  local out_dir=$3         # output directory

  cd "evaluation/T2I-CompBench/$project_dir" || exit 1
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} /workspace/miniconda3/bin/conda run -n compbench python "$eval_script" --outpath="$out_dir"
  cd - || exit 1
}

# ---------------------------------------------------------
# List of directories to evaluate
# ---------------------------------------------------------

# ---------------------------------------------------------
# 실행
# ---------------------------------------------------------
for out_dir_base in "${RESULT_PATHS[@]}"; do
  echo "Starting evaluation for $out_dir_base..."
  out_dir_base=$(readlink -f "$out_dir_base")
  # -------------------------------------------------------
  # Step 1: BLIP_vqa evaluation (color, shape, texture, complex)
  # -------------------------------------------------------
  echo "Starting BLIP_vqa evaluation..."

  # color
  out_dir="$out_dir_base/compbench_img/color"
  evaluate_project "BLIPvqa_eval" "BLIP_vqa.py" "$out_dir"

  # shape
  out_dir="$out_dir_base/compbench_img/shape"
  evaluate_project "BLIPvqa_eval" "BLIP_vqa.py" "$out_dir"

  # texture
  out_dir="$out_dir_base/compbench_img/texture"
  evaluate_project "BLIPvqa_eval" "BLIP_vqa.py" "$out_dir"

  # complex
  out_dir="$out_dir_base/compbench_img/complex"
  evaluate_project "BLIPvqa_eval" "BLIP_vqa.py" "$out_dir"

  # -------------------------------------------------------
  # Step 2: UniDet_eval (spatial, complex)
  # -------------------------------------------------------
  echo "Starting UniDet_eval evaluation..."

  # spatial
  out_dir="$out_dir_base/compbench_img/spatial"
  evaluate_project_path "UniDet_eval" "2D_spatial_eval.py" "$out_dir"

  # complex (2D_spatial_eval.py)
  out_dir="$out_dir_base/compbench_img/complex"
  evaluate_project_path "UniDet_eval" "2D_spatial_eval.py" "$out_dir"
  # -------------------------------------------------------
  # Step 3: CLIPScore_eval (non_spatial, complex)
  # -------------------------------------------------------
  echo "Starting CLIPScore evaluation..."

  # non_spatial
  out_dir="$out_dir_base/compbench_img/non_spatial"
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} /workspace/miniconda3/bin/conda run -n compbench python evaluation/T2I-CompBench/CLIPScore_eval/CLIP_similarity.py --outpath="$out_dir"

  # complex
  out_dir="$out_dir_base/compbench_img/complex"
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} /workspace/miniconda3/bin/conda run -n compbench python evaluation/T2I-CompBench/CLIPScore_eval/CLIP_similarity.py --outpath="$out_dir"

  # -------------------------------------------------------
  # Step 4: 3_in_1_eval (complex)
  # -------------------------------------------------------
  echo "Starting 3_in_1_eval evaluation..."

  out_dir="$out_dir_base/compbench_img/complex"
  evaluate_project_path "3_in_1_eval" "3_in_1.py" "$out_dir"

  echo "Evaluation for $out_dir_base completed successfully!"
  
  OUTFILES+=("$out_dir_base/compbench_img")
done

python evaluation/compbench_score.py "${OUTFILES[@]}" --output compbench_score.csv
echo "All evaluations completed successfully!"
