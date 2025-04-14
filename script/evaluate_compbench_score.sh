#!/bin/bash

# ---------------------------------------------------------
# out_dir에서 맨 앞의 '../'를 제거하는 함수 (필요한 경우만)
# ---------------------------------------------------------
remove_leading_parent_dir() {
  local path="$1"
  # '../'로 시작한다면 앞의 3글자('../') 제거
  if [[ "$path" == ../* ]]; then
    echo "${path:3}"
  else
    echo "$path"
  fi
}

# ---------------------------------------------------------
# project_dir 로 이동 후 eval_script 실행, 원위치 복귀
# ---------------------------------------------------------
evaluate_project() {
  local project_dir=$1     # 예: "BLIPvqa_eval"
  local eval_script=$2     # 예: "BLIP_vqa.py"
  local out_dir=$3         # 출력 디렉토리

  cd "$project_dir" || exit 1
  python "$eval_script" --out_dir="$out_dir"
  cd - || exit 1
}
evaluate_project_path() {
  local project_dir=$1     # 예: "BLIPvqa_eval"
  local eval_script=$2     # 예: "BLIP_vqa.py"
  local out_dir=$3         # 출력 디렉토리

  cd "$project_dir" || exit 1
  python "$eval_script" --outpath="$out_dir"
  cd - || exit 1
}

# ---------------------------------------------------------
# 실제로 돌릴 체크포인트/모델 output 디렉토리 배열
# ---------------------------------------------------------
output_dirs=(
  # "../generated_img/CompVis/stable-diffusion-v1-4"
  # "../generated_img/stabilityai/stable-diffusion-xl-base-1.0"
  # "../generated_img/RC/SDXL_100000"
  "../generated_img/test/checkpoint-500"
  "../generated_img/test/checkpoint-1000"
  # "../generated_img/abl_BK_T_G/checkpoint-100000"
  # "../generated_img/abl_BK_R_G/checkpoint-350000"
  # "../generated_img/nota/bk-sdm-base"
  # "../generated_img/abl_RC_R_G/checkpoint-325000"
  # "../generated_img/abl_RC_T_G/checkpoint-100000"
  # "../generated_img/abl_RC_T_R/checkpoint-75000"
)

# ---------------------------------------------------------
# 실행
# ---------------------------------------------------------
for out_dir_base in "${output_dirs[@]}"; do
  echo "Starting evaluation for $out_dir_base..."

  # -------------------------------------------------------
  # Step 1: BLIP_vqa evaluation (color, shape, texture, complex)
  # -------------------------------------------------------
  echo "Starting BLIP_vqa evaluation..."

  # color
  out_dir="$out_dir_base/color"
  evaluate_project "BLIPvqa_eval" "BLIP_vqa.py" "$out_dir"

  # shape
  out_dir="$out_dir_base/shape"
  evaluate_project "BLIPvqa_eval" "BLIP_vqa.py" "$out_dir"

  # texture
  out_dir="$out_dir_base/texture"
  evaluate_project "BLIPvqa_eval" "BLIP_vqa.py" "$out_dir"

  # complex
  out_dir="$out_dir_base/complex"
  evaluate_project "BLIPvqa_eval" "BLIP_vqa.py" "$out_dir"

  # -------------------------------------------------------
  # Step 2: UniDet_eval (spatial, complex)
  # -------------------------------------------------------
  echo "Starting UniDet_eval evaluation..."

  # spatial
  out_dir="$out_dir_base/spatial"
  evaluate_project_path "UniDet_eval" "2D_spatial_eval.py" "$out_dir"

  # complex (2D_spatial_eval.py)
  out_dir="$out_dir_base/complex"
  evaluate_project_path "UniDet_eval" "2D_spatial_eval.py" "$out_dir"
  # -------------------------------------------------------
  # Step 3: CLIPScore_eval (non_spatial, complex)
  # -------------------------------------------------------
  echo "Starting CLIPScore evaluation..."

  # non_spatial
  out_dir="$out_dir_base/non_spatial"
  # 맨 앞에 '../'가 있다면 제거
  out_dir_no_parent="$(remove_leading_parent_dir "$out_dir")"
  python CLIPScore_eval/CLIP_similarity.py --out_dir="$out_dir_no_parent"

  # complex
  out_dir="$out_dir_base/complex"
  out_dir_no_parent="$(remove_leading_parent_dir "$out_dir")"
  python CLIPScore_eval/CLIP_similarity.py --out_dir="$out_dir_no_parent"

  # -------------------------------------------------------
  # Step 4: 3_in_1_eval (complex)
  # -------------------------------------------------------
  echo "Starting 3_in_1_eval evaluation..."

  out_dir="$out_dir_base/complex"
  evaluate_project "3_in_1_eval" "3_in_1.py" "$out_dir"

  echo "Evaluation for $out_dir_base completed successfully!"
done

echo "All evaluations completed successfully!"
