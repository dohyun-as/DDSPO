#!/bin/bash

# 체크포인트 목록
ckpt_list=(
    "./results/dpo_method_1/checkpoint-100"
    "./results/dpo_method_1/checkpoint-200"
    "./results/dpo_method_1/checkpoint-300"
    "./results/dpo_method_2_up/checkpoint-100"
    "./results/dpo_method_2_up/checkpoint-200"
    "./results/dpo_method_2_up/checkpoint-300"
    "./results/dpo_method_1/checkpoint-400"
    "./results/dpo_method_1/checkpoint-500"
    "./results/dpo_method_1/checkpoint-600"
    "./results/dpo_method_1/checkpoint-700"
    "./results/dpo_method_1/checkpoint-800"
    "./results/dpo_method_1/checkpoint-900"
    "./results/dpo_method_1/checkpoint-1000"
    "./results/dpo_method_1/checkpoint-1100"
    "./results/dpo_method_1/checkpoint-1200"
    "./results/dpo_method_1/checkpoint-1300"
    "./results/dpo_method_1/checkpoint-1400"
    "./results/dpo_method_1/checkpoint-1500"
    "./results/dpo_method_1/checkpoint-1600"
    "./results/dpo_method_1/checkpoint-1700"
    "./results/dpo_method_1/checkpoint-1800"
    "./results/dpo_method_1/checkpoint-1900"
    "./results/dpo_method_1/checkpoint-2000"
    "./results/dpo_method_2_up/checkpoint-400"
    "./results/dpo_method_2_up/checkpoint-500"
    "./results/dpo_method_2_up/checkpoint-600"
    "./results/dpo_method_2_up/checkpoint-700"
    "./results/dpo_method_2_up/checkpoint-800"
    "./results/dpo_method_2_up/checkpoint-900"
    "./results/dpo_method_2_up/checkpoint-1000"
    "./results/dpo_method_2_up/checkpoint-1100"
    "./results/dpo_method_2_up/checkpoint-1200"
    "./results/dpo_method_2_up/checkpoint-1300"
    "./results/dpo_method_2_up/checkpoint-1400"
    "./results/dpo_method_2_up/checkpoint-1500"
    "./results/dpo_method_2_up/checkpoint-1600"
    "./results/dpo_method_2_up/checkpoint-1700"
    "./results/dpo_method_2_up/checkpoint-1800"
    "./results/dpo_method_2_up/checkpoint-1900"
    "./results/dpo_method_2_up/checkpoint-2000"
    "./results/dpo_method_2/checkpoint-1200"
    "./results/dpo_method_2/checkpoint-1300"
    "./results/dpo_method_2/checkpoint-1400"
    "./results/dpo_method_2/checkpoint-1500"
    "./results/dpo_method_2/checkpoint-1600"
    "./results/dpo_method_2/checkpoint-1700"
    "./results/dpo_method_2/checkpoint-1800"
    "./results/dpo_method_2/checkpoint-1900"
    "./results/dpo_method_2/checkpoint-2000"
)
# (
#     "./results/dpo_method_2/checkpoint-100"
#     "./results/dpo_method_2/checkpoint-200"
#     "./results/dpo_method_2/checkpoint-300"
#     "./results/dpo_method_2/checkpoint-400"
#     "./results/dpo_method_2/checkpoint-500"
#     "./results/dpo_method_2/checkpoint-600"
#     "./results/dpo_method_2/checkpoint-700"
#     "./results/dpo_method_2/checkpoint-800"
#     "./results/dpo_method_2/checkpoint-900"
#     "./results/dpo_method_2/checkpoint-1000"
#     "./results/dpo_method_2/checkpoint-1100"
#     "./results/dpo_method_2/checkpoint-1200"
#     "./results/dpo_method_2/checkpoint-1300"
#     "./results/dpo_method_2/checkpoint-1400"
#     "./results/dpo_method_2/checkpoint-1500"
#     "./results/dpo_method_2/checkpoint-1600"
#     "./results/dpo_method_2/checkpoint-1700"
#     "./results/dpo_method_2/checkpoint-1800"
#     "./results/dpo_method_2/checkpoint-1900"
#     "./results/dpo_method_2/checkpoint-2000"
#     "./results/dpo_method_2_up/checkpoint-100"
#     "./results/dpo_method_2_up/checkpoint-200"
#     "./results/dpo_method_2_up/checkpoint-300"
#     "./results/dpo_method_2_up/checkpoint-400"
#     "./results/dpo_method_2_up/checkpoint-500"
#     "./results/dpo_method_2_up/checkpoint-600"
#     "./results/dpo_method_2_up/checkpoint-700"
#     "./results/dpo_method_2_up/checkpoint-800"
#     "./results/dpo_method_2_up/checkpoint-900"
#     "./results/dpo_method_2_up/checkpoint-1000"
#     "./results/dpo_method_2_up/checkpoint-1100"
#     "./results/dpo_method_2_up/checkpoint-1200"
#     "./results/dpo_method_2_up/checkpoint-1300"
#     "./results/dpo_method_2_up/checkpoint-1400"
#     "./results/dpo_method_2_up/checkpoint-1500"
#     "./results/dpo_method_2_up/checkpoint-1600"
#     "./results/dpo_method_2_up/checkpoint-1700"
#     "./results/dpo_method_2_up/checkpoint-1800"
#     "./results/dpo_method_2_up/checkpoint-1900"
#     "./results/dpo_method_2_up/checkpoint-2000"
#     "./results/dpo_method_1/checkpoint-100"
#     "./results/dpo_method_1/checkpoint-200"
#     "./results/dpo_method_1/checkpoint-300"
#     "./results/dpo_method_1/checkpoint-400"
#     "./results/dpo_method_1/checkpoint-500"
#     "./results/dpo_method_1/checkpoint-600"
#     "./results/dpo_method_1/checkpoint-700"
#     "./results/dpo_method_1/checkpoint-800"
#     "./results/dpo_method_1/checkpoint-900"
#     "./results/dpo_method_1/checkpoint-1000"
#     "./results/dpo_method_1/checkpoint-1100"
#     "./results/dpo_method_1/checkpoint-1200"
#     "./results/dpo_method_1/checkpoint-1300"
#     "./results/dpo_method_1/checkpoint-1400"
#     "./results/dpo_method_1/checkpoint-1500"
#     "./results/dpo_method_1/checkpoint-1600"
#     "./results/dpo_method_1/checkpoint-1700"
#     "./results/dpo_method_1/checkpoint-1800"
#     "./results/dpo_method_1/checkpoint-1900"
#     "./results/dpo_method_1/checkpoint-2000"
# )
# 텍스트 파일 목록
file_list=(
    "./evaluation/T2I-CompBench/examples/dataset/color_val.txt"
    "./evaluation/T2I-CompBench/examples/dataset/shape_val.txt"
    "./evaluation/T2I-CompBench/examples/dataset/texture_val.txt"
    "./evaluation/T2I-CompBench/examples/dataset/spatial_val.txt"
    "./evaluation/T2I-CompBench/examples/dataset/non_spatial_val.txt"
    "./evaluation/T2I-CompBench/examples/dataset/complex_val.txt"
)

CUDA_VISIBLE_DEVICES=0,1 #$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)
NUM_GPUS=2 #$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# 배치 크기와 반복 수
batch_size=10
n_iter=1

# 실행 루프
for ckpt in "${ckpt_list[@]}"; do
    # 체크포인트 이름 추출 (예: checkpoint-350000 -> SDXL_350000)
    if [[ "$ckpt" == "nota-ai/bk-sdm-base" ]]; then
        # 모델 ID인 경우 모델 이름 처리
        ckpt_name="bk-sdm-base"
        ckpt_type="nota"
    else
        # 로컬 체크포인트인 경우 경로에서 이름 추출
        ckpt_name=$(basename "$ckpt")
        # 상위 디렉토리명에서 타입 추출 (예: abl_BK_R_G)
        ckpt_type=$(basename "$(dirname "$ckpt")")
    fi

    for file in "${file_list[@]}"; do
        # 텍스트 파일 이름 추출 (예: color_val.txt -> color)

        dataset_name=$(basename "$file" | sed 's/_val.*//')

        # 출력 디렉토리 설정 (데이터셋 이름 추가)
        outdir="${ckpt}/compbench_img/${dataset_name}"
        mkdir -p "$outdir"

        echo "Running with checkpoint: $ckpt, file: $file, and output directory: $outdir"
        
        
        COMMON_ARGS="--from_file "$file" \
            --ckpt "$ckpt" \
            --batch_size "$batch_size" \
            --n_iter "$n_iter" \
            --outdir "$outdir" \
            --model_id "CompVis/stable-diffusion-v1-4" \
            --cache_dir "./cache" 
        "
        PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

        if [ ${NUM_GPUS} -gt 1 ]; then
            CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT \
            --multi_gpu --num_processes ${NUM_GPUS} evaluation/compbench_generate.py $COMMON_ARGS
        else
            CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch evaluation/compbench_generate.py $COMMON_ARGS
        fi

    done
done

echo "All evaluations completed!"
