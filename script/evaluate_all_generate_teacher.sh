#!/bin/bash

# Set GPU info
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 #$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)
NUM_GPUS=6 #$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# List of result directories to evaluate

RESULT_PATHS=(
    "./results/teacher/Diffusion_KTO"
)

# RESULT_PATHS=(
#     "./results/teacher/Diffusion_DPO-v1-5"
# )
RESULT_PATHS=(
    "./results/teacher/std-v1-5"
)


file_list=(
    "./evaluation/T2I-CompBench/examples/dataset/color_val.txt"
    "./evaluation/T2I-CompBench/examples/dataset/shape_val.txt"
    "./evaluation/T2I-CompBench/examples/dataset/texture_val.txt"
    "./evaluation/T2I-CompBench/examples/dataset/spatial_val.txt"
    "./evaluation/T2I-CompBench/examples/dataset/non_spatial_val.txt"
    "./evaluation/T2I-CompBench/examples/dataset/complex_val.txt"
)

# 배치 크기와 반복 수
batch_size=10
n_iter=1

# 실행 루프
for ckpt in "${RESULT_PATHS[@]}"; do
    # 체크포인트 이름 추출 (예: checkpoint-350000 -> SDXL_350000)
    for file in "${file_list[@]}"; do
        # 텍스트 파일 이름 추출 (예: color_val.txt -> color)

        dataset_name=$(basename "$file" | sed 's/_val.*//')

        # 출력 디렉토리 설정 (데이터셋 이름 추가)
        outdir="${ckpt}/compbench_img/${dataset_name}"
        mkdir -p "$outdir"

        echo "Running with checkpoint: $ckpt, file: $file, and output directory: $outdir"
        
        
        COMMON_ARGS="--from_file "$file" \
            --ckpt "CompVis/stable-diffusion-v1-4" \
            --batch_size "$batch_size" \
            --n_iter "$n_iter" \
            --outdir "$outdir" \
            --model_id "CompVis/stable-diffusion-v1-4" \
            --cache_dir "./cache" \
            --scale 7.5
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


# #     # "./results/stdpo_base_cfg_randcond/checkpoint-100"
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

    PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
    if [ ${NUM_GPUS} -gt 1 ]; then
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
        accelerate launch --multi_gpu --num_processes ${NUM_GPUS} --main_process_port $PORT evaluation/geneval_generate.py $COMMON_ARGS
    else
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
        accelerate launch evaluation/geneval_generate.py $COMMON_ARGS
    fi

done


for ckpt in "${RESULT_PATHS[@]}"; do
    outdir="${ckpt}/HPS_img"
    COMMON_ARGS="\
    --model_id "runwayml/stable-diffusion-v1-5" \
    --ckpt "jacklishufan/diffusion-kto" \
    --outdir "$outdir" \
    --guidance_scale 7.5 \
    --num_inference_steps 25 \
    --batch_size 8
    "
    COMMON_ARGS2="--image_path "$outdir" \
    --hps_version "v2.0" \
    "

    PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
    if [ ${NUM_GPUS} -gt 1 ]; then
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT \
        --multi_gpu --num_processes ${NUM_GPUS} evaluation/HPSv2_generate.py $COMMON_ARGS
    else
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch evaluation/HPSv2_generate.py $COMMON_ARGS
    fi
done

# for ckpt in "${RESULT_PATHS[@]}"; do
#     outdir="${ckpt}/Pratiprompt_img"
#     COMMON_ARGS="\
#     --model_id "runwayml/stable-diffusion-v1-5" \
#     --ckpt "jacklishufan/diffusion-kto" \
#     --outdir "$outdir" \
#     --guidance_scale 7.5 \
#     --num_inference_steps 25 \
#     --batch_size 8
#     "

#     PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
#     if [ ${NUM_GPUS} -gt 1 ]; then
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT \
#         --multi_gpu --num_processes ${NUM_GPUS} evaluation/pickscore_generate.py $COMMON_ARGS
#     else
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch evaluation/pickscore_generate.py $COMMON_ARGS
#     fi
# done

# FID generation
# echo "Running FID generation for all checkpoints"

# batch_size=8

for ckpt in "${RESULT_PATHS[@]}"; do
    outdir="${ckpt}/MSCOCO_img"

    COMMON_ARGS="--ckpt "jacklishufan/diffusion-kto" \
    --batch_size "$batch_size" \
    --outdir "$outdir" \
    --model_id "runwayml/stable-diffusion-v1-5" \
    --cache_dir "./cache" 
    "

    echo "Running with checkpoint: $ckpt, and output directory: $outdir"

    PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
    if [ ${NUM_GPUS} -gt 1 ]; then
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT \
        --multi_gpu --num_processes ${NUM_GPUS} evaluation/FID_generation.py $COMMON_ARGS
    else
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch evaluation/FID_generation.py $COMMON_ARGS
    fi
done


# RESULT_PATHS=(
#     "./results/teacher/Diffusion_DPO-v1-5"
# )

# # FID generation
# echo "Running FID generation for all checkpoints"

# batch_size=8

# for ckpt in "${RESULT_PATHS[@]}"; do
#     outdir="${ckpt}/MSCOCO_img"

#     COMMON_ARGS="--ckpt "mhdang/dpo-sd1.5-text2image-v1" \
#     --batch_size "$batch_size" \
#     --outdir "$outdir" \
#     --model_id "runwayml/stable-diffusion-v1-5" \
#     --cache_dir "./cache" 
#     "

#     echo "Running with checkpoint: $ckpt, and output directory: $outdir"

#     PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
#     if [ ${NUM_GPUS} -gt 1 ]; then
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT \
#         --multi_gpu --num_processes ${NUM_GPUS} evaluation/FID_generation.py $COMMON_ARGS
#     else
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch evaluation/FID_generation.py $COMMON_ARGS
#     fi
# done


# for ckpt in "${RESULT_PATHS[@]}"; do
#     outdir="${ckpt}/HPS_img"
#     COMMON_ARGS="\
#     --model_id "runwayml/stable-diffusion-v1-5" \
#     --ckpt "mhdang/dpo-sd1.5-text2image-v1" \
#     --outdir "$outdir" \
#     --guidance_scale 7.5 \
#     --num_inference_steps 25 \
#     --batch_size 8
#     "
#     COMMON_ARGS2="--image_path "$outdir" \
#     --hps_version "v2.0" \
#     "

#     PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
#     if [ ${NUM_GPUS} -gt 1 ]; then
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT \
#         --multi_gpu --num_processes ${NUM_GPUS} evaluation/HPSv2_generate.py $COMMON_ARGS
#     else
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch evaluation/HPSv2_generate.py $COMMON_ARGS
#     fi
# done

# for ckpt in "${RESULT_PATHS[@]}"; do
#     outdir="${ckpt}/Pratiprompt_img"
#     COMMON_ARGS="\
#     --model_id "runwayml/stable-diffusion-v1-5" \
#     --ckpt "mhdang/dpo-sd1.5-text2image-v1" \
#     --outdir "$outdir" \
#     --guidance_scale 7.5 \
#     --num_inference_steps 25 \
#     --batch_size 8
#     "

#     PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
#     if [ ${NUM_GPUS} -gt 1 ]; then
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT \
#         --multi_gpu --num_processes ${NUM_GPUS} evaluation/pickscore_generate.py $COMMON_ARGS
#     else
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch evaluation/pickscore_generate.py $COMMON_ARGS
#     fi
# done

# RESULT_PATHS=(
#     "./results/teacher/SDXL-1"
# )


# for ckpt in "${RESULT_PATHS[@]}"; do
#     outdir="${ckpt}/HPS_img"
#     COMMON_ARGS="\
#     --model_id "stabilityai/stable-diffusion-xl-base-1.0" \
#     --outdir "$outdir" \
#     --guidance_scale 5.0 \
#     --num_inference_steps 25 \
#     --batch_size 1 \
#     --SDXL \
#     --img_sz 1024 \
#     --cache_dir "./cache"
#     "

#     PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
#     if [ ${NUM_GPUS} -gt 1 ]; then
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT \
#         --multi_gpu --num_processes ${NUM_GPUS} evaluation/HPSv2_generate.py $COMMON_ARGS
#     else
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch evaluation/HPSv2_generate.py $COMMON_ARGS
#     fi
# done

# for ckpt in "${RESULT_PATHS[@]}"; do
#     outdir="${ckpt}/Pratiprompt_img"
#     COMMON_ARGS="\
#     --model_id "stabilityai/stable-diffusion-xl-base-1.0" \
#     --ckpt "stabilityai/stable-diffusion-xl-base-1.0" \
#     --outdir "$outdir" \
#     --guidance_scale 5.0 \
#     --num_inference_steps 25 \
#     --batch_size 4 \
#     --SDXL \
#     --img_sz 1024
#     "

#     PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
#     if [ ${NUM_GPUS} -gt 1 ]; then
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT \
#         --multi_gpu --num_processes ${NUM_GPUS} evaluation/pickscore_generate.py $COMMON_ARGS
#     else
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch evaluation/pickscore_generate.py $COMMON_ARGS
#     fi
# done

# file_list=(
#     "./evaluation/T2I-CompBench/examples/dataset/color_val.txt"
#     "./evaluation/T2I-CompBench/examples/dataset/shape_val.txt"
#     "./evaluation/T2I-CompBench/examples/dataset/texture_val.txt"
#     "./evaluation/T2I-CompBench/examples/dataset/spatial_val.txt"
#     "./evaluation/T2I-CompBench/examples/dataset/non_spatial_val.txt"
#     "./evaluation/T2I-CompBench/examples/dataset/complex_val.txt"
# )

# # 배치 크기와 반복 수
# batch_size=10
# n_iter=1

# # 실행 루프
# for ckpt in "${RESULT_PATHS[@]}"; do
#     # 체크포인트 이름 추출 (예: checkpoint-350000 -> SDXL_350000)
#     for file in "${file_list[@]}"; do
#         # 텍스트 파일 이름 추출 (예: color_val.txt -> color)

#         dataset_name=$(basename "$file" | sed 's/_val.*//')

#         # 출력 디렉토리 설정 (데이터셋 이름 추가)
#         outdir="${ckpt}/compbench_img/${dataset_name}"
#         mkdir -p "$outdir"

#         echo "Running with checkpoint: $ckpt, file: $file, and output directory: $outdir"
        
        
#         COMMON_ARGS="--from_file "$file" \
#             --ckpt "stabilityai/stable-diffusion-xl-base-1.0" \
#             --batch_size "$batch_size" \
#             --n_iter "$n_iter" \
#             --outdir "$outdir" \
#             --model_id "stabilityai/stable-diffusion-xl-base-1.0" \
#             --cache_dir "./cache" \
#             --SDXL \
#             --scale 5.0 \
#             --img_sz 1024
#         "
#         PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

#         if [ ${NUM_GPUS} -gt 1 ]; then
#             CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT \
#             --multi_gpu --num_processes ${NUM_GPUS} evaluation/compbench_generate.py $COMMON_ARGS
#         else
#             CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch evaluation/compbench_generate.py $COMMON_ARGS
#         fi

#     done
# done

# # FID generation
# echo "Running FID generation for all checkpoints"

# batch_size=16

# for ckpt in "${RESULT_PATHS[@]}"; do
#     outdir="${ckpt}/MSCOCO_img"

#     COMMON_ARGS="--ckpt "stabilityai/stable-diffusion-xl-base-1.0" \
#     --batch_size "$batch_size" \
#     --outdir "$outdir" \
#     --model_id "stabilityai/stable-diffusion-xl-base-1.0" \
#     --cache_dir "./cache"  \
#     --SDXL \
#     --scale 5.0 \
#     --img_sz 1024
#     "

#     echo "Running with checkpoint: $ckpt, and output directory: $outdir"

#     PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
#     if [ ${NUM_GPUS} -gt 1 ]; then
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT \
#         --multi_gpu --num_processes ${NUM_GPUS} evaluation/FID_generation.py $COMMON_ARGS
#     else
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch evaluation/FID_generation.py $COMMON_ARGS
#     fi
# done


# RESULT_PATHS=(
#     "./results/SANA_600M_1024"
# )


# for RESULT_DIR in "${RESULT_PATHS[@]}"; do
#     OUTDIR="${RESULT_DIR}/geneval_img"
#     OUTFILE="${RESULT_DIR}/geneval_results.jsonl"
#     UNET_PATH="${RESULT_DIR}/unet"
#     METADATA="./evaluation/geneval/prompts/evaluation_metadata.jsonl"
#     DETECTOR="./evaluation/geneval/OBJECT_DETECTOR_FOLDER"

#     COMMON_ARGS="$METADATA \
#     --model "Efficient-Large-Model/Sana_600M_1024px_diffusers" \
#     --outdir "$OUTDIR" \
#     --batch_size 4 \
#     --scale 4.5 \
#     --steps 20 \
#     --cache_dir "./cache"\
#     --SANA \
#     --img_sz 1024
#     "

#     echo "🔁 Running generation and evaluation for: $RESULT_DIR"

#     PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
#     if [ ${NUM_GPUS} -gt 1 ]; then
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
#         accelerate launch --multi_gpu --num_processes ${NUM_GPUS} --main_process_port $PORT evaluation/geneval_generate.py $COMMON_ARGS
#     else
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
#         accelerate launch evaluation/geneval_generate.py $COMMON_ARGS
#     fi

# done


# for ckpt in "${RESULT_PATHS[@]}"; do
#     outdir="${ckpt}/HPS_img"
#     COMMON_ARGS="\
#     --model_id "Efficient-Large-Model/Sana_600M_1024px_diffusers" \
#     --outdir "$outdir" \
#     --guidance_scale 4.5 \
#     --num_inference_steps 20 \
#     --batch_size 8 \
#     --SANA \
#     --img_sz 1024 \
#     --cache_dir "./cache"
#     "

#     PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
#     if [ ${NUM_GPUS} -gt 1 ]; then
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT \
#         --multi_gpu --num_processes ${NUM_GPUS} evaluation/HPSv2_generate.py $COMMON_ARGS
#     else
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch evaluation/HPSv2_generate.py $COMMON_ARGS
#     fi
# done



# file_list=(
#     "./evaluation/T2I-CompBench/examples/dataset/color_val.txt"
#     "./evaluation/T2I-CompBench/examples/dataset/shape_val.txt"
#     "./evaluation/T2I-CompBench/examples/dataset/texture_val.txt"
#     "./evaluation/T2I-CompBench/examples/dataset/spatial_val.txt"
#     "./evaluation/T2I-CompBench/examples/dataset/non_spatial_val.txt"
#     "./evaluation/T2I-CompBench/examples/dataset/complex_val.txt"
# )

# # 배치 크기와 반복 수
# batch_size=10
# n_iter=1

# # 실행 루프
# for ckpt in "${RESULT_PATHS[@]}"; do
#     for file in "${file_list[@]}"; do
#         # 텍스트 파일 이름 추출 (예: color_val.txt -> color)

#         dataset_name=$(basename "$file" | sed 's/_val.*//')

#         # 출력 디렉토리 설정 (데이터셋 이름 추가)
#         outdir="${ckpt}/compbench_img/${dataset_name}"
#         mkdir -p "$outdir"

#         echo "Running with checkpoint: $ckpt, file: $file, and output directory: $outdir"
        
        
#         COMMON_ARGS="--from_file "$file" \
#             --batch_size "$batch_size" \
#             --n_iter "$n_iter" \
#             --outdir "$outdir" \
#             --model_id "Efficient-Large-Model/Sana_600M_1024px_diffusers" \
#             --cache_dir "./cache" \
#             --SANA \
#             --scale 4.5 \
#             --num_inference_steps 20 \
#             --img_sz 1024
#             "
#         PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

#         if [ ${NUM_GPUS} -gt 1 ]; then
#             CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT \
#             --multi_gpu --num_processes ${NUM_GPUS} evaluation/compbench_generate.py $COMMON_ARGS
#         else
#             CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch evaluation/compbench_generate.py $COMMON_ARGS
#         fi

#     done
# done


# RESULT_PATHS=(
#     "./results/IterComp"
# )

# for RESULT_DIR in "${RESULT_PATHS[@]}"; do
#     OUTDIR="${RESULT_DIR}/geneval_img"
#     OUTFILE="${RESULT_DIR}/geneval_results.jsonl"
#     UNET_PATH="${RESULT_DIR}/unet"
#     METADATA="./evaluation/geneval/prompts/evaluation_metadata.jsonl"
#     DETECTOR="./evaluation/geneval/OBJECT_DETECTOR_FOLDER"

#     COMMON_ARGS="$METADATA \
#     --model "comin/IterComp" \
#     --outdir "$OUTDIR" \
#     --batch_size 4 \
#     --scale 5.0 \
#     --steps 25 \
#     --cache_dir "./cache"\
#     --SDXL \
#     --itercomp \
#     --img_sz 1024
#     "

#     echo "🔁 Running generation and evaluation for: $RESULT_DIR"

#     PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
#     if [ ${NUM_GPUS} -gt 1 ]; then
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
#         accelerate launch --multi_gpu --num_processes ${NUM_GPUS} --main_process_port $PORT evaluation/geneval_generate.py $COMMON_ARGS
#     else
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
#         accelerate launch evaluation/geneval_generate.py $COMMON_ARGS
#     fi

# done

# for ckpt in "${RESULT_PATHS[@]}"; do
#     outdir="${ckpt}/HPS_img"
#     COMMON_ARGS="\
#     --model_id "comin/IterComp" \
#     --outdir "$outdir" \
#     --guidance_scale 5.0 \
#     --num_inference_steps 25 \
#     --batch_size 8 \
#     --SDXL \
#     --itercomp \
#     --img_sz 1024
#     "
#     COMMON_ARGS2="--image_path "$outdir" \
#     --hps_version "v2.0" \
#     "

#     PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
#     if [ ${NUM_GPUS} -gt 1 ]; then
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT \
#         --multi_gpu --num_processes ${NUM_GPUS} evaluation/HPSv2_generate.py $COMMON_ARGS
#     else
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch evaluation/HPSv2_generate.py $COMMON_ARGS
#     fi
# done


# file_list=(
#     "./evaluation/T2I-CompBench/examples/dataset/color_val.txt"
#     "./evaluation/T2I-CompBench/examples/dataset/shape_val.txt"
#     "./evaluation/T2I-CompBench/examples/dataset/texture_val.txt"
#     "./evaluation/T2I-CompBench/examples/dataset/spatial_val.txt"
#     "./evaluation/T2I-CompBench/examples/dataset/non_spatial_val.txt"
#     "./evaluation/T2I-CompBench/examples/dataset/complex_val.txt"
# )

# # 배치 크기와 반복 수
# batch_size=10
# n_iter=1

# # 실행 루프
# for ckpt in "${RESULT_PATHS[@]}"; do
#     # 체크포인트 이름 추출 (예: checkpoint-350000 -> SDXL_350000)
#     for file in "${file_list[@]}"; do
#         # 텍스트 파일 이름 추출 (예: color_val.txt -> color)

#         dataset_name=$(basename "$file" | sed 's/_val.*//')

#         # 출력 디렉토리 설정 (데이터셋 이름 추가)
#         outdir="${ckpt}/compbench_img/${dataset_name}"
#         mkdir -p "$outdir"

#         echo "Running with checkpoint: $ckpt, file: $file, and output directory: $outdir"
        
        
#         COMMON_ARGS="--from_file "$file" \
#             --batch_size "$batch_size" \
#             --n_iter "$n_iter" \
#             --outdir "$outdir" \
#             --model_id "comin/IterComp" \
#             --cache_dir "./cache" \
#             --SDXL \
#             --itercomp \
#             --scale 5.0 \
#             --img_sz 1024
#         "
#         PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

#         if [ ${NUM_GPUS} -gt 1 ]; then
#             CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT \
#             --multi_gpu --num_processes ${NUM_GPUS} evaluation/compbench_generate.py $COMMON_ARGS
#         else
#             CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch evaluation/compbench_generate.py $COMMON_ARGS
#         fi

#     done
# done

# for ckpt in "${RESULT_PATHS[@]}"; do
#     outdir="${ckpt}/Pratiprompt_img"
#     COMMON_ARGS="\
#     --model_id "comin/IterComp" \
#     --outdir "$outdir" \
#     --guidance_scale 5.0 \
#     --num_inference_steps 25 \
#     --batch_size 4 \
#     --SDXL \
#     --itercomp \
#     --img_sz 1024
#     "

#     PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
#     if [ ${NUM_GPUS} -gt 1 ]; then
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT \
#         --multi_gpu --num_processes ${NUM_GPUS} evaluation/pickscore_generate.py $COMMON_ARGS
#     else
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch evaluation/pickscore_generate.py $COMMON_ARGS
#     fi
# done



# RESULT_PATHS=(
#     "./results/Diffusion_DPO"
# )

# for ckpt in "${RESULT_PATHS[@]}"; do
#     outdir="${ckpt}/HPS_img"
#     COMMON_ARGS="\
#     --model_id "stabilityai/stable-diffusion-xl-base-1.0" \
#     --ckpt "mhdang/dpo-sdxl-text2image-v1" \
#     --outdir "$outdir" \
#     --guidance_scale 5.0 \
#     --num_inference_steps 25 \
#     --batch_size 8 \
#     --SDXL \
#     --img_sz 1024
#     "
#     COMMON_ARGS2="--image_path "$outdir" \
#     --hps_version "v2.0" \
#     "

#     PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
#     if [ ${NUM_GPUS} -gt 1 ]; then
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT \
#         --multi_gpu --num_processes ${NUM_GPUS} evaluation/HPSv2_generate.py $COMMON_ARGS
#     else
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch evaluation/HPSv2_generate.py $COMMON_ARGS
#     fi
# done

# for ckpt in "${RESULT_PATHS[@]}"; do
#     outdir="${ckpt}/Pratiprompt_img"
#     COMMON_ARGS="\
#     --model_id "stabilityai/stable-diffusion-xl-base-1.0" \
#     --ckpt "mhdang/dpo-sdxl-text2image-v1" \
#     --outdir "$outdir" \
#     --guidance_scale 5.0 \
#     --num_inference_steps 25 \
#     --batch_size 8 \
#     --SDXL \
#     --img_sz 1024
#     "

#     PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
#     if [ ${NUM_GPUS} -gt 1 ]; then
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT \
#         --multi_gpu --num_processes ${NUM_GPUS} evaluation/pickscore_generate.py $COMMON_ARGS
#     else
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch evaluation/pickscore_generate.py $COMMON_ARGS
#     fi
# done


# # FID generation
# echo "Running FID generation for all checkpoints"

# batch_size=16

# for ckpt in "${RESULT_PATHS[@]}"; do
#     outdir="${ckpt}/MSCOCO_img"

#     COMMON_ARGS="--ckpt "mhdang/dpo-sdxl-text2image-v1" \
#     --batch_size "$batch_size" \
#     --outdir "$outdir" \
#     --model_id "stabilityai/stable-diffusion-xl-base-1.0" \
#     --cache_dir "./cache"  \
#     --SDXL \
#     --scale 5.0 \
#     --img_sz 1024
#     "

#     echo "Running with checkpoint: $ckpt, and output directory: $outdir"

#     PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
#     if [ ${NUM_GPUS} -gt 1 ]; then
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT \
#         --multi_gpu --num_processes ${NUM_GPUS} evaluation/FID_generation.py $COMMON_ARGS
#     else
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch evaluation/FID_generation.py $COMMON_ARGS
#     fi
# done


# file_list=(
#     "./evaluation/T2I-CompBench/examples/dataset/color_val.txt"
#     "./evaluation/T2I-CompBench/examples/dataset/shape_val.txt"
#     "./evaluation/T2I-CompBench/examples/dataset/texture_val.txt"
#     "./evaluation/T2I-CompBench/examples/dataset/spatial_val.txt"
#     "./evaluation/T2I-CompBench/examples/dataset/non_spatial_val.txt"
#     "./evaluation/T2I-CompBench/examples/dataset/complex_val.txt"
# )

# # 배치 크기와 반복 수
# batch_size=10
# n_iter=1

# # 실행 루프
# for ckpt in "${RESULT_PATHS[@]}"; do
#     # 체크포인트 이름 추출 (예: checkpoint-350000 -> SDXL_350000)
#     for file in "${file_list[@]}"; do
#         # 텍스트 파일 이름 추출 (예: color_val.txt -> color)

#         dataset_name=$(basename "$file" | sed 's/_val.*//')

#         # 출력 디렉토리 설정 (데이터셋 이름 추가)
#         outdir="${ckpt}/compbench_img/${dataset_name}"
#         mkdir -p "$outdir"

#         echo "Running with checkpoint: $ckpt, file: $file, and output directory: $outdir"
        
        
#         COMMON_ARGS="--from_file "$file" \
#             --ckpt "mhdang/dpo-sdxl-text2image-v1" \
#             --batch_size "$batch_size" \
#             --n_iter "$n_iter" \
#             --outdir "$outdir" \
#             --model_id "stabilityai/stable-diffusion-xl-base-1.0" \
#             --cache_dir "./cache" \
#             --SDXL \
#             --scale 5.0 \
#             --img_sz 1024
#         "
#         PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

#         if [ ${NUM_GPUS} -gt 1 ]; then
#             CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT \
#             --multi_gpu --num_processes ${NUM_GPUS} evaluation/compbench_generate.py $COMMON_ARGS
#         else
#             CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch evaluation/compbench_generate.py $COMMON_ARGS
#         fi

#     done
# done


# for RESULT_DIR in "${RESULT_PATHS[@]}"; do
#     OUTDIR="${RESULT_DIR}/geneval_img"
#     OUTFILE="${RESULT_DIR}/geneval_results.jsonl"
#     UNET_PATH="${RESULT_DIR}/unet"
#     METADATA="./evaluation/geneval/prompts/evaluation_metadata.jsonl"
#     DETECTOR="./evaluation/geneval/OBJECT_DETECTOR_FOLDER"

#     COMMON_ARGS="$METADATA \
#     --model "stabilityai/stable-diffusion-xl-base-1.0" \
#     --unet_path "mhdang/dpo-sdxl-text2image-v1/unet" \
#     --outdir "$OUTDIR" \
#     --batch_size 4 \
#     --scale 5.0 \
#     --steps 25 \
#     --cache_dir "./cache"\
#     --SDXL \
#     --img_sz 1024
#     "

#     echo "🔁 Running generation and evaluation for: $RESULT_DIR"

#     PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
#     if [ ${NUM_GPUS} -gt 1 ]; then
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
#         accelerate launch --multi_gpu --num_processes ${NUM_GPUS} --main_process_port $PORT evaluation/geneval_generate.py $COMMON_ARGS
#     else
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
#         accelerate launch evaluation/geneval_generate.py $COMMON_ARGS
#     fi

# done