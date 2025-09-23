#!/bin/bash

# GPU 설정
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
NUM_GPUS=6

# 사용할 체크포인트 디렉토리 리스트
RESULT_PATHS=(
    "./results/SDXL/our_beta16k_sdxl/checkpoint-100"
)

# JSONL 프롬프트 경로
PROMPT_JSONL="./data/captions/pickapic/metadata_rank0.jsonl"

# 공통 설정
BATCH_SIZE=8
MODEL_ID="stabilityai/stable-diffusion-xl-base-1.0"
IMG_SIZE=1024  # 필요시 512로 조정
GUIDANCE=5
STEPS=25

# 각 ckpt 디렉토리에 대해 반복
for CKPT_PATH in "${RESULT_PATHS[@]}"; do
    OUTDIR="${CKPT_PATH}/qualitative_pickapic_img"

    echo "🔁 Generating images with checkpoint: $CKPT_PATH"
    echo "📂 Output directory: $OUTDIR"

    # 포트 무작위 선택 (충돌 방지)
    PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

    # argument 문자열 구성
    COMMON_ARGS="\
    --from_jsonl ${PROMPT_JSONL} \
    --ckpt ${CKPT_PATH} \
    --outdir ${OUTDIR} \
    --batch_size ${BATCH_SIZE} \
    --model_id ${MODEL_ID} \
    --guidance_scale ${GUIDANCE} \
    --num_inference_steps ${STEPS} \
    --img_sz ${IMG_SIZE} \
    --SDXL \
    "

    if [ ${NUM_GPUS} -gt 1 ]; then
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --multi_gpu --num_processes ${NUM_GPUS} --main_process_port ${PORT} \
        evaluation/qualitative_generate.py ${COMMON_ARGS}
    else
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch evaluation/qualitative_generate.py ${COMMON_ARGS}
    fi
done

########################################################################
# 사용할 체크포인트 디렉토리 리스트
RESULT_PATHS=(
    "./results/teacher/SDXL-1"
)

# JSONL 프롬프트 경로
PROMPT_JSONL="./data/captions/pickapic/metadata_rank0.jsonl"

# 공통 설정
BATCH_SIZE=8
MODEL_ID="stabilityai/stable-diffusion-xl-base-1.0"
IMG_SIZE=1024  # 필요시 512로 조정
GUIDANCE=5
STEPS=25

# 각 ckpt 디렉토리에 대해 반복
for CKPT_PATH in "${RESULT_PATHS[@]}"; do
    OUTDIR="${CKPT_PATH}/qualitative_pickapic_img"

    echo "🔁 Generating images with checkpoint: $CKPT_PATH"
    echo "📂 Output directory: $OUTDIR"

    # 포트 무작위 선택 (충돌 방지)
    PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

    # argument 문자열 구성
    COMMON_ARGS="\
    --from_jsonl ${PROMPT_JSONL} \
    --ckpt "stabilityai/stable-diffusion-xl-base-1.0" \
    --outdir ${OUTDIR} \
    --batch_size ${BATCH_SIZE} \
    --model_id ${MODEL_ID} \
    --guidance_scale ${GUIDANCE} \
    --num_inference_steps ${STEPS} \
    --img_sz ${IMG_SIZE} \
    --SDXL \
    "

    if [ ${NUM_GPUS} -gt 1 ]; then
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --multi_gpu --num_processes ${NUM_GPUS} --main_process_port ${PORT} \
        evaluation/qualitative_generate.py ${COMMON_ARGS}
    else
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch evaluation/qualitative_generate.py ${COMMON_ARGS}
    fi
done

###########################################################

# 사용할 체크포인트 디렉토리 리스트
RESULT_PATHS=(
    "./results/final_ours_sd14_align/align_dpo_beta8k/checkpoint-100"
    "./results/final_ours_sd14_align/align_our_beta16k/checkpoint-100"
)

# JSONL 프롬프트 경로
PROMPT_JSONL="./data/captions/pickapic/metadata_rank0.jsonl"

# 공통 설정
BATCH_SIZE=8
MODEL_ID="CompVis/stable-diffusion-v1-4"
IMG_SIZE=512  # 필요시 512로 조정
GUIDANCE=7.5
STEPS=25

# 각 ckpt 디렉토리에 대해 반복
for CKPT_PATH in "${RESULT_PATHS[@]}"; do
    OUTDIR="${CKPT_PATH}/qualitative_pickapic_img"

    echo "🔁 Generating images with checkpoint: $CKPT_PATH"
    echo "📂 Output directory: $OUTDIR"

    # 포트 무작위 선택 (충돌 방지)
    PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

    # argument 문자열 구성
    COMMON_ARGS="\
    --from_jsonl ${PROMPT_JSONL} \
    --ckpt ${CKPT_PATH} \
    --outdir ${OUTDIR} \
    --batch_size ${BATCH_SIZE} \
    --model_id ${MODEL_ID} \
    --guidance_scale ${GUIDANCE} \
    --num_inference_steps ${STEPS} \
    --img_sz ${IMG_SIZE}
    "

    if [ ${NUM_GPUS} -gt 1 ]; then
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --multi_gpu --num_processes ${NUM_GPUS} --main_process_port ${PORT} \
        evaluation/qualitative_generate.py ${COMMON_ARGS}
    else
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch evaluation/qualitative_generate.py ${COMMON_ARGS}
    fi
done


###########################################################

# 사용할 체크포인트 디렉토리 리스트
RESULT_PATHS=(
    "./results/teacher/std-v1-4"
)

# JSONL 프롬프트 경로
PROMPT_JSONL="./data/captions/pickapic/metadata_rank0.jsonl"

# 공통 설정
BATCH_SIZE=8
MODEL_ID="CompVis/stable-diffusion-v1-4"
IMG_SIZE=512  # 필요시 512로 조정
GUIDANCE=7.5
STEPS=25

# 각 ckpt 디렉토리에 대해 반복
for CKPT_PATH in "${RESULT_PATHS[@]}"; do
    OUTDIR="${CKPT_PATH}/qualitative_pickapic_img"

    echo "🔁 Generating images with checkpoint: $CKPT_PATH"
    echo "📂 Output directory: $OUTDIR"

    # 포트 무작위 선택 (충돌 방지)
    PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

    # argument 문자열 구성
    COMMON_ARGS="\
    --from_jsonl ${PROMPT_JSONL} \
    --ckpt "CompVis/stable-diffusion-v1-4" \
    --outdir ${OUTDIR} \
    --batch_size ${BATCH_SIZE} \
    --model_id ${MODEL_ID} \
    --guidance_scale ${GUIDANCE} \
    --num_inference_steps ${STEPS} \
    --img_sz ${IMG_SIZE}
    "

    if [ ${NUM_GPUS} -gt 1 ]; then
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --multi_gpu --num_processes ${NUM_GPUS} --main_process_port ${PORT} \
        evaluation/qualitative_generate.py ${COMMON_ARGS}
    else
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch evaluation/qualitative_generate.py ${COMMON_ARGS}
    fi
done


######################################################
# 사용할 체크포인트 디렉토리 리스트
RESULT_PATHS=(
    "./results/SANA/ours_beta2k_lora512/checkpoint-100"
)

# JSONL 프롬프트 경로
PROMPT_JSONL="./data/captions/pickapic/metadata_rank0.jsonl"

# 공통 설정
BATCH_SIZE=8
MODEL_ID="Efficient-Large-Model/Sana_600M_1024px_diffusers"
IMG_SIZE=1024  # 필요시 512로 조정
GUIDANCE=4.5
STEPS=20

# 각 ckpt 디렉토리에 대해 반복
for CKPT_PATH in "${RESULT_PATHS[@]}"; do
    OUTDIR="${CKPT_PATH}/qualitative_pickapic_img"

    echo "🔁 Generating images with checkpoint: $CKPT_PATH"
    echo "📂 Output directory: $OUTDIR"

    # 포트 무작위 선택 (충돌 방지)
    PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

    # argument 문자열 구성
    COMMON_ARGS="\
    --from_jsonl ${PROMPT_JSONL} \
    --ckpt ${CKPT_PATH} \
    --outdir ${OUTDIR} \
    --batch_size ${BATCH_SIZE} \
    --model_id ${MODEL_ID} \
    --guidance_scale ${GUIDANCE} \
    --num_inference_steps ${STEPS} \
    --img_sz ${IMG_SIZE} \
    --SANA \
    "

    if [ ${NUM_GPUS} -gt 1 ]; then
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --multi_gpu --num_processes ${NUM_GPUS} --main_process_port ${PORT} \
        evaluation/qualitative_generate.py ${COMMON_ARGS}
    else
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch evaluation/qualitative_generate.py ${COMMON_ARGS}
    fi
done


######################################################
# 사용할 체크포인트 디렉토리 리스트
RESULT_PATHS=(
    "./results/teacher/SANA_600M_1024"
)

# JSONL 프롬프트 경로
PROMPT_JSONL="./data/captions/pickapic/metadata_rank0.jsonl"

# 공통 설정
BATCH_SIZE=8
MODEL_ID="Efficient-Large-Model/Sana_600M_1024px_diffusers"
IMG_SIZE=1024  # 필요시 512로 조정
GUIDANCE=4.5
STEPS=20

# 각 ckpt 디렉토리에 대해 반복
for CKPT_PATH in "${RESULT_PATHS[@]}"; do
    OUTDIR="${CKPT_PATH}/qualitative_pickapic_img"

    echo "🔁 Generating images with checkpoint: $CKPT_PATH"
    echo "📂 Output directory: $OUTDIR"

    # 포트 무작위 선택 (충돌 방지)
    PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

    # argument 문자열 구성
    COMMON_ARGS="\
    --from_jsonl ${PROMPT_JSONL} \
    --outdir ${OUTDIR} \
    --batch_size ${BATCH_SIZE} \
    --model_id ${MODEL_ID} \
    --guidance_scale ${GUIDANCE} \
    --num_inference_steps ${STEPS} \
    --img_sz ${IMG_SIZE} \
    --SANA \
    "

    if [ ${NUM_GPUS} -gt 1 ]; then
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --multi_gpu --num_processes ${NUM_GPUS} --main_process_port ${PORT} \
        evaluation/qualitative_generate.py ${COMMON_ARGS}
    else
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch evaluation/qualitative_generate.py ${COMMON_ARGS}
    fi
done
