#!/bin/bash
# GenEval + CompBench generation for SD1.4 with random-removal negative-prompt CFG.
# Each sample's negative prompt is built by randomly dropping a fixed fraction
# of the words in the (positive) prompt — same recipe as
# prompts_generation/DiffusionDB_random_removal.py.
#
# Sweeps over (ratio x cfg_scale). For each (ratio, scale) the negative-prompt
# drop fraction is fixed (LOW=HIGH=ratio), so each condition is deterministic.
# Order: ALL GenEval runs first, then ALL CompBench runs.

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NUM_GPUS=${NUM_GPUS:-4}
batch_size=${BATCH_SIZE:-10}

MODEL_ID="CompVis/stable-diffusion-v1-4"
STEPS=${STEPS:-50}
NEG_SEED=${NEG_SEED:-42}

# Sweep grid.
RATIOS=(${RATIOS:-0.25 0.5 0.75})
SCALES=(${SCALES:-5.0 7.5})

# Output paths chosen so they fit existing scoring patterns.
OUTROOT=${OUTROOT:-./results/sd14_neg_prompt_cfg/random_removal}

METADATA="./evaluation/geneval/prompts/evaluation_metadata.jsonl"
COMPBENCH_FILES=(
    "./evaluation/T2I-CompBench/examples/dataset/color_val.txt"
    "./evaluation/T2I-CompBench/examples/dataset/shape_val.txt"
    "./evaluation/T2I-CompBench/examples/dataset/texture_val.txt"
    "./evaluation/T2I-CompBench/examples/dataset/spatial_val.txt"
    "./evaluation/T2I-CompBench/examples/dataset/non_spatial_val.txt"
    "./evaluation/T2I-CompBench/examples/dataset/complex_val.txt"
)

mkdir -p "$OUTROOT"

run_accelerate() {
    local script=$1
    shift
    local args="$@"
    local PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
    if [ ${NUM_GPUS} -gt 1 ]; then
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT \
        --multi_gpu --num_processes ${NUM_GPUS} "$script" $args
    else
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch "$script" $args
    fi
}

# scale=5.0 -> "5p0"; ratio=0.25 -> "0p25"
fmt() { echo "$1" | sed 's/\./p/g'; }

cond_outdir() {
    local scale=$1
    local ratio=$2
    echo "${OUTROOT}/scale$(fmt "$scale")_ratio$(fmt "$ratio")"
}

echo "############################################"
echo "  SD1.4 random-removal negative-prompt CFG sweep"
echo "  steps=$STEPS  scales=(${SCALES[*]})  ratios=(${RATIOS[*]})"
echo "  out: $OUTROOT"
echo "############################################"

# =====================================================================
# Phase 1: GenEval generation for ALL (ratio, scale)
# =====================================================================
echo "########################################"
echo "  Phase 1: GenEval generation (all conditions)"
echo "########################################"
for SCALE in "${SCALES[@]}"; do
    for RATIO in "${RATIOS[@]}"; do
        outdir=$(cond_outdir "$SCALE" "$RATIO")
        geneval_outdir="${outdir}/geneval_img"
        mkdir -p "$geneval_outdir"

        echo "----------------------------------------"
        echo "  GenEval | scale=$SCALE  ratio=$RATIO"
        echo "  out: $geneval_outdir"
        echo "----------------------------------------"
        run_accelerate evaluation/geneval_generate.py \
            "$METADATA" \
            --model "$MODEL_ID" \
            --outdir "$geneval_outdir" \
            --batch_size 4 \
            --scale "$SCALE" \
            --steps "$STEPS" \
            --neg_prompt_random_removal \
            --neg_prompt_seed "$NEG_SEED" \
            --neg_prompt_ratio_low "$RATIO" \
            --neg_prompt_ratio_high "$RATIO" \
            --cache_dir "./cache"
    done
done

# =====================================================================
# Phase 2: CompBench generation for ALL (ratio, scale)
# =====================================================================
echo "########################################"
echo "  Phase 2: CompBench generation (all conditions)"
echo "########################################"
for SCALE in "${SCALES[@]}"; do
    for RATIO in "${RATIOS[@]}"; do
        outdir=$(cond_outdir "$SCALE" "$RATIO")
        compbench_base="${outdir}/compbench_img"
        mkdir -p "$compbench_base"

        echo "----------------------------------------"
        echo "  CompBench | scale=$SCALE  ratio=$RATIO"
        echo "  out: $compbench_base"
        echo "----------------------------------------"
        for file in "${COMPBENCH_FILES[@]}"; do
            dataset_name=$(basename "$file" | sed 's/_val.*//')
            ds_outdir="${compbench_base}/${dataset_name}"
            mkdir -p "$ds_outdir"

            echo "    -> $dataset_name"
            run_accelerate evaluation/compbench_generate.py \
                --from_file "$file" \
                --batch_size $batch_size \
                --n_iter 1 \
                --outdir "$ds_outdir" \
                --model_id "$MODEL_ID" \
                --scale "$SCALE" \
                --num_inference_steps "$STEPS" \
                --neg_prompt_random_removal \
                --neg_prompt_seed "$NEG_SEED" \
                --neg_prompt_ratio_low "$RATIO" \
                --neg_prompt_ratio_high "$RATIO" \
                --cache_dir "./cache"
        done
    done
done

echo "########################################"
echo "  DONE: $OUTROOT"
echo "########################################"
