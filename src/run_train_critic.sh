#!/bin/bash
set -e

echo "=== CoRM-RAG Critic Training ==="
echo "Node: ${RANK:-0} / ${WORLD_SIZE:-1}"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
echo "=== Host memory ==="
free -h || true
echo "=== /tmp mount ==="
df -hT /tmp || true
echo "==================="

export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
mkdir -p "$HF_HOME"

# ─────────────────────────────────────────────────────────
# Stage data + model from source storage to node-local disk ONCE per node.
# This script runs once per node (process_count_per_node=1), then
# torchrun fans out into NGPUS ranks that all read from local SSD.
# ─────────────────────────────────────────────────────────
LOCAL_STAGE_DIR="${LOCAL_STAGE_DIR:-/tmp/critic_stage}"
mkdir -p "$LOCAL_STAGE_DIR"

stage_dir () {
    # $1 = src dir, $2 = dst dir on local
    local SRC="$1"; local DST="$2"
    mkdir -p "$DST"
    local SENTINEL="$DST/.STAGE_DONE"
    if [ -f "$SENTINEL" ]; then
        echo "[stage] $DST already staged ($(ls "$DST" | wc -l) files), skipping"
        return
    fi
    local N=$(find "$SRC" -maxdepth 1 -type f | wc -l)
    echo "[stage] copying $N files from $SRC -> $DST"
    local T0=$(date +%s)
    # Parallel cp (8 streams) with progress.
    find "$SRC" -maxdepth 1 -type f -printf '%f\n' \
        | xargs -P 8 -I{} sh -c 'cp "'"$SRC"'/$1" "'"$DST"'/$1" && echo "$1"' _ {} \
        | awk -v total="$N" -v t0="$T0" '
            { n++
              if (n % 10 == 0 || n == total) {
                t = systime() - t0
                printf("[stage]  %d/%d files copied (%.0fs, %.1fMB/s)\n", n, total, t, (n*210)/(t>0?t:1)) > "/dev/stderr"
              } }'
    local T1=$(date +%s)
    local SIZE=$(du -sm "$DST" | awk '{print $1}')
    echo "[stage] done in $((T1-T0))s, ${SIZE}MB"
    touch "$SENTINEL"
}

DATA_SRC="${DATA_SRC:-./data}"
TOKENIZED_DIR=""
TOK_SUBDIR="${CRITIC_TOKENIZED_SUBDIR:-critic_tokenized}"
for prefix in \
    "$DATA_SRC/training/$TOK_SUBDIR" \
    "$DATA_SRC/training/critic_tokenized"; do
    if compgen -G "$prefix/shard-*.parquet" > /dev/null; then
        TOKENIZED_DIR="$prefix"
        N=$(ls "$prefix"/shard-*.parquet | wc -l)
        echo "Found pre-tokenized data at $prefix ($N shards)"
        break
    fi
done

DATA_ARG=""
if [ -n "$TOKENIZED_DIR" ]; then
    if [ "${CRITIC_SKIP_STAGE:-0}" = "1" ]; then
        echo "[stage] CRITIC_SKIP_STAGE=1, reading tokenized dir directly: $TOKENIZED_DIR"
        LOCAL_TOK="$TOKENIZED_DIR"
    else
        LOCAL_TOK="$LOCAL_STAGE_DIR/critic_tokenized"
        stage_dir "$TOKENIZED_DIR" "$LOCAL_TOK"
    fi
    DATA_ARG="--tokenized_dir $LOCAL_TOK"
else
    DATA_FILE=""
    INDEX_FILE=""
    for prefix in "$DATA_SRC/training"; do
        if [ -f "$prefix/train_groups.jsonl" ] && [ -f "$prefix/train_index.json" ]; then
            DATA_FILE="$prefix/train_groups.jsonl"
            INDEX_FILE="$prefix/train_index.json"
            echo "Found training data at $prefix"
            break
        fi
    done
    if [ -z "$DATA_FILE" ]; then
        echo "ERROR: neither tokenized_dir nor train_groups.jsonl found!"
        exit 1
    fi
    DATA_ARG="--data $DATA_FILE --index $INDEX_FILE"
fi

# Stage model (~1.7GB) to local disk so all ranks load from SSD.
MODEL_SRC="${BACKBONE_PATH:-microsoft/deberta-v3-large}"
LOCAL_MODEL="$LOCAL_STAGE_DIR/deberta-v3-large"
if [ -d "$MODEL_SRC" ]; then
    stage_dir "$MODEL_SRC" "$LOCAL_MODEL"
    BACKBONE_PATH="$LOCAL_MODEL"
else
    BACKBONE_PATH="$MODEL_SRC"
fi

# Find eval sets (small, ok to read directly)
EVAL_DIR=""
for eval_src in "$DATA_SRC/eval_sets"; do
    if [ -d "$eval_src" ]; then
        EVAL_DIR="$eval_src"
        echo "Eval sets at $EVAL_DIR"
        break
    fi
done

NGPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
EVAL_ARG=""
[ -n "$EVAL_DIR" ] && EVAL_ARG="--eval_dir $EVAL_DIR"

torchrun \
    --nnodes="${WORLD_SIZE:-1}" \
    --node_rank="${RANK:-0}" \
    --nproc_per_node="$NGPUS" \
    --master_addr="${MASTER_ADDR:-localhost}" \
    --master_port="${MASTER_PORT:-29500}" \
    train_critic.py \
        $DATA_ARG \
        --backbone "$BACKBONE_PATH" \
        --output_dir "${OUTPUT_DIR:-./checkpoints/critic}" \
        --epochs "${CRITIC_EPOCHS:-3}" \
        --max_steps "${CRITIC_MAX_STEPS:-0}" \
        --batch_size "${CRITIC_BATCH_SIZE:-32}" \
        --lr "${CRITIC_LR:-5e-5}" \
        --tau "${CRITIC_TAU:-1.0}" \
        --neg_per_pos "${CRITIC_NEG_PER_POS:-10}" \
        --num_workers "${CRITIC_NUM_WORKERS:-4}" \
        --log_every "${CRITIC_LOG_EVERY:-100}" \
        --save_every "${CRITIC_SAVE_EVERY:-2000}" \
        --eval_every_steps "${CRITIC_EVAL_EVERY_STEPS:-0}" \
        --amp_dtype "${CRITIC_AMP_DTYPE:-bf16}" \
        --distributed \
        $EVAL_ARG
