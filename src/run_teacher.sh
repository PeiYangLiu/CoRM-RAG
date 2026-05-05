#!/bin/bash
# Teacher evaluation script. Input perturbations file name is configurable
# via $PERT_FILE_NAME.
set -e

echo "=== Teacher Evaluation (Qwen3-14B on perturbed queries) ==="
echo "Node: ${RANK:-0} / ${WORLD_SIZE:-1}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -2 || true

export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
mkdir -p "$HF_HOME"

export NODE_RANK="${RANK:-0}"
export NUM_NODES="${WORLD_SIZE:-1}"
export GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)

: "${TP_SIZE:=1}"
: "${MAX_MODEL_LEN:=3072}"
: "${GPU_MEM_UTIL:=0.85}"
: "${STREAM_BATCH:=1024}"
: "${PERT_FILE_NAME:=perturbations_filtered.jsonl}"
: "${RETRIEVAL_FILE_NAME:=retrieval_top100.jsonl}"
export TP_SIZE MAX_MODEL_LEN GPU_MEM_UTIL STREAM_BATCH

DATA_SRC="${DATA_SRC:-./data}"
export DATA_DIR="${DATA_DIR:-data}"
mkdir -p "$DATA_DIR"
echo "Copying data from $DATA_SRC ..."
# Teacher script reads perturbations_filtered.jsonl plus RETRIEVAL_PATH.
cp "$DATA_SRC/$PERT_FILE_NAME" "$DATA_DIR/perturbations_filtered.jsonl"
echo "  perturbations OK ($(wc -l < $DATA_DIR/perturbations_filtered.jsonl) lines)"
md5sum "$DATA_DIR/perturbations_filtered.jsonl"
cp "$DATA_SRC/$RETRIEVAL_FILE_NAME" "$DATA_DIR/$RETRIEVAL_FILE_NAME"
export RETRIEVAL_PATH="$DATA_DIR/$RETRIEVAL_FILE_NAME"
echo "  retrieval OK ($(wc -l < "$RETRIEVAL_PATH") lines)"
ls -lh "$DATA_DIR"/*.jsonl

export OUTPUT_DIR="${OUTPUT_DIR:-teacher_out}"
mkdir -p "$OUTPUT_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
ls -lh "$OUTPUT_DIR" | head -5 || true

echo ""
echo "Env: MODEL_PATH=$MODEL_PATH TP=$TP_SIZE MAX_LEN=$MAX_MODEL_LEN BATCH=$STREAM_BATCH"
python teacher_evaluation.py
