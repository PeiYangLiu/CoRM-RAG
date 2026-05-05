#!/bin/bash
set -e

echo "=== CoRM-RAG Distributed Perturbation Generation ==="
echo "Node: ${OMPI_COMM_WORLD_RANK:-0} / ${OMPI_COMM_WORLD_SIZE:-1}"
echo "GPUs per node: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -2 || true
echo ""

export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

export NODE_RANK="${RANK:-0}"
export NUM_NODES="${WORLD_SIZE:-1}"
export GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
export TP_SIZE="${TP_SIZE:-2}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"

echo "NODE_RANK=$NODE_RANK NUM_NODES=$NUM_NODES GPUS_PER_NODE=$GPUS_PER_NODE TP_SIZE=$TP_SIZE"
echo "MODEL_PATH=${MODEL_PATH:-(default from HF)} MAX_MODEL_LEN=$MAX_MODEL_LEN GPU_MEM_UTIL=$GPU_MEM_UTIL"

python gen_perturbations_distributed.py
