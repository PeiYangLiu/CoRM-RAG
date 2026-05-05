#!/bin/bash
set -e

echo "=== CoRM-RAG Evaluation ==="
echo "Checkpoint: ${CRITIC_CHECKPOINT:-best}"
echo "Results dir: ${CRITIC_RESULTS_DIR}"

export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
mkdir -p "$HF_HOME"

# Pre-download models with retries
echo "Pre-downloading HuggingFace models ..."
for i in 1 2 3; do
    python -c "
from transformers import AutoModel, AutoTokenizer
AutoTokenizer.from_pretrained('microsoft/deberta-v3-large')
AutoModel.from_pretrained('microsoft/deberta-v3-large')
AutoTokenizer.from_pretrained('facebook/contriever-msmarco')
AutoModel.from_pretrained('facebook/contriever-msmarco')
print('Models downloaded OK')
" && break || echo "Retry $i failed, waiting 30s..." && sleep 30
done

# ── Stage large data to node-local SSD (faster than network mount) ──
DATA_DIR="/tmp/eval_data"
mkdir -p "$DATA_DIR"

stage_file () {
    # $1 = src, $2 = dst
    local SRC="$1"; local DST="$2"
    if [ -f "$DST" ] && [ -f "$DST.DONE" ]; then
        echo "[stage] $DST already done, skip"
        return
    fi
    local SZ=$(du -sm "$SRC" | awk '{print $1}')
    echo "[stage] copy $SRC ($SZ MB) -> $DST"
    local T0=$(date +%s)
    cp "$SRC" "$DST"
    echo "[stage] done in $(( $(date +%s) - T0 ))s"
    touch "$DST.DONE"
}

DATA_SRC="${DATA_SRC:-./data}"

# Only stage wiki.faiss — it's mmap+random-reads (55GB), networked filesystems can't handle that well.
# wiki_passages.jsonl / biased_nq_test.jsonl / critic checkpoint are sequential one-pass reads, fine to read directly.
(stage_file "$DATA_SRC/wiki.faiss" "$DATA_DIR/wiki.faiss") &

# Symlink other inputs into DATA_DIR so run_evaluation.py's --data_dir sees them.
ln -sf "$DATA_SRC/wiki_passages.jsonl" "$DATA_DIR/wiki_passages.jsonl"
ln -sf "$DATA_SRC/biased_nq_test.jsonl" "$DATA_DIR/biased_nq_test.jsonl"

# Critic checkpoint: point at source path directly (torch.load = sequential read).
if [ -n "${CRITIC_PATH:-}" ]; then
    CKPT_SRC="$CRITIC_PATH"
else
    CKPT_NAME="${CRITIC_CHECKPOINT:-best}"
    CKPT_SRC="${CRITIC_CKPT_DIR:-./checkpoints/critic}/${CRITIC_RESULTS_DIR}/fold_0/checkpoint-${CKPT_NAME}/state.pt"
fi
if [ ! -f "$CKPT_SRC" ]; then
    echo "ERROR: checkpoint not found: $CKPT_SRC"
    exit 1
fi
mkdir -p "$DATA_DIR/critic"
ln -sf "$CKPT_SRC" "$DATA_DIR/critic/state.pt"

wait
echo "=== FAISS staged; other inputs read directly. ==="
ls -lah "$DATA_DIR/" "$DATA_DIR/critic/"
df -h /tmp

EXTRA_ARGS=()
if [ -n "${EVAL_RERANK_DEPTH:-}" ]; then
    EXTRA_ARGS+=(--rerank_depth "${EVAL_RERANK_DEPTH}")
fi
if [ -n "${EVAL_MAX_CONTEXT_DOCS:-}" ]; then
    EXTRA_ARGS+=(--max_context_docs "${EVAL_MAX_CONTEXT_DOCS}")
fi
if [ -n "${EVAL_ABSTAIN_THRESHOLD:-}" ]; then
    EXTRA_ARGS+=(--abstain_threshold "${EVAL_ABSTAIN_THRESHOLD}")
fi
if [ -n "${EVAL_DATASETS:-}" ]; then
    EXTRA_ARGS+=(--datasets "${EVAL_DATASETS}")
fi

python -u run_evaluation.py \
    --data_dir "$DATA_DIR" \
    --critic_path "$DATA_DIR/critic/state.pt" \
    --generator "${GENERATOR_MODEL:-Qwen/Qwen3-8B}" \
    --output_dir "${OUTPUT_DIR:-./results}" \
    "${EXTRA_ARGS[@]}"
