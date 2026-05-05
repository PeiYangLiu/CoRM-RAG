#!/bin/bash
# Retrieve perturbed queries with Contriever + FAISS.
# Input file name is configurable via $PERT_FILE_NAME (default
# perturbations_filtered.jsonl) and output dir is $OUTPUT_DIR.
set -e

echo "=== Retrieve perturbed queries (Contriever + FAISS) ==="
export NODE_RANK="${RANK:-0}"
export NUM_NODES="${WORLD_SIZE:-1}"
export GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
echo "NODE_RANK=$NODE_RANK NUM_NODES=$NUM_NODES GPUS_PER_NODE=$GPUS_PER_NODE"

export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
mkdir -p "$HF_HOME"

# Pre-download Contriever with retries
for i in 1 2 3; do
    python -c "
from transformers import AutoModel, AutoTokenizer
AutoTokenizer.from_pretrained('facebook/contriever-msmarco')
AutoModel.from_pretrained('facebook/contriever-msmarco')
print('Contriever ready')
" && break || { echo "Retry $i failed, sleep 30s"; sleep 30; }
done

DATA_DIR="/tmp/ret_data"
mkdir -p "$DATA_DIR"

: "${PERT_FILE_NAME:=perturbations_filtered.jsonl}"

stage_file () {
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
# Only stage wiki.faiss — it's mmap+random-reads (55GB), networked filesystems don't handle that well.
# wiki_passages.jsonl and perturbations file are sequential one-pass reads, fine to read directly.
(stage_file "$DATA_SRC/wiki.faiss" "$DATA_DIR/wiki.faiss") &
wait
echo "=== FAISS staged; reading other inputs directly. ==="
ls -lah "$DATA_DIR/"
md5sum "$DATA_SRC/$PERT_FILE_NAME"
df -h /tmp

export PERT_PATH="$DATA_SRC/$PERT_FILE_NAME"
export PASSAGES_PATH="$DATA_SRC/wiki_passages.jsonl"
export FAISS_PATH="$DATA_DIR/wiki.faiss"

python -u retrieve_perturbed_distributed.py
