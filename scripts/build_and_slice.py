#!/usr/bin/env python3
"""
Stream training shards, group by (query_idx, pert_idx), and distribute
the groups into N_SLICES sliced .jsonl.gz files.

Input  : data/training/train_expanded.shard_*.jsonl.gz
         (each record: query_idx, pert_idx, question, document, robustness_score,
          doc_rank, train_mode)
Output : OUT_DIR/slice_{k}.jsonl.gz         (k = 0..N_SLICES-1)
         OUT_DIR/manifest.json              (slice boundaries)

Group schema (matches preprocess_training_data.py output):
    {key, question, train_mode, docs: [{text, score, doc_rank}]}
where key = "{qi}_{pi}".

Distribution: round-robin by group so shards are balanced in group count.
"""
import os, json, glob, gzip, time, sys
from collections import defaultdict

SRC_GLOB  = os.environ.get("SRC_GLOB",  "data/training/train_expanded.shard_*.jsonl.gz")
OUT_DIR   = os.environ.get("OUT_DIR",   "data/training/slices")
N_SLICES  = int(os.environ.get("N_SLICES", "16"))


def emit_group(writers, group_key, question, train_mode, docs, slice_idx):
    line = json.dumps({
        "key": f"{group_key[0]}_{group_key[1]}",
        "question": question,
        "train_mode": train_mode,
        "docs": docs,
    }) + "\n"
    writers[slice_idx].write(line)


def main():
    t0 = time.time()
    files = sorted(glob.glob(SRC_GLOB))
    if not files:
        sys.exit(f"No shards matched {SRC_GLOB}")
    print(f"[in] {len(files)} source shards", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    writers = []
    for k in range(N_SLICES):
        p = os.path.join(OUT_DIR, f"slice_{k}.jsonl.gz")
        writers.append(gzip.open(p, "wt", compresslevel=3))

    group_counts = [0] * N_SLICES
    records_counts = [0] * N_SLICES

    cur_key = None
    cur_q = None
    cur_mode = None
    cur_docs = []
    n_groups = 0
    n_records = 0

    def flush():
        nonlocal n_groups
        if cur_key is None:
            return
        # round-robin by cumulative n_groups; keeps slices balanced and deterministic
        slc = n_groups % N_SLICES
        emit_group(writers, cur_key, cur_q, cur_mode, cur_docs, slc)
        group_counts[slc] += 1
        records_counts[slc] += len(cur_docs)
        n_groups += 1

    for i, f in enumerate(files):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                n_records += 1
                r = json.loads(line)
                qi, pi = r["query_idx"], r["pert_idx"]
                key = (qi, pi)
                if key != cur_key:
                    flush()
                    cur_key = key
                    cur_q = r["question"]
                    cur_mode = r["train_mode"]
                    cur_docs = []
                cur_docs.append({
                    "text": r["document"],
                    "score": r["robustness_score"],
                    "doc_rank": r["doc_rank"],
                })
                if n_records % 5_000_000 == 0:
                    el = time.time() - t0
                    print(f"  [{el:.0f}s] records={n_records:,} groups={n_groups:,}",
                          flush=True)
        print(f"  done shard {i+1}/{len(files)}: {f}  groups={n_groups:,}", flush=True)
    flush()

    for w in writers:
        w.close()

    manifest = {
        "n_slices": N_SLICES,
        "total_groups": n_groups,
        "total_records": n_records,
        "slices": [
            {"k": k, "groups": group_counts[k], "records": records_counts[k],
             "path": f"slice_{k}.jsonl.gz"}
            for k in range(N_SLICES)
        ],
    }
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    el = (time.time() - t0) / 60
    total_gb = sum(os.path.getsize(os.path.join(OUT_DIR, f"slice_{k}.jsonl.gz"))
                   for k in range(N_SLICES)) / 1e9
    print(f"\n[done] {el:.1f} min", flush=True)
    print(f"  groups: {n_groups:,}  records: {n_records:,}", flush=True)
    for k in range(N_SLICES):
        print(f"    slice_{k}: {group_counts[k]:,} groups, {records_counts[k]:,} records", flush=True)
    print(f"  total output: {total_gb:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
