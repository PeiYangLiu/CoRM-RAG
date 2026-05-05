#!/usr/bin/env python3
"""
Preprocess train_expanded.jsonl into index format for lazy-loading DataLoader.

Creates:
  - train_groups.jsonl: one JSON per line, each line = one query group (all docs for one query+pert)
  - train_index.json: list of {offset, length, key, n_pos, n_neg, train_mode} for each group
  
This way DataLoader can seek to any group by byte offset without loading all data.
"""
import json, os, sys, time
from collections import defaultdict

def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "data/training/train_expanded.jsonl"
    output_dir = os.path.dirname(input_path)
    groups_path = os.path.join(output_dir, "train_groups.jsonl")
    index_path = os.path.join(output_dir, "train_index.json")

    t0 = time.time()
    print(f"Reading {input_path} and grouping ...", flush=True)

    # Single-pass streaming: group by (query_idx, pert_idx) and flush to disk
    # as soon as the key changes. Only the current group is held in RAM.
    # Assumes input is pre-sorted by (query_idx, pert_idx).
    current_key = None
    current_group = None
    index = []
    n_lines = 0
    n_groups = 0

    def flush(group, key, fout):
        nonlocal n_groups
        offset = fout.tell()
        line = json.dumps(group) + "\n"
        fout.write(line)
        n_pos = sum(1 for d in group["docs"] if d["score"] > 0)
        n_neg = sum(1 for d in group["docs"] if d["score"] == 0)
        index.append({
            "offset": offset,
            "length": len(line.encode('utf-8')),
            "key": key,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_docs": len(group["docs"]),
            "train_mode": group["train_mode"],
        })
        n_groups += 1

    with open(input_path) as fin, open(groups_path, "w") as fout:
        for line in fin:
            d = json.loads(line)
            qi = d["query_idx"]
            pi = d.get("pert_idx", -1)
            key = f"{qi}_{pi}"

            if key != current_key:
                if current_group is not None:
                    flush(current_group, current_key, fout)
                current_key = key
                current_group = {
                    "key": key,
                    "question": d["question"],
                    "train_mode": d["train_mode"],
                    "docs": [],
                }
            current_group["docs"].append({
                "text": d["document"],
                "score": d["robustness_score"],
                "rank": d["doc_rank"],
            })
            n_lines += 1
            if n_lines % 5_000_000 == 0:
                print(f"  {n_lines/1e6:.0f}M lines, {n_groups:,} groups flushed ...", flush=True)

        if current_group is not None:
            flush(current_group, current_key, fout)

    print(f"Read {n_lines:,} lines -> {n_groups:,} groups in {time.time()-t0:.0f}s", flush=True)

    # Write index
    with open(index_path, "w") as f:
        json.dump(index, f)

    groups_size = os.path.getsize(groups_path) / 1e9
    index_size = os.path.getsize(index_path) / 1e6
    n_with_pos = sum(1 for e in index if e["n_pos"] > 0)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"Groups: {n_groups:,} ({n_with_pos:,} with positives), {groups_size:.1f} GB", flush=True)
    print(f"Index: {index_size:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
