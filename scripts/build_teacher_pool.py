#!/usr/bin/env python3
"""
Merge clean and perturbed retrieval pools into one per-query candidate file for
teacher evaluation.

The teacher then evaluates every candidate document under every perturbation,
which provides the soft robustness score s_{i,d} used by build_train_expanded.py.
"""
import argparse
import glob
import gzip
import hashlib
import json
import os
from collections import OrderedDict


def open_text(path, mode="rt"):
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)


def doc_key(cand):
    pid = cand.get("passage_idx", cand.get("idx"))
    if pid is not None:
        return f"pid:{pid}"
    text = cand.get("text", "")
    return "sha1:" + hashlib.sha1(text.encode("utf-8")).hexdigest()


def add_candidate(pool, cand):
    key = doc_key(cand)
    if key in pool["seen"]:
        return
    pool["seen"].add(key)
    out = {
        "text": cand["text"],
        "score": float(cand.get("score", 0.0)),
    }
    if cand.get("passage_idx") is not None:
        out["passage_idx"] = cand["passage_idx"]
    elif cand.get("idx") is not None:
        out["passage_idx"] = cand["idx"]
    pool["candidates"].append(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_retrieval", default="data/retrieval_top100.jsonl")
    ap.add_argument("--perturbed_retrieval_glob",
                    default="data/retrieval_perturbed_shard_*.jsonl")
    ap.add_argument("--output", default="data/retrieval_teacher_pool.jsonl")
    ap.add_argument("--max_per_query", type=int, default=0,
                    help="Optional cap after de-duplication (0 = keep full union).")
    args = ap.parse_args()

    pools = OrderedDict()
    with open_text(args.clean_retrieval) as f:
        for line in f:
            d = json.loads(line)
            qi = int(d["query_idx"])
            pools[qi] = {
                "query_idx": qi,
                "question": d["question"],
                "correct_answer": d.get("correct_answer", ""),
                "all_answers": d.get("all_answers", []),
                "candidates": [],
                "seen": set(),
            }
            for cand in d.get("candidates", []):
                add_candidate(pools[qi], cand)

    files = sorted(glob.glob(args.perturbed_retrieval_glob))
    if not files:
        raise FileNotFoundError(f"No files matched {args.perturbed_retrieval_glob}")
    for path in files:
        with open_text(path) as f:
            for line in f:
                d = json.loads(line)
                qi = int(d["query_idx"])
                if qi not in pools:
                    continue
                for cand in d.get("candidates", []):
                    add_candidate(pools[qi], cand)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_open = gzip.open if args.output.endswith(".gz") else open
    with out_open(args.output, "wt") as fout:
        for qi, pool in pools.items():
            cands = pool["candidates"]
            if args.max_per_query > 0:
                cands = cands[:args.max_per_query]
            fout.write(json.dumps({
                "query_idx": qi,
                "question": pool["question"],
                "correct_answer": pool["correct_answer"],
                "all_answers": pool["all_answers"],
                "candidates": cands,
            }, ensure_ascii=False) + "\n")
    print(f"Wrote {len(pools):,} query pools to {args.output}")


if __name__ == "__main__":
    main()
