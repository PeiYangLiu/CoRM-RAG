#!/usr/bin/env python3
"""
Build train_expanded.jsonl(.gz) following the paper's Counterfactual Data
Generation step.

For each query i:
  1. Estimate a soft robustness score s_{i,d} for each teacher-evaluated
     candidate document by averaging correctness over perturbations.
  2. For every perturbation k, form one compact listwise group with one
     positive sampled from clean retrieval (s > 0) and N hard negatives sampled
     from the perturbed-query retrieval pool (s = 0).

The output is record-per-document and can be consumed by build_and_slice.py or
preprocess_training_data.py.
"""
import argparse
import glob
import gzip
import hashlib
import json
import os
import random
import time
from collections import defaultdict


def open_text(path, mode="rt"):
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)


def doc_key_from_text(text):
    return "sha1:" + hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def doc_key(cand):
    pid = cand.get("passage_idx", cand.get("idx"))
    return f"pid:{pid}" if pid is not None else doc_key_from_text(cand.get("text", ""))


def load_clean_retrieval(path):
    clean = {}
    with open_text(path) as f:
        for line in f:
            d = json.loads(line)
            qi = int(d["query_idx"])
            clean[qi] = d
    return clean


def load_teacher_scores(paths, clean_retrieval):
    hits = defaultdict(lambda: defaultdict(list))
    for path in paths:
        with open_text(path) as f:
            for line in f:
                d = json.loads(line)
                qi = int(d["query_idx"])
                key = None
                if d.get("passage_idx") is not None:
                    key = f"pid:{d['passage_idx']}"
                elif d.get("doc_text_hash"):
                    key = f"sha1:{d['doc_text_hash']}"
                elif qi in clean_retrieval:
                    dr = int(d["doc_rank"])
                    cands = clean_retrieval[qi].get("candidates", [])
                    if 0 <= dr < len(cands):
                        key = doc_key(cands[dr])
                if key is None:
                    continue
                hits[qi][key].append(bool(d["correct"]))

    scores = {}
    for qi, per_doc in hits.items():
        scores[qi] = {k: sum(v) / len(v) for k, v in per_doc.items() if v}
    return scores


def iter_jsonl_glob(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {pattern}")
    for path in files:
        with open_text(path) as f:
            for line in f:
                yield json.loads(line)


def candidate_record(qi, pert_idx, question, cand, score, doc_rank, train_mode):
    return {
        "query_idx": qi,
        "pert_idx": pert_idx,
        "question": question,
        "document": cand["text"],
        "robustness_score": round(float(score), 4),
        "retriever_score": float(cand.get("score", 0.0)),
        "doc_rank": int(doc_rank),
        "train_mode": train_mode,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_retrieval", default="data/retrieval_top100.jsonl")
    ap.add_argument("--perturbed_retrieval_glob",
                    default="data/retrieval_perturbed_shard_*.jsonl")
    ap.add_argument("--teacher_glob",
                    default="data/teacher_results/teacher-eval/teacher_shard_*.jsonl")
    ap.add_argument("--output", default="data/training/train_expanded.jsonl")
    ap.add_argument("--neg_per_pos", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"Loading clean retrieval: {args.clean_retrieval}", flush=True)
    clean = load_clean_retrieval(args.clean_retrieval)
    print(f"  clean queries: {len(clean):,}", flush=True)

    teacher_files = sorted(glob.glob(args.teacher_glob))
    if not teacher_files:
        raise FileNotFoundError(f"No teacher shards matched {args.teacher_glob}")
    print(f"Loading teacher scores from {len(teacher_files)} shards ...", flush=True)
    scores = load_teacher_scores(teacher_files, clean)
    print(f"  scored queries: {len(scores):,}", flush=True)

    n_groups = n_records = n_skip_no_pos = n_skip_no_neg = 0
    out_open = gzip.open if args.output.endswith(".gz") else open
    with out_open(args.output, "wt") as fout:
        for rec in iter_jsonl_glob(args.perturbed_retrieval_glob):
            qi = int(rec["query_idx"])
            pert_idx = int(rec.get("pert_idx", 0))
            if qi not in clean or qi not in scores:
                continue

            q_scores = scores[qi]
            positives = []
            for rank, cand in enumerate(clean[qi].get("candidates", [])):
                s = q_scores.get(doc_key(cand))
                if s is not None and s > 0:
                    positives.append((rank, cand, s))
            if not positives:
                n_skip_no_pos += 1
                continue

            negatives = []
            seen = set()
            for rank, cand in enumerate(rec.get("candidates", [])):
                key = doc_key(cand)
                if key in seen:
                    continue
                seen.add(key)
                s = q_scores.get(key)
                if s == 0:
                    negatives.append((rank, cand))
            if len(negatives) < args.neg_per_pos:
                n_skip_no_neg += 1
                continue

            pos_rank, pos_cand, pos_score = rng.choice(positives)
            pos_key = doc_key(pos_cand)
            neg_pool = [(r, c) for r, c in negatives if doc_key(c) != pos_key]
            if len(neg_pool) < args.neg_per_pos:
                n_skip_no_neg += 1
                continue
            neg_chosen = rng.sample(neg_pool, args.neg_per_pos)

            question = rec.get("perturbed_query") or rec.get("question") or clean[qi]["question"]
            rows = [candidate_record(qi, pert_idx, question, pos_cand, pos_score,
                                     pos_rank, "listwise+pointwise")]
            rows.extend(candidate_record(qi, pert_idx, question, cand, 0.0,
                                         rank, "listwise+pointwise")
                        for rank, cand in neg_chosen)
            for row in rows:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_records += 1
            n_groups += 1
            if n_groups % 10000 == 0:
                print(f"  groups={n_groups:,} records={n_records:,}", flush=True)

    print(f"Done in {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"  groups: {n_groups:,}", flush=True)
    print(f"  records: {n_records:,}", flush=True)
    print(f"  skipped no positive: {n_skip_no_pos:,}", flush=True)
    print(f"  skipped insufficient negatives: {n_skip_no_neg:,}", flush=True)


if __name__ == "__main__":
    main()
