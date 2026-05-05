#!/usr/bin/env python
"""
Pre-tokenize train_groups.jsonl into parquet shards for fast critic training.

Output schema per row (one row = one group):
  key:           str    (e.g. "1234_0")
  train_mode:    str    ('pointwise_only' | 'listwise+pointwise')
  pos_mask:      list<bool>            (length = n_docs)
  scores:        list<float32>         (soft robustness score per doc)
  input_ids:    list<list<int32>>     (length = n_docs, each inner list <= max_length)

Note: token_type_ids are NOT stored — EvidenceCritic.forward only consumes
input_ids + attention_mask, and attention_mask is trivially derived from length.

At training time, dataloader picks 1 random pos doc + N random neg docs from each group
without any tokenization (just an index lookup) and no big-file random seek (parquet
shards are small, sequential read).

Tokenization throughput is driven by the Rust 'tokenizers' library, which already
parallelizes over CPU cores, so a single Python process is enough.
"""
import os, sys, json, gzip, time, argparse, glob
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def open_writer(out_path, schema):
    return pq.ParquetWriter(out_path, schema, compression="zstd", compression_level=3)


def flush_buffer(buf, writer, schema):
    if not buf["key"]:
        return 0
    table = pa.Table.from_pydict(buf, schema=schema)
    writer.write_table(table)
    n = len(buf["key"])
    for k in buf:
        buf[k].clear()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default="data/training/train_groups.jsonl")
    ap.add_argument("--output_dir", default="data/training/critic_tokenized")
    ap.add_argument("--tokenizer", default="microsoft/deberta-v3-large")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--groups_per_shard", type=int, default=8000,
                    help="~8K groups/shard => ~120 shards, each ~300-500MB parquet")
    ap.add_argument("--tok_batch_size", type=int, default=2048,
                    help="Pairs sent to fast tokenizer per call (rust threads parallelize).")
    ap.add_argument("--skip_groups", type=int, default=0,
                    help="Skip the first N input groups (for parallel slicing).")
    ap.add_argument("--max_groups", type=int, default=0,
                    help="Process at most this many input groups (0 = all). For parallel slicing.")
    ap.add_argument("--shard_offset", type=int, default=0,
                    help="Starting shard index (default 0). Useful when multiple workers "
                         "write to the same output_dir.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    print(f"[tok] {tok.__class__.__name__} fast={tok.is_fast} vocab={tok.vocab_size}", flush=True)

    schema = pa.schema([
        ("key",            pa.string()),
        ("train_mode",     pa.string()),
        ("pos_mask",       pa.list_(pa.bool_())),
        ("scores",         pa.list_(pa.float32())),
        ("input_ids",      pa.list_(pa.list_(pa.int32()))),
    ])

    # Buffer per shard
    buf = {k: [] for k in ["key", "train_mode", "pos_mask", "scores", "input_ids"]}

    # Pending tokenize queue: collect ~ tok_batch_size (q,d) pairs across multiple groups
    pend_q = []     # list[str]
    pend_d = []     # list[str]
    pend_owner = [] # list[(group_index_in_buf, doc_index_in_group)]
    # docs_per_group[group_idx_in_buf] = total #docs (so we know when group is "complete")
    pending_groups = []   # list of dict(key, train_mode, pos_mask, scores, n_docs, ids=[None]*n)

    shard_idx = args.shard_offset
    shard_path = os.path.join(args.output_dir, f"shard-{shard_idx:05d}.parquet")
    writer = open_writer(shard_path, schema)

    t0 = time.time()
    n_groups_total = 0
    n_groups_in_shard = 0
    n_groups_with_pos = 0
    seq_lens_sum = 0
    n_pairs_total = 0

    def flush_pending():
        nonlocal n_pairs_total, seq_lens_sum
        if not pend_q:
            return
        enc = tok(
            pend_q, pend_d,
            max_length=args.max_length,
            truncation=True,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        for k, (gi, di) in enumerate(pend_owner):
            ids = enc["input_ids"][k]
            pending_groups[gi]["ids"][di] = ids
            seq_lens_sum += len(ids)
        n_pairs_total += len(pend_q)
        pend_q.clear(); pend_d.clear(); pend_owner.clear()

    def drain_completed_groups():
        nonlocal n_groups_in_shard, n_groups_total, n_groups_with_pos, shard_idx, writer, shard_path
        # Flush from front of pending_groups while their ids are all filled
        while pending_groups and all(x is not None for x in pending_groups[0]["ids"]):
            g = pending_groups.pop(0)
            buf["key"].append(g["key"])
            buf["train_mode"].append(g["train_mode"])
            buf["pos_mask"].append(g["pos_mask"])
            buf["scores"].append(g["scores"])
            buf["input_ids"].append(g["ids"])
            n_groups_in_shard += 1
            n_groups_total += 1
            if any(g["pos_mask"]):
                n_groups_with_pos += 1
            if n_groups_in_shard >= args.groups_per_shard:
                flush_buffer(buf, writer, schema)
                writer.close()
                size_mb = os.path.getsize(shard_path) / 1e6
                rate = n_groups_total / (time.time() - t0)
                print(f"[shard {shard_idx:05d}] {n_groups_in_shard} groups, {size_mb:.1f}MB | "
                      f"total {n_groups_total:,} groups, {rate:.1f} g/s, "
                      f"avg seq len {seq_lens_sum/max(n_pairs_total,1):.1f}",
                      flush=True)
                shard_idx += 1
                shard_path = os.path.join(args.output_dir, f"shard-{shard_idx:05d}.parquet")
                writer = open_writer(shard_path, schema)
                n_groups_in_shard = 0

    _open = gzip.open if args.input.endswith(".gz") else open
    with _open(args.input, "rt") as f:
        groups_seen = 0
        groups_accepted = 0
        for line in f:
            groups_seen += 1
            if groups_seen <= args.skip_groups:
                continue
            if args.max_groups > 0 and groups_accepted >= args.max_groups:
                break
            groups_accepted += 1
            g = json.loads(line)
            docs = g["docs"]
            n_docs = len(docs)
            scores = [float(d["score"]) for d in docs]
            pos_mask = [bool(s > 0) for s in scores]

            gi = len(pending_groups)
            pending_groups.append({
                "key": g["key"],
                "train_mode": g["train_mode"],
                "pos_mask": pos_mask,
                "scores": scores,
                "n_docs": n_docs,
                "ids":   [None] * n_docs,
            })
            q = g["question"]
            for di, d in enumerate(docs):
                pend_q.append(q)
                pend_d.append(d["text"])
                pend_owner.append((gi, di))

            if len(pend_q) >= args.tok_batch_size:
                flush_pending()
                drain_completed_groups()

    # final flush
    flush_pending()
    drain_completed_groups()
    flush_buffer(buf, writer, schema)
    writer.close()

    elapsed = time.time() - t0
    print(f"[done] {n_groups_total:,} groups ({n_groups_with_pos:,} with pos) "
          f"in {elapsed/60:.1f} min ({n_groups_total/elapsed:.1f} g/s, "
          f"{n_pairs_total/elapsed:.1f} pairs/s)", flush=True)
    # summary of shards
    shards = sorted(glob.glob(os.path.join(args.output_dir, "shard-*.parquet")))
    total_mb = sum(os.path.getsize(s) for s in shards) / 1e6
    print(f"[shards] {len(shards)} files, total {total_mb:.1f}MB at {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
