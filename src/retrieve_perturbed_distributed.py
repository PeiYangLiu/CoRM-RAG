"""
Retrieve top-100 passages for each perturbed query using Contriever + FAISS.

Each perturbation record is treated as a stand-alone retrieval query.

Multi-node multi-GPU:
    NODE_RANK=0 NUM_NODES=16 GPUS_PER_NODE=4 python retrieve_perturbed_distributed.py

Env:
    PERT_PATH       : perturbation JSONL file (staged local path)
    PASSAGES_PATH   : wiki_passages.jsonl (staged local)
    FAISS_PATH      : wiki.faiss (staged local)
    OUTPUT_DIR      : where shard output goes
    ENCODE_BATCH    : batch size for query encoding (default 256)
    TOP_K           : top-K to retrieve (default 100)
"""
import os, sys, json, time, logging, gc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("retrieve_pert")


def load_perts(path):
    """Load perturbations in file order. pert_idx = position within qi (file order)."""
    perts = []
    counter = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            qi = d["query_idx"]
            pi = counter.get(qi, 0)
            counter[qi] = pi + 1
            perts.append({
                "query_idx": qi,
                "pert_idx": pi,
                "question": d["question"],
                "perturbed_query": d["perturbed_query"],
                "perturbation_type": d.get("perturbation_type"),
                "correct_answer": d.get("correct_answer", ""),
                "all_answers": d.get("all_answers", []),
            })
    return perts


def mean_pool(out, mask):
    import torch
    emb = out[0]
    m = mask.unsqueeze(-1).float()
    return (emb * m).sum(1) / m.sum(1).clamp(min=1e-9)


def run_shard(shard_id, num_shards, perts, passages_path, faiss_path, output_dir):
    import torch, faiss, numpy as np
    from transformers import AutoTokenizer, AutoModel

    device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES already set by parent

    # --- FAISS via mmap (shared across processes on same node) ---
    log.info(f"[Shard {shard_id}] FAISS read_index (mmap) {faiss_path} ...")
    t0 = time.time()
    index = faiss.read_index(faiss_path, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
    # Boost recall for IVF indexes (no-op for Flat)
    try:
        nprobe = int(os.environ.get("FAISS_NPROBE", "16"))
        faiss.ParameterSpace().set_index_parameter(index, "nprobe", nprobe)
        log.info(f"[Shard {shard_id}] nprobe={nprobe}")
    except Exception as e:
        log.info(f"[Shard {shard_id}] nprobe not settable: {e}")
    log.info(f"[Shard {shard_id}] FAISS ntotal={index.ntotal} loaded in {time.time()-t0:.0f}s")

    # --- Passages (text) ---
    log.info(f"[Shard {shard_id}] Loading passages text ...")
    t0 = time.time()
    passages = []
    with open(passages_path) as f:
        for line in f:
            passages.append(json.loads(line)["text"])
    log.info(f"[Shard {shard_id}] {len(passages):,} passages in {time.time()-t0:.0f}s")

    # --- Model ---
    model_name = "facebook/contriever-msmarco"
    log.info(f"[Shard {shard_id}] Loading {model_name} ...")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    # fp32 to match the passage index (encoded fp32 in encode_wikipedia.py).

    # --- Shard the pert list ---
    N = len(perts)
    start = shard_id * N // num_shards
    end = (shard_id + 1) * N // num_shards
    my = perts[start:end]
    log.info(f"[Shard {shard_id}] queries {start}-{end} ({len(my)})")

    # Write to node-local /tmp first (some networked filesystems have occasional
    # silent data loss on append+fsync workloads). Copy to output_dir atomically
    # at the end.
    final_out_path = os.path.join(output_dir, f"retrieval_perturbed_shard_{shard_id:04d}.jsonl")
    tmp_dir = os.environ.get("LOCAL_TMP_DIR", "/tmp/ret_out")
    os.makedirs(tmp_dir, exist_ok=True)
    out_path = os.path.join(tmp_dir, f"retrieval_perturbed_shard_{shard_id:04d}.jsonl")

    # Resume: prefer already-valid final file if present (skip this shard); else
    # resume from tmp partial.
    if os.path.exists(final_out_path) and os.path.getsize(final_out_path) > 0:
        # Verify it has the expected number of lines before declaring done
        try:
            with open(final_out_path, "rb") as f:
                nlines = sum(1 for _ in f)
        except Exception:
            nlines = 0
        if nlines == len(my):
            log.info(f"[Shard {shard_id}] final already complete ({nlines} lines), skip")
            return
        else:
            log.info(f"[Shard {shard_id}] final has {nlines}/{len(my)} lines, redoing in tmp")

    done = 0
    if os.path.exists(out_path):
        last_good_end = 0
        with open(out_path, "rb") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    # Partial tail — ignore; we'll truncate to last_good_end.
                    break
                try:
                    json.loads(line)
                except Exception:
                    break
                done += 1
                last_good_end = f.tell()
        # Truncate trailing junk if any
        cur_size = os.path.getsize(out_path)
        if last_good_end < cur_size:
            log.info(f"[Shard {shard_id}] Truncating partial tail: "
                     f"{cur_size - last_good_end} bytes after last valid line {done}")
            with open(out_path, "r+b") as f:
                f.truncate(last_good_end)
        log.info(f"[Shard {shard_id}] Resuming at {done}/{len(my)}")

    BATCH = int(os.environ.get("ENCODE_BATCH", "256"))
    TOPK = int(os.environ.get("TOP_K", "100"))
    MAXLEN = int(os.environ.get("Q_MAXLEN", "256"))

    t0 = time.time()
    processed_this_run = 0
    with open(out_path, "a", buffering=1) as fout:
        for i in range(done, len(my), BATCH):
            batch = my[i:i+BATCH]
            texts = [p["perturbed_query"] for p in batch]
            enc = tok(texts, max_length=MAXLEN, truncation=True,
                      padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**enc)
                emb = mean_pool(out, enc["attention_mask"])
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            emb_np = emb.float().cpu().numpy().astype("float32")
            D, I = index.search(emb_np, TOPK)
            for j, p in enumerate(batch):
                cands = []
                for k in range(TOPK):
                    idx = int(I[j][k])
                    if idx < 0:
                        continue
                    cands.append({"passage_idx": idx, "text": passages[idx],
                                  "rank": k, "score": float(D[j][k])})
                fout.write(json.dumps({
                    "query_idx": p["query_idx"],
                    "pert_idx": p["pert_idx"],
                    "question": p["question"],
                    "perturbed_query": p["perturbed_query"],
                    "perturbation_type": p["perturbation_type"],
                    "correct_answer": p["correct_answer"],
                    "all_answers": p["all_answers"],
                    "candidates": cands,
                }) + "\n")
            fout.flush()
            try:
                os.fsync(fout.fileno())
            except OSError:
                pass
            processed_this_run += len(batch)
            total_done = done + processed_this_run
            elapsed = time.time() - t0
            rate = processed_this_run / max(elapsed, 1e-6)
            eta = (len(my) - total_done) / max(rate, 1e-6)
            log.info(f"[Shard {shard_id}] {total_done}/{len(my)} "
                     f"({100*total_done/len(my):.1f}%) "
                     f"rate={rate:.1f} q/s eta={eta/60:.1f}min")

    log.info(f"[Shard {shard_id}] DONE {len(my)} queries in {(time.time()-t0)/60:.1f}min -> {out_path}")

    # Atomic copy of the complete tmp file to the final output dir (safer than append+fsync).
    # Use temp name + rename so partial copies are not visible.
    tmp_final = final_out_path + ".partial"
    log.info(f"[Shard {shard_id}] copying {out_path} -> {final_out_path}")
    t_cp = time.time()
    import shutil
    shutil.copyfile(out_path, tmp_final)
    os.replace(tmp_final, final_out_path)
    # Verify final output size and line count
    try:
        final_size = os.path.getsize(final_out_path)
        tmp_size = os.path.getsize(out_path)
        if final_size != tmp_size:
            raise RuntimeError(f"final size {final_size} != tmp size {tmp_size}")
        with open(final_out_path, "rb") as f:
            final_lines = sum(1 for _ in f)
        if final_lines != len(my):
            raise RuntimeError(f"final lines {final_lines} != expected {len(my)}")
        log.info(f"[Shard {shard_id}] copy verified: {final_lines} lines, {final_size} bytes in {time.time()-t_cp:.0f}s")
    except Exception as e:
        log.error(f"[Shard {shard_id}] COPY VERIFICATION FAILED: {e}")
        raise
    del model, index
    gc.collect()


def main():
    node_rank = int(os.environ.get("NODE_RANK", os.environ.get("RANK", 0)))
    num_nodes = int(os.environ.get("NUM_NODES", os.environ.get("WORLD_SIZE", 1)))
    gpus_per_node = int(os.environ.get("GPUS_PER_NODE", 4))
    num_shards = num_nodes * gpus_per_node

    output_dir = os.environ.get("OUTPUT_DIR", "./output")
    os.makedirs(output_dir, exist_ok=True)

    pert_path = os.environ["PERT_PATH"]
    passages_path = os.environ["PASSAGES_PATH"]
    faiss_path = os.environ["FAISS_PATH"]

    log.info(f"Node {node_rank}/{num_nodes}, gpus/node={gpus_per_node}, "
             f"total shards={num_shards}, output={output_dir}")

    # Load perts once (parent); children inherit via fork (COW) — saves RAM
    perts = load_perts(pert_path)
    log.info(f"Loaded {len(perts):,} perturbations")

    my_shards = list(range(node_rank * gpus_per_node, (node_rank + 1) * gpus_per_node))
    log.info(f"Node {node_rank} shards: {my_shards}")

    pids = []
    for local_idx, shard_id in enumerate(my_shards):
        pid = os.fork()
        if pid == 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(local_idx)
            try:
                run_shard(shard_id, num_shards, perts, passages_path,
                          faiss_path, output_dir)
            except Exception as e:
                log.error(f"[Shard {shard_id}] FAILED: {e}")
                import traceback; traceback.print_exc()
                os._exit(1)
            os._exit(0)
        else:
            pids.append((pid, shard_id))
            log.info(f"  Launched shard {shard_id} on GPU {local_idx} (PID {pid})")

    failed = 0
    for pid, sid in pids:
        _, st = os.waitpid(pid, 0)
        ec = os.WEXITSTATUS(st)
        if ec != 0:
            log.error(f"Shard {sid} (PID {pid}) exit {ec}")
            failed += 1
        else:
            log.info(f"Shard {sid} (PID {pid}) done")

    log.info(f"Node {node_rank} done: {len(my_shards)-failed}/{len(my_shards)} succeeded")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
