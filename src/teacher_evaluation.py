"""Teacher Evaluation: two-stage vLLM (answer + LLM judge), streaming + resume.

Distributed: NODE_RANK, NUM_NODES env vars (torch.distributed-style).
Shards per node: GPUS_PER_NODE / TP_SIZE (with TP_SIZE=1 by default).

For each (perturbed_query, doc) pair:
  Stage A: teacher LLM answers based on context
  Stage B: judge LLM decides correctness against gold answers

Deterministic sharding by qi range. Streaming append-mode output with fsync
per batch, resume by line-counting existing output file.

Required env:
  MODEL_PATH        : model directory (Qwen3-14B)
  NODE_RANK, NUM_NODES, GPUS_PER_NODE, TP_SIZE
  MAX_MODEL_LEN     : default 3072
  GPU_MEM_UTIL      : default 0.85
  STREAM_BATCH      : default 1024
  DATA_DIR          : where perturbations_filtered.jsonl and retrieval inputs live
  OUTPUT_DIR        : shard JSONL goes here
"""
import os, sys, json, time, re, logging, glob, hashlib
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("teacher")

TEACHER_SYS = ("Answer the question using ONLY the information in the context. "
               "Output the short answer span only (<= 10 words, no explanation). "
               "If the context does not support an answer, output exactly: unknown.")
TEACHER_USR = "Context: {ctx}\n\nQuestion: {q}"

JUDGE_SYS = ("You are an expert QA evaluator. Decide whether the prediction conveys "
             "the same information as ANY of the ground-truth answer(s), allowing "
             "paraphrase, abbreviations, date/number formatting and extra context. "
             "Output EXACTLY one word: yes or no. No explanation.")
JUDGE_USR = ("Question: {q}\nGround-truth answer(s): {golds}\nPrediction: {pred}\n\n"
             "Does the prediction match any of the ground truths? (yes/no)")

DATA_DIR = os.environ.get("DATA_DIR", "data")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "teacher_results")
RETRIEVAL_PATH = os.environ.get("RETRIEVAL_PATH", os.path.join(DATA_DIR, "retrieval_top100.jsonl"))


_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def _normalize(s: str) -> str:
    s = s.lower()
    s = _PUNCT.sub(" ", s)
    s = _ARTICLES.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def _exact_match_any(pred: str, golds) -> bool:
    """SQuAD-style normalized exact match: skips LLM judge when pred matches any gold.
    Also treats 'unknown' as never an EM to avoid mis-matching a literal gold 'unknown'."""
    if not pred: return False
    np = _normalize(pred)
    if not np or np == "unknown":
        return False
    for g in golds:
        if not g: continue
        ng = _normalize(g)
        if ng and np == ng:
            return True
    return False


def _doc_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def load_perts():
    """Group by query_idx, preserve file order as local pert_idx."""
    perts = defaultdict(list)
    fp = os.path.join(DATA_DIR, "perturbations_filtered.jsonl")
    with open(fp) as f:
        for ln in f:
            d = json.loads(ln)
            perts[int(d["query_idx"])].append(d)
    return dict(perts)


def count_pairs_for_shard(perts_by_qi, shard_qis, retrieval_path):
    total = 0
    with open(retrieval_path) as f:
        for ln in f:
            d = json.loads(ln)
            qi = int(d.get("query_idx"))
            if qi in shard_qis and qi in perts_by_qi:
                total += len(perts_by_qi[qi]) * len(d.get("candidates", []))
    return total


def shard_qi_range(sorted_qis, shard_id, num_shards):
    """Even qi slicing — since ~5 perts/qi is near-uniform, pair count per shard is balanced."""
    n = len(sorted_qis)
    start = shard_id * n // num_shards
    end = (shard_id + 1) * n // num_shards
    return set(sorted_qis[start:end])


def iter_shard_pairs(perts_by_qi, shard_qis, retrieval_path):
    """Stream retrieval, emit pairs for qis in this shard in deterministic order:
    for qi in sorted(shard_qis): for pi in 0..M: for dr in 0..99: yield
    """
    # Index retrieval by qi (just line offsets, no text content in memory)
    ret_by_qi = {}
    with open(retrieval_path) as f:
        pos = 0
        for ln in f:
            try:
                d = json.loads(ln)
            except Exception:
                pos = f.tell(); continue
            qi = d.get("query_idx")
            if qi in shard_qis:
                ret_by_qi[qi] = d
    log.info(f"Loaded retrieval for {len(ret_by_qi)}/{len(shard_qis)} shard qis")
    for qi in sorted(shard_qis):
        if qi not in ret_by_qi or qi not in perts_by_qi:
            continue
        ret = ret_by_qi[qi]
        golds = ret.get("all_answers") or [ret.get("correct_answer", "")]
        golds = [g for g in golds if g]
        perts = perts_by_qi[qi]
        cands = ret["candidates"]
        for pi, pert in enumerate(perts):
            for dr, cand in enumerate(cands):
                yield {
                    "query_idx": qi, "doc_rank": dr, "pert_idx": pi,
                    "perturbed_query": pert["perturbed_query"],
                    "doc_text": cand["text"],
                    "passage_idx": cand.get("passage_idx", cand.get("idx")),
                    "doc_text_hash": _doc_hash(cand["text"]),
                    "golds": golds,
                    "doc_score": cand["score"],
                    "pert_type": pert["perturbation_type"],
                }


def count_completed(output_path):
    if not os.path.exists(output_path):
        return 0
    n = 0
    with open(output_path) as f:
        for _ in f:
            n += 1
    return n


def build_prompts_teacher(tok, pairs):
    out = []
    for p in pairs:
        doc = p["doc_text"][:1400]
        msgs = [{"role": "system", "content": TEACHER_SYS},
                {"role": "user", "content": TEACHER_USR.format(ctx=doc, q=p["perturbed_query"][:500])}]
        out.append(tok.apply_chat_template(msgs, tokenize=False,
                   add_generation_prompt=True, enable_thinking=False))
    return out


def build_prompts_judge(tok, pairs, preds):
    out = []
    for p, pred in zip(pairs, preds):
        msgs = [{"role": "system", "content": JUDGE_SYS},
                {"role": "user", "content": JUDGE_USR.format(
                    q=p["perturbed_query"][:500],
                    golds=" | ".join(p["golds"][:8]),
                    pred=(pred or "unknown")[:200])}]
        out.append(tok.apply_chat_template(msgs, tokenize=False,
                   add_generation_prompt=True, enable_thinking=False))
    return out


def run_shard(shard_id, num_shards, tp_size, gpu_start):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_start + i) for i in range(tp_size))
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    model_path = os.environ["MODEL_PATH"]
    max_model_len = int(os.environ.get("MAX_MODEL_LEN", "3072"))
    gpu_mem = float(os.environ.get("GPU_MEM_UTIL", "0.85"))
    batch = int(os.environ.get("STREAM_BATCH", "1024"))

    out_path = os.path.join(OUTPUT_DIR, f"teacher_shard_{shard_id:04d}.jsonl")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    completed = count_completed(out_path)
    log.info(f"[Shard {shard_id}] CUDA={os.environ['CUDA_VISIBLE_DEVICES']} "
             f"TP={tp_size} resume from {completed}")

    perts = load_perts()
    sorted_qis = sorted(perts.keys())
    shard_qis = shard_qi_range(sorted_qis, shard_id, num_shards)
    # Quick total count for this shard (for ETA only)
    shard_total = count_pairs_for_shard(perts, shard_qis, RETRIEVAL_PATH)
    log.info(f"[Shard {shard_id}] qis={len(shard_qis)} pairs={shard_total:,} done={completed:,}")

    if completed >= shard_total:
        log.info(f"[Shard {shard_id}] Already complete.")
        return

    tok = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, tensor_parallel_size=tp_size,
              max_model_len=max_model_len, gpu_memory_utilization=gpu_mem,
              trust_remote_code=True, enforce_eager=False)
    sp_teacher = SamplingParams(temperature=0.0, max_tokens=32, stop=["<|im_end|>", "\n\n"])
    sp_judge = SamplingParams(temperature=0.0, max_tokens=4, stop=["<|im_end|>", "\n"])

    t0 = time.time()
    processed = 0
    # Iterator — skip completed items.
    it = iter_shard_pairs(perts, shard_qis, RETRIEVAL_PATH)
    for _ in range(completed):
        try: next(it)
        except StopIteration: break

    fout = open(out_path, "a", buffering=1)
    buf = []
    def flush_buf():
        nonlocal buf, processed
        if not buf: return
        t_prompts = build_prompts_teacher(tok, buf)
        t_outs = llm.generate(t_prompts, sp_teacher, use_tqdm=False)
        preds = [o.outputs[0].text.strip() for o in t_outs]
        # Short-circuit: if pred has an exact (normalized) match with any gold,
        # skip judge LLM. This cuts judge calls by ~30-50% on typical QA.
        em_hits = [_exact_match_any(pred, p["golds"]) for pred, p in zip(preds, buf)]
        need_judge_idx = [i for i, em in enumerate(em_hits) if not em]
        verdicts = [""] * len(buf)
        if need_judge_idx:
            sub_buf = [buf[i] for i in need_judge_idx]
            sub_preds = [preds[i] for i in need_judge_idx]
            j_prompts = build_prompts_judge(tok, sub_buf, sub_preds)
            j_outs = llm.generate(j_prompts, sp_judge, use_tqdm=False)
            sub_verdicts = [o.outputs[0].text.strip().lower() for o in j_outs]
            for i, v in zip(need_judge_idx, sub_verdicts):
                verdicts[i] = v
        for i, em in enumerate(em_hits):
            if em: verdicts[i] = "yes (em)"
        em_skipped = sum(em_hits)
        correct_cnt = 0
        for p, pred, v in zip(buf, preds, verdicts):
            ok = v.startswith("yes")
            if ok: correct_cnt += 1
            fout.write(json.dumps({
                "query_idx": p["query_idx"], "doc_rank": p["doc_rank"],
                "pert_idx": p["pert_idx"], "pert_type": p["pert_type"],
                "pred": pred, "judge_raw": v, "correct": ok,
                "doc_score": p["doc_score"],
                "passage_idx": p["passage_idx"],
                "doc_text_hash": p["doc_text_hash"],
            }) + "\n")
        fout.flush()
        os.fsync(fout.fileno())
        processed += len(buf)
        elapsed = time.time() - t0
        total_done = completed + processed
        rate = processed / elapsed if elapsed > 0 else 0
        remaining = shard_total - total_done
        eta_min = (remaining / rate / 60) if rate > 0 else 0
        log.info(f"[Shard {shard_id}] +{len(buf)} ({correct_cnt} yes, em_skip={em_skipped}) "
                 f"total={total_done:,}/{shard_total:,} "
                 f"({100*total_done/shard_total:.1f}%) "
                 f"rate={rate:.0f}/s ETA={eta_min:.0f}min")
        buf = []

    for pair in it:
        buf.append(pair)
        if len(buf) >= batch:
            flush_buf()
    flush_buf()
    fout.close()
    log.info(f"[Shard {shard_id}] DONE total={completed+processed:,} "
             f"elapsed={(time.time()-t0)/60:.1f}min")


def main():
    node_rank = int(os.environ.get("NODE_RANK", os.environ.get("RANK", "0")))
    num_nodes = int(os.environ.get("NUM_NODES", os.environ.get("WORLD_SIZE", "1")))
    gpus_per_node = int(os.environ.get("GPUS_PER_NODE", "4"))
    tp_size = int(os.environ.get("TP_SIZE", "1"))
    shards_per_node = gpus_per_node // tp_size
    num_shards = num_nodes * shards_per_node

    log.info(f"Node {node_rank}/{num_nodes}  gpus/node={gpus_per_node}  TP={tp_size}  "
             f"{shards_per_node}/node, total_shards={num_shards}")

    my_shards = [node_rank * shards_per_node + i for i in range(shards_per_node)]
    log.info(f"Node {node_rank} owns shards {my_shards}")

    # Fork one child per shard (so vLLM can be imported fresh with isolated CUDA_VISIBLE_DEVICES)
    children = []
    for i, shard_id in enumerate(my_shards):
        gpu_start = i * tp_size
        pid = os.fork()
        if pid == 0:
            try:
                run_shard(shard_id, num_shards, tp_size, gpu_start)
                os._exit(0)
            except Exception as e:
                log.exception(f"Shard {shard_id} crashed: {e}")
                os._exit(1)
        else:
            children.append((pid, shard_id))

    ok = 0
    for pid, sid in children:
        _, st = os.waitpid(pid, 0)
        rc = (st >> 8) & 0xff
        if rc == 0: ok += 1; log.info(f"Shard {sid} OK")
        else: log.error(f"Shard {sid} failed rc={rc}")
    log.info(f"Node {node_rank}: {ok}/{len(children)} shards ok")
    sys.exit(0 if ok == len(children) else 1)


if __name__ == "__main__":
    main()
