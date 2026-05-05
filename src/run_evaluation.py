#!/usr/bin/env python3
"""
Full evaluation pipeline for CoRM-RAG.

Evaluates: Contriever (first-stage retrieval) + CoRM-RAG (Critic rerank)
On: NQ-clean, Biased-NQ, TruthfulQA
Metrics: local answer-match accuracy, ECE, MCE, Spearman ρ, Risk-Coverage
"""
import os, json, time, re, logging, argparse
import numpy as np
import torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from scipy.stats import spearmanr, pearsonr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("eval")


# ═══════════════════════════════════════════════════
# Answer checking
# ═══════════════════════════════════════════════════

def strip_thinking(text):
    """Strip Qwen3 <think>...</think> reasoning block from the model output.
    Keep only the final answer after </think>, so EM matching isn't polluted
    by gold strings that appear inside the reasoning.

    If the thinking block was truncated by max_tokens (no closing </think>),
    salvage the last 2 sentences of the thinking as a best-effort answer so
    we don't lose the case entirely."""
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    # No closing tag: thinking was truncated.
    if "<think>" in text:
        prefix, thinking = text.split("<think>", 1)
        prefix = prefix.strip()
        if prefix:
            return prefix
        # Salvage: last 2 non-empty sentences of the truncated thinking.
        sents = re.split(r"(?<=[.!?])\s+", thinking.strip())
        sents = [s.strip() for s in sents if s.strip()]
        return " ".join(sents[-2:]) if sents else ""
    return text.strip()


def check_answer(prediction, gold_answers):
    pred = prediction.lower().strip().rstrip(".")
    for gold in gold_answers:
        g = gold.lower().strip()
        if len(g) <= 2:
            continue
        if re.search(r'\b' + re.escape(g) + r'\b', pred):
            return True
        if g in pred or (len(pred) > 3 and pred in g):
            return True
    return False


# ═══════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════

def expected_calibration_error(confidences, corrects, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0
    mce = 0
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = corrects[mask].mean()
        conf = confidences[mask].mean()
        gap = abs(acc - conf)
        ece += mask.sum() / len(confidences) * gap
        mce = max(mce, gap)
    return {"ece": float(ece), "mce": float(mce)}


def risk_coverage_curve(confidences, corrects, n_points=100):
    thresholds = np.linspace(0, 1, n_points)
    coverages, accuracies = [], []
    for t in thresholds:
        mask = confidences >= t
        cov = mask.mean()
        acc = corrects[mask].mean() if mask.any() else 0
        coverages.append(float(cov))
        accuracies.append(float(acc))
    return {"coverages": coverages, "accuracies": accuracies, "thresholds": thresholds.tolist()}


# ═══════════════════════════════════════════════════
# Load models
# ═══════════════════════════════════════════════════

def load_critic(checkpoint_path, backbone="microsoft/deberta-v3-large", device="cuda"):
    from train_critic import EvidenceCritic
    model = EvidenceCritic(backbone=backbone)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model"])
    # Disable grad checkpointing for eval (no-op under no_grad, but avoids any
    # library-side dtype/memory tricks that aren't needed at inference).
    try:
        model.encoder.gradient_checkpointing_disable()
    except Exception:
        pass
    model = model.to(device).float().eval()
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    return model, tokenizer


def score_with_model(model, tokenizer, queries, docs, device="cuda", batch_size=512, max_length=256):
    """Score (query, doc) pairs with the Evidence Critic."""
    all_scores = []
    for i in range(0, len(queries), batch_size):
        q_batch = queries[i:i+batch_size]
        d_batch = docs[i:i+batch_size]
        enc = tokenizer(q_batch, d_batch, max_length=max_length, truncation=True,
                       padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            scores = model.predict_robustness(enc["input_ids"], enc["attention_mask"])
        all_scores.extend(scores.cpu().tolist())
        if (i + batch_size) % (batch_size * 500) == 0:
            log.info(f"    scored {i+len(q_batch)}/{len(queries)} pairs")
    return all_scores


# ═══════════════════════════════════════════════════
# Retrieval
# ═══════════════════════════════════════════════════

def retrieve_contriever(queries, faiss_index, passages, embeddings_path, top_k=100):
    """Retrieve top-k passages using Contriever + FAISS."""
    import faiss
    
    model_name = "facebook/contriever-msmarco"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).cuda().eval()

    def encode(texts, batch_size=256):
        all_emb = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tok(batch, max_length=256, truncation=True, padding=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model(**enc)
                mask = enc["attention_mask"].unsqueeze(-1).float()
                emb = (out[0] * mask).sum(1) / mask.sum(1)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            all_emb.append(emb.cpu().numpy())
        return np.concatenate(all_emb, axis=0).astype(np.float32)

    log.info(f"Encoding {len(queries)} queries ...")
    q_emb = encode(queries)

    log.info(f"Searching top-{top_k} ...")
    scores, indices = faiss_index.search(q_emb, top_k)

    results = []
    for qi in range(len(queries)):
        cands = []
        for rank in range(top_k):
            pid = int(indices[qi][rank])
            if 0 <= pid < len(passages):
                cands.append({"text": passages[pid], "score": float(scores[qi][rank]), "idx": pid})
        results.append(cands)

    del model
    torch.cuda.empty_cache()
    return results


# ═══════════════════════════════════════════════════
# LLM Generation
# ═══════════════════════════════════════════════════

def generate_answers(queries, doc_lists, model_name="Qwen/Qwen3-8B", max_context_docs=3):
    """Generate answers using LLM given retrieved documents."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer as AT

    tok = AT.from_pretrained(model_name)
    llm = LLM(model=model_name, tensor_parallel_size=1, max_model_len=8192,
              gpu_memory_utilization=0.85, trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=4096, stop=["<|im_end|>"])

    prompts = []
    for q, docs in zip(queries, doc_lists):
        evidence = "\n\n".join([d["text"][:1500] for d in docs[:max_context_docs]])
        messages = [
            {"role": "system", "content": "Answer the question based on the evidence. Start your response with the correct factual answer in the first sentence (a few words is enough). Only after giving the answer, you may briefly note if the question contained a false premise."},
            {"role": "user", "content": f"Evidence:\n{evidence}\n\nQuestion: {q}"},
        ]
        prompts.append(tok.apply_chat_template(messages, tokenize=False,
                                                add_generation_prompt=True,
                                                enable_thinking=True))

    log.info(f"Generating {len(prompts)} answers ...")
    outputs = llm.generate(prompts, sp)
    answers = [strip_thinking(o.outputs[0].text) for o in outputs]

    del llm
    torch.cuda.empty_cache()
    return answers


# ═══════════════════════════════════════════════════
# Main evaluation
# ═══════════════════════════════════════════════════

def score_dataset(ds_name, queries, faiss_index, passages,
                  critic_model, critic_tok, rerank_depth=100,
                  output_dir=None, retrieval_queries=None):
    """Phase 1: Retrieve and score for one dataset. Returns methods dict.
    rerank_depth: only rerank top-N candidates.
    Caches results to output_dir/scored_{ds_name}.json for reuse.
    retrieval_queries: if provided, used for retrieval while `queries` (typically the
        perturbed query) is used for Critic scoring.
    """
    if retrieval_queries is None:
        retrieval_queries = queries
    # Check cache
    cache_path = os.path.join(output_dir, f"scored_{ds_name}.json") if output_dir else None
    if cache_path and os.path.exists(cache_path):
        log.info(f"Loading cached scoring from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    log.info(f"\n{'='*60}")
    log.info(f"Scoring: {ds_name} ({len(queries)} queries, rerank top-{rerank_depth})")
    log.info(f"  retrieval query != scoring query: {retrieval_queries is not queries}")
    log.info(f"{'='*60}")

    # Step 1: Retrieve top-100 with Contriever (uses retrieval_queries)
    contriever_results = retrieve_contriever(retrieval_queries, faiss_index, passages, None)

    # Truncate to rerank_depth for scoring (saves ~80% compute)
    truncated = [cands[:rerank_depth] for cands in contriever_results]

    # Flatten all (query, doc) pairs for batch scoring
    all_q, all_d, boundaries = [], [], [0]
    for qi, cands in enumerate(truncated):
        for c in cands:
            all_q.append(queries[qi])
            all_d.append(c["text"])
        boundaries.append(len(all_q))
    log.info(f"  Total pairs to score: {len(all_q)}")

    # CoRM-RAG Critic rerank
    log.info("  CoRM-RAG Critic scoring ...")
    critic_scores = score_with_model(critic_model, critic_tok, all_q, all_d, max_length=256)
    critic_reranked = []
    for qi in range(len(truncated)):
        cands = truncated[qi]
        scores = critic_scores[boundaries[qi]:boundaries[qi+1]]
        reranked = sorted(zip(cands, scores), key=lambda x: x[1], reverse=True)
        critic_reranked.append([{**c, "rerank_score": s} for c, s in reranked])
    methods = {"CoRM-RAG": critic_reranked}
    log.info("  Critic done.")

    # Cache results
    if cache_path:
        with open(cache_path, "w") as f:
            json.dump(methods, f)
        log.info(f"  Cached scoring to {cache_path}")

    return methods


ABSTAIN_STR = "Abstain: Insufficient reliable evidence"


def metric_confidences(confidences):
    """Use calibrated critic probabilities as-is; normalize non-probability retrieval scores."""
    conf = np.array(confidences, dtype=float)
    if len(conf) == 0:
        return conf
    if np.nanmin(conf) >= 0.0 and np.nanmax(conf) <= 1.0:
        return conf
    if np.nanmax(conf) > np.nanmin(conf):
        return (conf - np.nanmin(conf)) / (np.nanmax(conf) - np.nanmin(conf))
    return np.ones_like(conf) * 0.5


def select_context_docs(docs, gamma=None, max_context_docs=3):
    """Algorithm 1: gate robustly scored docs by gamma up to max context C."""
    if gamma is None or not docs or "rerank_score" not in docs[0]:
        return docs[:max_context_docs], False

    if docs[0]["rerank_score"] < gamma:
        return [], True

    selected = []
    for d in docs[:max_context_docs]:
        if d.get("rerank_score", 0.0) < gamma:
            break
        selected.append(d)
    return selected, False


def generate_and_evaluate(ds_name, queries, gold_answers, methods,
                          llm, llm_tok, llm_sp, output_dir,
                          abstain_threshold=None, max_context_docs=3):
    """Phase 2: Generate answers and compute metrics for one dataset.

    abstain_threshold (gamma): if set, robustly scored methods use Algorithm 1:
        top-1 below gamma abstains, otherwise only docs with score >= gamma are
        included in the generator context. Critic scores are raw sigmoid
        probabilities; non-probability retrieval scores are normalized only for
        metric curves.
    """
    log.info(f"\n{'='*60}")
    log.info(f"Generating & evaluating: {ds_name} ({len(queries)} queries)")
    log.info(f"{'='*60}")

    result_path = os.path.join(output_dir, f"results_{ds_name}.json")

    # Build all prompts across methods in one batch
    method_names = list(methods.keys())
    method_answers = {m: [None]*len(queries) for m in method_names}
    all_prompts = []
    prompt_map = []
    gamma = float(abstain_threshold) if abstain_threshold is not None else None
    for mi, method_name in enumerate(method_names):
        ranked_docs = methods[method_name]
        for qi, (q, docs) in enumerate(zip(queries, ranked_docs)):
            context_docs, abstained = select_context_docs(
                docs, gamma=gamma, max_context_docs=max_context_docs)
            if abstained:
                method_answers[method_name][qi] = ABSTAIN_STR
                continue
            evidence = "\n\n".join([d["text"][:1500] for d in context_docs])
            messages = [
                {"role": "system", "content": "Answer the question based on the evidence. Start your response with the correct factual answer in the first sentence (a few words is enough). Only after giving the answer, you may briefly note if the question contained a false premise. Always give the factual answer first."},
                {"role": "user", "content": f"Evidence:\n{evidence}\n\nQuestion: {q}"},
            ]
            all_prompts.append(llm_tok.apply_chat_template(messages, tokenize=False,
                                                            add_generation_prompt=True,
                                                            enable_thinking=True))
            prompt_map.append((mi, qi))

    log.info(f"  Generating {len(all_prompts)} answers ...")
    outputs = llm.generate(all_prompts, llm_sp) if all_prompts else []

    for idx, out in enumerate(outputs):
        mi, qi = prompt_map[idx]
        method_answers[method_names[mi]][qi] = strip_thinking(out.outputs[0].text)

    answers_path = os.path.join(output_dir, f"answers_{ds_name}.json")

    # Evaluate each method
    all_results = {}
    for method_name in method_names:
        ranked_docs = methods[method_name]
        answers = method_answers[method_name]

        corrects = []
        confidences = []
        for qi, (ans, golds) in enumerate(zip(answers, gold_answers)):
            hit = check_answer(ans, golds)
            corrects.append(hit)
            if ranked_docs[qi]:
                conf = ranked_docs[qi][0].get("rerank_score", ranked_docs[qi][0]["score"])
            else:
                conf = 0.0
            confidences.append(conf)

        corrects = np.array(corrects, dtype=float)
        confidences = np.array(confidences, dtype=float)
        conf_for_metrics = metric_confidences(confidences)

        acc = corrects.mean()
        cal = expected_calibration_error(conf_for_metrics, corrects)
        rho, p = spearmanr(conf_for_metrics, corrects)
        r, _ = pearsonr(conf_for_metrics, corrects)
        rc = risk_coverage_curve(conf_for_metrics, corrects)

        result = {
            "accuracy": float(acc),
            "ece": cal["ece"],
            "mce": cal["mce"],
            "spearman_rho": float(rho),
            "pearson_r": float(r),
            "n_examples": len(queries),
        }

        # Algorithm 1: Risk-Aware Abstention at threshold gamma.
        # Critic scores are already probabilities; retrieval scores are normalized
        # only for selective-prediction curves.
        if abstain_threshold is not None:
            answer_mask = conf_for_metrics >= gamma
            n_answered = int(answer_mask.sum())
            coverage = float(answer_mask.mean())
            sel_acc = float(corrects[answer_mask].mean()) if n_answered > 0 else 0.0
            for qi, keep in enumerate(answer_mask):
                if not keep:
                    method_answers[method_name][qi] = ABSTAIN_STR
            result.update({
                "abstain_threshold": gamma,
                "coverage": coverage,
                "selective_accuracy": sel_acc,
                "n_answered": n_answered,
                "n_abstained": len(queries) - n_answered,
            })
            log.info(f"  {method_name}: γ={gamma} → coverage={coverage:.3f}, "
                     f"selective_acc={sel_acc:.4f} ({n_answered}/{len(queries)})")

        all_results[method_name] = result
        log.info(f"  {method_name}: Acc={acc:.4f} ECE={cal['ece']:.4f} ρ={rho:.4f}")

        rc_path = os.path.join(output_dir, f"risk_coverage_{ds_name}_{method_name}.json")
        with open(rc_path, "w") as f:
            json.dump(rc, f)

    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"  Saved to {result_path}")

    # Save raw answers (post-abstention if gamma was set) for later re-evaluation.
    save_data = []
    for qi in range(len(queries)):
        entry = {"query": queries[qi], "gold_answers": gold_answers[qi]}
        for m in method_names:
            entry[f"answer_{m}"] = method_answers[m][qi]
            docs = methods[m][qi]
            entry[f"confidence_{m}"] = docs[0].get("rerank_score", docs[0]["score"]) if docs else 0.0
        save_data.append(entry)
    with open(answers_path, "w") as f:
        json.dump(save_data, f, indent=1)
    log.info(f"  Raw answers saved to {answers_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--critic_path", required=True)
    parser.add_argument("--generator", default="Qwen/Qwen3-8B")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--rerank_depth", type=int, default=100)
    parser.add_argument("--max_context_docs", type=int, default=3,
                        help="Maximum robust documents C included in the generator context.")
    parser.add_argument("--abstain_threshold", type=float, default=None,
                        help="Safety threshold γ (Algorithm 1). Queries whose top-1 "
                             "critic probability < γ are abstained. "
                             "Default: no gating.")
    parser.add_argument("--datasets", default="NQ_clean,Biased_NQ,TruthfulQA",
                        help="Comma-separated list of datasets to evaluate.")
    args = parser.parse_args()

    output_dir = args.output_dir or os.environ.get("OUTPUT_DIR", "./results")
    os.makedirs(output_dir, exist_ok=True)

    # Parallel loading: FAISS + passages + models simultaneously
    import faiss
    from concurrent.futures import ThreadPoolExecutor
    faiss.omp_set_num_threads(os.cpu_count())

    def load_faiss():
        log.info("Loading FAISS index (mmap) ...")
        idx = faiss.read_index(os.path.join(args.data_dir, "wiki.faiss"),
                               faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
        idx.nprobe = 64
        log.info(f"FAISS: {idx.ntotal} vectors")
        return idx

    def load_passages():
        log.info("Loading passages ...")
        p = []
        with open(os.path.join(args.data_dir, "wiki_passages.jsonl")) as f:
            for line in f:
                p.append(json.loads(line)["text"])
        log.info(f"Passages: {len(p)}")
        return p

    with ThreadPoolExecutor(max_workers=3) as pool:
        fut_faiss = pool.submit(load_faiss)
        fut_pass = pool.submit(load_passages)
        fut_critic = pool.submit(load_critic, args.critic_path)

    faiss_index = fut_faiss.result()
    passages = fut_pass.result()
    critic_model, critic_tok = fut_critic.result()

    # Load test datasets
    selected = set(s.strip() for s in args.datasets.split(",") if s.strip())
    log.info(f"Datasets selected: {sorted(selected)}")

    datasets = []

    if "NQ_clean" in selected:
        # 1. NQ-clean
        from datasets import load_dataset
        log.info("Loading NQ-Open validation ...")
        nq = load_dataset("google-research-datasets/nq_open")
        nq_queries = [it["question"] for it in nq["validation"] if it["answer"]]
        nq_golds = [it["answer"] for it in nq["validation"] if it["answer"]]
        datasets.append(("NQ_clean", nq_queries, nq_golds, None))

    if "Biased_NQ" in selected:
        # 2. Biased-NQ
        log.info("Loading Biased-NQ ...")
        biased_queries, biased_golds, biased_retrieval_queries = [], [], []
        with open(os.path.join(args.data_dir, "biased_nq_test.jsonl")) as f:
            for line in f:
                d = json.loads(line)
                # Pick one perturbation per query so that the test set is
                # balanced 1:1:1 across the three perturbation types
                # (1=False Premise, 2=Confirmation Bias, 3=Distraction).
                # We rotate the target type by (query_idx % 3); since every
                # query has all three types among its 5 slots, this works
                # every query has all three types among its five perturbations.
                pq = None
                if d.get("perturbations"):
                    target_type = 1 + (int(d.get("query_idx", 0)) % 3)
                    for p in d["perturbations"]:
                        if p.get("perturbation_type") == target_type:
                            pq = p["perturbed_query"]
                            break
                    if pq is None:
                        pq = d["perturbations"][0]["perturbed_query"]
                else:
                    pq = d["question"]
                biased_queries.append(pq)
                biased_retrieval_queries.append(pq)
                biased_golds.append(d["all_answers"])
        datasets.append(("Biased_NQ", biased_queries, biased_golds, biased_retrieval_queries))

    if "TruthfulQA" in selected:
        # 3. TruthfulQA
        from datasets import load_dataset
        log.info("Loading TruthfulQA ...")
        tqa = load_dataset("truthful_qa", "generation")
        tqa_queries = [it["question"] for it in tqa["validation"]]
        tqa_golds = [it.get("correct_answers", [it["best_answer"]]) for it in tqa["validation"]]
        datasets.append(("TruthfulQA", tqa_queries, tqa_golds, None))

    # ── Phase 1: Retrieval + Critic Scoring (small models on GPU ~6GB) ──
    log.info("\n" + "="*70)
    log.info("PHASE 1: Retrieval + Critic Scoring")
    log.info("="*70)
    all_methods = {}
    for ds_name, queries, golds, ret_queries in datasets:
        all_methods[ds_name] = score_dataset(
            ds_name, queries, faiss_index, passages,
            critic_model, critic_tok,
            rerank_depth=args.rerank_depth,
            output_dir=output_dir, retrieval_queries=ret_queries,
        )

    # Unload scoring models to free GPU for vLLM
    log.info("Unloading scoring models ...")
    del critic_model
    torch.cuda.empty_cache()

    # ── Phase 2: Generation + Evaluation (vLLM on GPU ~70GB) ──
    log.info("\n" + "="*70)
    log.info("PHASE 2: LLM Generation + Evaluation")
    log.info("="*70)
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer as AT
    log.info(f"Loading generator: {args.generator} ...")
    llm_tok = AT.from_pretrained(args.generator)
    llm = LLM(model=args.generator, tensor_parallel_size=1, max_model_len=8192,
              gpu_memory_utilization=0.85, trust_remote_code=True)
    llm_sp = SamplingParams(temperature=0.0, max_tokens=4096, stop=["<|im_end|>"])

    all_results = {}
    for ds_name, queries, golds, _ret in datasets:
        results = generate_and_evaluate(
            ds_name, queries, golds, all_methods[ds_name],
            llm, llm_tok, llm_sp, output_dir,
            abstain_threshold=args.abstain_threshold,
            max_context_docs=args.max_context_docs,
        )
        all_results[ds_name] = results
        with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)
        log.info(f"Cumulative results saved ({len(all_results)} datasets done)")

    # Print summary
    log.info("\n" + "="*70)
    log.info("RESULTS SUMMARY")
    log.info("="*70)
    header = f"{'Method':<16} | {'Dataset':<12} | {'Acc':>6} | {'ECE':>6} | {'ρ':>7}"
    log.info(header)
    log.info("-" * len(header))
    for ds, methods in all_results.items():
        for m, r in methods.items():
            log.info(f"{m:<16} | {ds:<12} | {r['accuracy']:.4f} | {r['ece']:.4f} | {r['spearman_rho']:.4f}")

    log.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
