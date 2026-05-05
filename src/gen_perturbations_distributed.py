"""
Generate perturbations for the NQ-Open training subset used by CoRM-RAG.
Multi-node multi-GPU: each GPU runs one shard independently.

Usage:
    NODE_RANK=0 NUM_NODES=16 GPUS_PER_NODE=8 python gen_perturbations_distributed.py
"""
import os, sys, json, time, random, logging, re
import subprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("gen_perts")

PERT_SYSTEM = 'You rewrite a question so that the SAME information need is asked, but the speaker holds a FALSE background belief about something related. The rewritten question must be answerable with the EXACT SAME answer as the original — you are NOT writing a new question, you are re-voicing the SAME question from the mouth of someone who sincerely believes a false fact. Output ONLY the rewritten question as a single interrogative sentence ending with "?". No explanation, no quotes, no labels.'

PERT_USER_TYPE1 = '''Rewrite the following question into ONE natural-sounding interrogative sentence. CRITICAL: you are re-voicing the SAME question from the perspective of someone who sincerely believes the given "wrong belief". You are NOT writing a different question about a different topic — the information being asked must be EXACTLY the same as the original, answerable with the EXACT same correct answer.

Hard constraints:
- The rewritten question must ask for the SAME information as the original. A knowledgeable reader who ignores the false belief should give the EXACT answer "{avoid}" to both the original and the rewrite.
- The subject of the question must remain the same (same entity, same event, same object). Do NOT swap the subject for the wrong-belief entity.
- The wh-word / question type must be preserved ("who" stays "who", "when" stays "when", "how many" stays "how many", etc.).
- The "wrong belief" should be woven in as a PRESUPPOSITION / background assumption the speaker holds, NOT as the new subject of the question.
- Do NOT mention, hint at, or reveal the correct answer "{avoid}" anywhere.
- Do NOT include any of these tell-tale words that would flag the claim as false: false, falsely, mistaken, wrong, incorrect, misconception, setting aside, despite, ignoring, contrary, although, even though, in reality, actually, supposedly, allegedly, putting aside, correcting, really, truly.
- Vary the sentence structure naturally; do not fall into a fixed template.
- Output exactly ONE interrogative sentence ending with "?". Nothing else.

Examples (the rewrite asks for the SAME thing, just with a false presupposition):

Original: who painted the mona lisa
Correct answer: Leonardo da Vinci
Wrong belief: Michelangelo
Rewritten: In Michelangelo's portrait that we call the Mona Lisa, who did the painting?
(Note: still asks WHO painted the Mona Lisa; answer is still Leonardo da Vinci.)

Original: when did world war 2 end
Correct answer: 1945
Wrong belief: 1920
Rewritten: World War II came to a close in 1920 — on what date did it actually conclude?
(Note: still asks WHEN WW2 ended; answer is still 1945. The speaker just has the wrong year in mind.)

Original: who wrote hamlet
Correct answer: William Shakespeare
Wrong belief: Christopher Marlowe
Rewritten: In Christopher Marlowe's famous tragedy Hamlet, who's the credited playwright?
(Note: still asks WHO wrote Hamlet; answer is still Shakespeare.)

Original: who is the ceo of apple
Correct answer: Tim Cook
Wrong belief: Elon Musk
Rewritten: Since Elon Musk took over Apple, who runs the company as CEO?
(Note: still asks WHO is the CEO of Apple; answer is still Tim Cook.)

Original: how tall is mount everest
Correct answer: 29029 feet
Wrong belief: 5000 feet
Rewritten: Mount Everest stands at roughly 5000 feet — what is its height?
(Note: still asks the HEIGHT of Everest; answer is still 29029 feet.)

Now do the same. The information being asked must stay identical to the original question.
Original: {q}
Correct answer: {avoid}
Wrong belief: {wa}
Rewritten:'''

PERT_USER_TYPE2_A = '''Rewrite the following question into ONE natural-sounding interrogative sentence. CRITICAL: you are re-voicing the SAME question from the perspective of someone who sincerely believes a FALSE historical / temporal / existential claim related to the topic. You are NOT writing a different question — the information being asked must stay EXACTLY the same, answerable with the EXACT same correct answer.

Hard constraints:
- The rewritten question must ask for the SAME information as the original. A knowledgeable reader should give "{avoid}" as the answer to both.
- The subject and wh-word must be preserved (a "who" question stays "who", a "when" question stays "when", etc.). Do NOT change the topic.
- The embedded false premise must NOT invalidate or contradict the question — it should feel like a background assumption the asker holds while still asking the original question.
- Do NOT mention, hint at, or reveal the correct answer "{avoid}".
- Do NOT include any of these tell-tale words: false, falsely, mistaken, wrong, incorrect, misconception, setting aside, despite, ignoring, contrary, although, even though, in reality, actually, supposedly, allegedly, putting aside, correcting, really, truly.
- Vary the sentence structure naturally; do not fall into a fixed template.
- Output exactly ONE interrogative sentence ending with "?".

Examples (the rewrite asks for the SAME thing, just with a false historical/temporal presupposition):

Original: who painted the mona lisa
Correct answer: Leonardo da Vinci
Rewritten: The Mona Lisa, that famous 19th-century Impressionist masterpiece, was painted by whom?
(Still asks WHO painted the Mona Lisa; answer is still Leonardo da Vinci.)

Original: when did world war 2 end
Correct answer: 1945
Rewritten: World War II came to a close just a few years after the Berlin Wall fell — when did it end?
(Still asks WHEN WW2 ended; answer is still 1945.)

Original: who is the ceo of apple
Correct answer: Tim Cook
Rewritten: Steve Jobs still runs Apple today — who's the CEO?
(Still asks WHO is the CEO of Apple; answer is still Tim Cook.)

Original: what language is spoken in brazil
Correct answer: Portuguese
Rewritten: In Brazil, where every South American country shares one single language, what language is spoken?
(Still asks what language is spoken in Brazil; answer is still Portuguese.)

Now do the same. The information being asked must stay identical to the original question.
Original: {q}
Correct answer: {avoid}
Rewritten:'''

PERT_USER_TYPE2_B = '''Rewrite the following question into ONE natural-sounding interrogative sentence. CRITICAL: you are re-voicing the SAME question from the perspective of someone who sincerely believes a FALSE quantitative / relational / causal claim related to the topic. You are NOT writing a different question — the information being asked must stay EXACTLY the same, answerable with the EXACT same correct answer.

Hard constraints:
- The rewritten question must ask for the SAME information as the original. A knowledgeable reader should give "{avoid}" as the answer to both.
- The subject and wh-word must be preserved. Do NOT change the topic.
- The embedded false premise must NOT invalidate the question — it should feel like a background assumption the asker holds while still asking the original question.
- Do NOT mention, hint at, or reveal the correct answer "{avoid}".
- Do NOT include any of these tell-tale words: false, falsely, mistaken, wrong, incorrect, misconception, setting aside, despite, ignoring, contrary, although, even though, in reality, actually, supposedly, allegedly, putting aside, correcting, really, truly.
- Vary the sentence structure naturally; do not fall into a fixed template.
- Output exactly ONE interrogative sentence ending with "?".

Examples (the rewrite asks for the SAME thing, just with a false quantitative/relational presupposition):

Original: how tall is mount everest
Correct answer: 29029 feet
Rewritten: As the shortest peak in the entire Himalayan range, how tall is Mount Everest?
(Still asks the HEIGHT of Everest; answer is still 29029 feet.)

Original: who wrote hamlet
Correct answer: William Shakespeare
Rewritten: Hamlet, the play written collaboratively by more than twenty co-authors, is attributed to whom?
(Still asks WHO wrote Hamlet; answer is still Shakespeare.)

Original: what language is spoken in brazil
Correct answer: Portuguese
Rewritten: In Brazil — the only country in South America with three official national languages — what language is spoken?
(Still asks what language is spoken in Brazil; answer is still Portuguese.)

Original: who sang thriller
Correct answer: Michael Jackson
Rewritten: Thriller sold roughly ten thousand copies worldwide, so who sang it?
(Still asks WHO sang Thriller; answer is still Michael Jackson.)

Now do the same. The information being asked must stay identical to the original question.
Original: {q}
Correct answer: {avoid}
Rewritten:'''

PERT_USER_TYPE3 = '''You will be given a question along with its CORRECT answer. Your task is to write ONE plausible sentence on a COMPLETELY DIFFERENT topic from the question. The sentence will be appended to the original question as a distractor.

HARD REQUIREMENTS:
1. Your output is ONLY the distractor sentence (no question, no reference to the original topic).
2. The sentence must be about a completely unrelated domain (use the suggested topic for inspiration).
3. Your sentence MUST NOT state, imply, or contain the correct answer "{avoid}".
4. Your sentence MUST NOT mention anything about the subject of the original question.
5. Do NOT use overused examples: the Great Wall of China, the Eiffel Tower, Mount Everest, Great Barrier Reef, Amazon River, honeybees, elephants, jellyfish.
6. When the sentence is appended to the original question, a knowledgeable reader should still be able to answer "{avoid}" — the sentence must not contradict or override the original question's answer.

Examples:

Q: What is the speed of light?
Correct answer: 299792458 m/s
Topic: classical music
Sentence: Beethoven composed his Ninth Symphony while completely deaf.

Q: Who painted the Mona Lisa?
Correct answer: Leonardo da Vinci
Topic: marine biology
Sentence: The mantis shrimp has 16 types of photoreceptors in its eyes.

Now do the same:
Q: {q}
Correct answer: {avoid}
Topic: {topic}
Sentence:'''

DISTRACTION_TOPICS = [
    "classical music", "jazz history", "modern cinema", "board games", "chess openings",
    "marine biology", "entomology", "ornithology", "botany", "mycology",
    "ancient Egypt", "Roman empire", "medieval Europe", "Renaissance art", "Ming dynasty",
    "quantum physics", "organic chemistry", "geology", "meteorology", "astronomy",
    "computer science", "cryptography", "linguistics", "philosophy", "economics",
    "Formula 1 racing", "cricket history", "olympics", "mountaineering", "surfing culture",
    "Japanese folklore", "Norse mythology", "African literature", "South American cuisine",
    "coffee production", "wine regions", "tea ceremony", "pottery techniques",
    "architecture", "urban planning", "transportation history", "aviation",
    "textile industry", "paleontology", "archaeology", "genetics", "neuroscience",
    "world currencies", "postal systems", "cartography", "lexicography",
]

# --- Output validators (mirrored from gen_perturbations_api.py) ---
HEDGE_WORDS = re.compile(
    r"\b(false|falsely|mistaken|mistakenly|wrong|incorrect|incorrectly|misconception|"
    r"setting aside|despite|ignoring|contrary|although|even though|in reality|actually|"
    r"supposedly|allegedly|putting aside|correcting|though|really|truly)\b",
    re.I,
)

def _is_hedged(text: str) -> bool:
    return bool(HEDGE_WORDS.search(text or ""))

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

def _leaks(text: str, avoid_terms) -> bool:
    t = (text or "").lower()
    for term in avoid_terms:
        if term and len(term) > 2 and term.lower() in t:
            return True
    return False

def _contains_wrong_belief(text: str, wa: str) -> bool:
    if not wa:
        return False
    t = _norm(text); w = _norm(wa)
    if not w: return False
    tokens = [tok for tok in w.split() if len(tok) >= 2]
    if not tokens: return False
    words = set(t.split())
    return all(tok in words for tok in tokens)

def _too_similar_to_original(text: str, q: str) -> bool:
    t_tokens = _norm(text).split()
    q_set = set(_norm(q).split())
    new_tokens = [tok for tok in t_tokens if tok not in q_set and len(tok) >= 3]
    return len(new_tokens) < 4

def _validate(text: str, ptype: int, q: str, wa: str, avoid_terms) -> bool:
    """Returns True if output passes all filters."""
    if not text: return False
    if _leaks(text, avoid_terms): return False
    if ptype == 3: return True
    if _is_hedged(text): return False
    if _too_similar_to_original(text, q): return False
    if ptype == 1 and not _contains_wrong_belief(text, wa): return False
    return True


def load_all_queries():
    """Load the NQ-Open training subset used for critic distillation."""
    from datasets import load_dataset

    logger.info("Loading NQ-Open ...")
    nq = load_dataset("google-research-datasets/nq_open")
    all_queries = []
    for it in nq["train"]:
        if not it["answer"]:
            continue
        all_queries.append({
            "question": it["question"],
            "correct_answer": it["answer"][0],
            "all_answers": it["answer"],
            "wrong_answers": [],
            "source": "nq_open",
        })
    logger.info(f"NQ-Open train: {len(all_queries)}")

    random.seed(42)
    random.shuffle(all_queries)
    max_train = int(os.environ.get("MAX_TRAIN_QUERIES", "10000"))
    all_queries = all_queries[:max_train]
    for i, e in enumerate(all_queries):
        e["query_idx"] = i

    # Build a proper-noun-like wrong-belief pool.
    # Previously used raw correct_answers -> polluted with numbers, dates, phrases,
    # producing perturbations like "How did 54 Mbit/s write the lyrics...".
    def _is_entity_like(s: str) -> bool:
        s = s.strip()
        if not (3 <= len(s) <= 40):
            return False
        if re.search(r"\d", s):
            return False
        if not re.match(r"^[A-Z]", s):
            return False
        if not re.match(r"^[A-Za-z'\-\. ]+$", s):
            return False
        if len(s.split()) > 5:
            return False
        return True

    raw_pool = list({e["correct_answer"] for e in all_queries[:20000]})
    entity_pool = [a for a in raw_pool if _is_entity_like(a)]
    logger.info(f"wrong-belief pool: {len(entity_pool)}/{len(raw_pool)} entity-like answers")

    # Per-query independent sample, excluding the query's own correct answer.
    for e in all_queries:
        ca = e["correct_answer"].lower()
        candidates = [a for a in entity_pool if a.lower() != ca]
        e["wrong_answers"] = random.sample(candidates, min(3, len(candidates)))

    return all_queries


def run_shard(shard_id, num_shards, queries, output_dir, model_name=None, tp_size=2):
    """Run a single shard on the current GPU."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    if model_name is None:
        model_name = os.environ.get("MODEL_PATH", "Qwen/Qwen3-32B")

    total = len(queries)
    shard_size = (total + num_shards - 1) // num_shards
    start = shard_id * shard_size
    end = min(start + shard_size, total)
    shard_data = queries[start:end]

    if not shard_data:
        logger.info(f"[Shard {shard_id}] Empty shard, skipping.")
        return

    logger.info(f"[Shard {shard_id}] Queries {start}-{end} ({len(shard_data)} of {total})")

    # Load LLM
    # NOTE: max_model_len must accommodate prompt (~500 tok) + thinking (~2-3k) + answer (~100 tok).
    # max_tokens=4096 (output) requires max_model_len >= prompt_len + 4096. 8192 is safe.
    gpu_mem_util = float(os.environ.get("GPU_MEM_UTIL", "0.92"))
    max_model_len = int(os.environ.get("MAX_MODEL_LEN", "8192"))
    logger.info(f"[Shard {shard_id}] Loading {model_name} "
                f"(TP={tp_size}, max_model_len={max_model_len}, mem_util={gpu_mem_util}) ...")
    max_num_seqs = int(os.environ.get("MAX_NUM_SEQS", "32"))
    llm = LLM(model=model_name, tensor_parallel_size=tp_size,
              max_model_len=max_model_len, gpu_memory_utilization=gpu_mem_util,
              max_num_seqs=max_num_seqs,
              trust_remote_code=True, enforce_eager=False)
    tok = AutoTokenizer.from_pretrained(model_name)
    sp = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=4096,
                        stop=["<|im_end|>"])

    def strip_think(text: str) -> str:
        """Extract the post-thinking answer from a Qwen3 thinking-mode output.

        Handles 4 shapes:
          1. "<think>...</think>\nANSWER"   -> ANSWER
          2. "(pre-opened think)...</think>\nANSWER"  -> ANSWER
          3. "<think>... (truncated, no closing)"     -> "" (failed generation)
          4. "ANSWER" (no thinking tag)                -> ANSWER
        """
        # Case 1: complete <think>...</think> block
        text2 = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        if text2 != text:
            text = text2
        # Case 2: template pre-opened <think>, output starts with thinking content then </think>
        elif "</think>" in text:
            text = text.split("</think>", 1)[1]
        # Case 3: <think> present but no closing tag => truncated, signal failure
        elif "<think>" in text:
            return ""
        for line in text.strip().splitlines():
            line = line.strip()
            if line and not line.startswith("<think>"):
                return line
        return text.strip()

    def make_prompt(ptype, q, wa="", topic="", t2_variant="A", avoid=""):
        if ptype == 1:
            user_content = PERT_USER_TYPE1.format(q=q, wa=wa, avoid=avoid or "<unknown>")
        elif ptype == 2:
            tmpl = PERT_USER_TYPE2_A if t2_variant == "A" else PERT_USER_TYPE2_B
            user_content = tmpl.format(q=q, avoid=avoid or "<unknown>")
        else:
            user_content = PERT_USER_TYPE3.format(q=q, topic=topic, avoid=avoid or "<unknown>")
        messages = [
            {"role": "system", "content": PERT_SYSTEM},
            {"role": "user", "content": user_content},
        ]
        return tok.apply_chat_template(messages, tokenize=False,
                                        add_generation_prompt=True,
                                        enable_thinking=True)

    # Build five perturbations per query with a rotated 1:1:1 global balance
    # across Type I / Type II / Type III.
    topic_rng = random.Random(1337 + shard_id)
    all_prompts = []
    # meta: (local_idx, global_qi, ptype, wa)  wa only set for T1
    all_meta = []
    slot_patterns = [
        [1, 2, 3, 1, 2],
        [2, 3, 1, 2, 3],
        [3, 1, 2, 3, 1],
    ]
    for idx, e in enumerate(shard_data):
        q = e["question"]
        ca = e.get("correct_answer", "")
        was = e.get("wrong_answers", ["Unknown"])
        global_qi = int(e.get("query_idx", start + idx))
        pattern = slot_patterns[global_qi % 3]
        t1_idx = 0
        t2_idx = 0
        for ptype in pattern:
            if ptype == 1:
                wa = was[t1_idx % len(was)] if was else "Unknown"
                t1_idx += 1
                all_prompts.append(make_prompt(1, q, wa=wa, avoid=ca))
                all_meta.append((idx, global_qi, 1, wa))
            elif ptype == 2:
                variant = "A" if (t2_idx % 2 == 0) else "B"
                t2_idx += 1
                all_prompts.append(make_prompt(2, q, t2_variant=variant, avoid=ca))
                all_meta.append((idx, global_qi, 2, ""))
            else:
                topic = topic_rng.choice(DISTRACTION_TOPICS)
                all_prompts.append(make_prompt(3, q, topic=topic, avoid=ca))
                all_meta.append((idx, global_qi, 3, ""))

    logger.info(f"[Shard {shard_id}] Built {len(all_prompts)} prompts "
                f"(across {len(shard_data)} queries).")

    output_path = os.path.join(output_dir, f"perturbations_shard_{shard_id:04d}.jsonl")

    # --- Resume: count already-written prompts, skip those. ---
    done = 0
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for _ in f:
                done += 1
        logger.info(f"[Shard {shard_id}] Resuming: {done}/{len(all_prompts)} prompts "
                    f"already written in {output_path}")

    if done >= len(all_prompts):
        logger.info(f"[Shard {shard_id}] All prompts already done. Skipping generation.")
        del llm
        import gc, torch; gc.collect(); torch.cuda.empty_cache()
        return

    remaining_prompts = all_prompts[done:]
    remaining_meta = all_meta[done:]

            # --- Stream in batches. vLLM can batch requests, but we flush every BATCH prompts
    #     to disk so preemption at minute N only loses the in-flight batch.
    BATCH = int(os.environ.get("STREAM_BATCH", "512"))
    logger.info(f"[Shard {shard_id}] Generating {len(remaining_prompts)} remaining prompts "
                f"in batches of {BATCH} ...")

    MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
    t0 = time.time()
    written = done
    leak_count = 0
    testable = 0
    fallback_count = 0
    # Append mode; flush after each batch
    with open(output_path, "a", buffering=1) as fout:
        for bi in range(0, len(remaining_prompts), BATCH):
            batch_prompts = remaining_prompts[bi : bi + BATCH]
            batch_meta = remaining_meta[bi : bi + BATCH]
            outs = llm.generate(batch_prompts, sp)
            texts = [strip_think(o.outputs[0].text) for o in outs]

            # --- Retry invalid outputs ---
            for attempt in range(MAX_RETRIES):
                bad_idx = []
                for j, (text, (qi, _global_qi, ptype, wa)) in enumerate(zip(texts, batch_meta)):
                    e = shard_data[qi]
                    if not _validate(text, ptype, e["question"], wa, e["all_answers"]):
                        bad_idx.append(j)
                if not bad_idx:
                    break
                retry_sp = SamplingParams(temperature=0.95, top_p=0.95, max_tokens=4096,
                                          stop=["<|im_end|>"], seed=1000 + attempt)
                retry_prompts = [batch_prompts[j] for j in bad_idx]
                retry_outs = llm.generate(retry_prompts, retry_sp)
                for j, o in zip(bad_idx, retry_outs):
                    texts[j] = strip_think(o.outputs[0].text)
                logger.info(f"[Shard {shard_id}] batch {bi//BATCH} retry {attempt+1}: "
                            f"{len(bad_idx)} invalid -> regenerating")

            for text, (qi, global_qi, ptype, wa) in zip(texts, batch_meta):
                e = shard_data[qi]
                q = e["question"]
                ok = _validate(text, ptype, q, wa, e["all_answers"])
                if not ok:
                    fallback_count += 1
                    text_final = "" if ptype in (1, 2) else text  # T3 keeps whatever
                else:
                    text_final = text
                if ptype in (1, 2):
                    perturbed_q = text_final if text_final else q
                else:
                    perturbed_q = f"{q} [Note]: {text_final}" if text_final else q
                fout.write(json.dumps({
                    "query_idx": global_qi,
                    "question": q,
                    "correct_answer": e["correct_answer"],
                    "all_answers": e["all_answers"],
                    "perturbation_type": ptype,
                    "wrong_belief": wa if ptype == 1 else None,
                    "perturbation_text": text_final,
                    "perturbed_query": perturbed_q,
                    "shard_id": shard_id,
                }) + "\n")
                ca = e["correct_answer"]
                if len(ca) > 3 and ca.lower() not in q.lower():
                    testable += 1
                    if re.search(r"\b" + re.escape(ca.lower()) + r"\b", (text_final or "").lower()):
                        leak_count += 1
            fout.flush()
            try:
                os.fsync(fout.fileno())
            except OSError:
                pass
            written += len(batch_prompts)
            elapsed = time.time() - t0
            rate = (written - done) / max(elapsed, 1e-6)
            eta_s = (len(all_prompts) - written) / max(rate, 1e-6)
            logger.info(f"[Shard {shard_id}] {written}/{len(all_prompts)} "
                        f"({100*written/len(all_prompts):.1f}%) "
                        f"rate={rate:.2f} p/s  eta={eta_s/60:.1f} min  "
                        f"leak={leak_count}/{testable} fallback={fallback_count}")

    elapsed = time.time() - t0
    logger.info(f"[Shard {shard_id}] Generation done in {elapsed:.0f}s "
                f"({(len(all_prompts)-done)/max(elapsed,1e-6):.1f} prompts/s). "
                f"Final leak: {leak_count}/{testable} "
                f"({100*leak_count/max(testable,1):.1f}%)  fallback={fallback_count} → {output_path}")

    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()


def main():
    # --- Single-shard mode: one GPU, one shard, controlled by NUM_SHARDS / SHARD_ID env ---
    if os.environ.get("SHARD_MODE") == "single":
        shard_id = int(os.environ["SHARD_ID"])
        num_shards = int(os.environ["NUM_SHARDS"])
        tp_size = int(os.environ.get("TP_SIZE", 1))
        output_dir = os.environ.get("OUTPUT_DIR", "./output")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"[single-shard] shard_id={shard_id}/{num_shards} tp={tp_size} output={output_dir}")
        queries = load_all_queries()
        logger.info(f"Total queries: {len(queries):,}")
        run_shard(shard_id, num_shards, queries, output_dir, tp_size=tp_size)
        return

    node_rank = int(os.environ.get("NODE_RANK", os.environ.get("RANK", 0)))
    num_nodes = int(os.environ.get("NUM_NODES", os.environ.get("WORLD_SIZE", 1)))
    gpus_per_node = int(os.environ.get("GPUS_PER_NODE", 8))
    tp_size = int(os.environ.get("TP_SIZE", 2))  # tensor parallel per model

    models_per_node = gpus_per_node // tp_size
    num_shards = num_nodes * models_per_node
    output_dir = os.environ.get("OUTPUT_DIR", "./output")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Node {node_rank}/{num_nodes}, {gpus_per_node} GPUs, "
                f"TP={tp_size}, {models_per_node} models/node, "
                f"total shards={num_shards}, output={output_dir}")

    # Load data (all nodes load the same data, same seed → same order)
    queries = load_all_queries()
    logger.info(f"Total queries: {len(queries):,}")

    # Launch one process per model instance on this node
    my_shards = list(range(node_rank * models_per_node, (node_rank + 1) * models_per_node))
    logger.info(f"Node {node_rank} running shards: {my_shards}")

    pids = []
    for local_idx, shard_id in enumerate(my_shards):
        pid = os.fork()
        if pid == 0:
            # Child process: assign TP_SIZE consecutive GPUs
            gpu_start = local_idx * tp_size
            gpu_ids = ",".join(str(gpu_start + g) for g in range(tp_size))
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
            try:
                run_shard(shard_id, num_shards, queries, output_dir,
                         tp_size=tp_size)
            except Exception as e:
                logger.error(f"[Shard {shard_id}] FAILED: {e}")
                import traceback; traceback.print_exc()
                sys.exit(1)
            sys.exit(0)
        else:
            pids.append((pid, shard_id))
            gpu_start = local_idx * tp_size
            gpu_ids = ",".join(str(gpu_start + g) for g in range(tp_size))
            logger.info(f"  Launched shard {shard_id} on GPUs {gpu_ids} (PID {pid})")

    # Wait for all children
    failed = 0
    for pid, shard_id in pids:
        _, status = os.waitpid(pid, 0)
        exitcode = os.WEXITSTATUS(status)
        if exitcode != 0:
            logger.error(f"  Shard {shard_id} (PID {pid}) exited with code {exitcode}")
            failed += 1
        else:
            logger.info(f"  Shard {shard_id} (PID {pid}) completed OK")

    logger.info(f"\nNode {node_rank} done: {len(my_shards)-failed}/{len(my_shards)} shards succeeded")

    # List outputs
    import glob
    files = sorted(glob.glob(os.path.join(output_dir, "perturbations_shard_*.jsonl")))
    total_lines = 0
    for f in files:
        n = sum(1 for _ in open(f))
        total_lines += n
    logger.info(f"Output files: {len(files)}, total records: {total_lines:,}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
