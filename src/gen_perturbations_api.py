"""
Generate NQ perturbation files via an OpenAI-compatible async API.
Generates 5 perturbations per query with a rotated 1:1:1 global type balance.

Usage:
    python gen_perturbations_api.py \
        --output ./data/perturbations.jsonl \
        --concurrency 200 \
        --model <model> [--smoke 10]
"""
import os, sys, json, time, random, re, asyncio, argparse, logging
from typing import List, Dict, Any

import openai

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("gen_perts_api")

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


def load_nq_validation():
    from datasets import load_dataset
    nq = load_dataset("google-research-datasets/nq_open")
    out = []
    for idx, it in enumerate(nq["validation"]):
        if not it["answer"]:
            continue
        out.append({
            "query_idx": idx,
            "question": it["question"],
            "correct_answer": it["answer"][0],
            "all_answers": it["answer"],
        })
    return out


def build_wrong_pool(queries: List[Dict[str, Any]]) -> List[str]:
    raw = list({e["correct_answer"] for e in queries})
    return [a for a in raw if _is_entity_like(a)]


def make_prompt(ptype: int, q: str, wa: str = "", topic: str = "", t2_variant: str = "A", avoid: str = "") -> List[Dict[str, Any]]:
    if ptype == 1:
        user = PERT_USER_TYPE1.format(q=q, wa=wa, avoid=avoid)
    elif ptype == 2:
        tpl = PERT_USER_TYPE2_A if t2_variant == "A" else PERT_USER_TYPE2_B
        user = tpl.format(q=q, avoid=avoid)
    else:
        user = PERT_USER_TYPE3.format(q=q, topic=topic, avoid=avoid)
    return [
        {"role": "system", "content": [{"type": "input_text", "text": PERT_SYSTEM}]},
        {"role": "user", "content": [{"type": "input_text", "text": user}]},
    ]


def build_tasks(queries, entity_pool, seed=42):
    rnd = random.Random(seed)
    tasks = []  # list of (query_idx, slot_idx, ptype, prompt, meta)
    for e in queries:
        q = e["question"]
        ca = e["correct_answer"]
        avoid = ca
        candidates = [a for a in entity_pool if a.lower() != ca.lower()]
        was = rnd.sample(candidates, min(2, len(candidates))) if candidates else ["Unknown", "Unknown"]
        while len(was) < 2:
            was.append("Unknown")
        # Balance perturbation types across the dataset.
        # Each query has 5 slots; the slot types rotate over 3 patterns by
        # (query_idx % 3) so that, summed over any 3 consecutive queries,
        # type 1 / type 2 / type 3 each appear exactly 5 times (1:1:1).
        # This guarantees a globally balanced corpus for both training and
        # evaluation. Per-slot kw is type-specific.
        SLOT_PATTERNS = [
            [1, 2, 3, 1, 2],
            [2, 3, 1, 2, 3],
            [3, 1, 2, 3, 1],
        ]
        pattern = SLOT_PATTERNS[e["query_idx"] % 3]
        # Counters within this query for assigning the right kw to each slot
        t1_idx = 0  # picks was[0] then was[1]
        t2_idx = 0  # picks "A" then "B"
        slots = []
        for slot_idx, ptype in enumerate(pattern):
            if ptype == 1:
                wa = was[t1_idx % len(was)]
                t1_idx += 1
                slots.append((slot_idx, 1, {"wa": wa, "avoid": avoid}))
            elif ptype == 2:
                variant = "A" if (t2_idx % 2 == 0) else "B"
                t2_idx += 1
                slots.append((slot_idx, 2, {"t2_variant": variant, "avoid": avoid}))
            else:  # ptype == 3
                slots.append((slot_idx, 3, {"topic": rnd.choice(DISTRACTION_TOPICS), "avoid": avoid}))
        for slot_idx, ptype, kw in slots:
            prompt = make_prompt(ptype, q, **kw)
            avoid_terms = list(e["all_answers"])
            tasks.append({
                "query_idx": e["query_idx"],
                "slot_idx": slot_idx,
                "ptype": ptype,
                "prompt": prompt,
                "kw": kw,
                "q": q,
                "avoid_terms": avoid_terms,
            })
    return tasks


def clean_output(text: str) -> str:
    text = text.strip()
    # Strip optional prefixes the model sometimes adds
    text = re.sub(r"^(Rewritten|Sentence|Output|Answer)\s*:\s*", "", text, flags=re.IGNORECASE)
    # Take only first non-empty line
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return text


def _leaks(text: str, avoid_terms: List[str]) -> bool:
    t = text.lower()
    for a in avoid_terms:
        a = (a or "").strip().lower()
        if a and len(a) >= 3 and a in t:
            return True
    return False


HEDGE_WORDS = re.compile(r"\b(false|falsely|mistaken|mistakenly|wrong|incorrect|incorrectly|misconception|setting aside|despite|ignoring|contrary|although|even though|in reality|actually|supposedly|allegedly|putting aside|correcting|though|really|truly)\b", re.IGNORECASE)


def _is_hedged(text: str) -> bool:
    return bool(HEDGE_WORDS.search(text or ""))


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _contains_wrong_belief(text: str, wa: str) -> bool:
    if not wa:
        return False
    t = _norm(text)
    w = _norm(wa)
    if not w:
        return False
    # require all tokens of wa to appear (order-insensitive, whole-word)
    tokens = [tok for tok in w.split() if len(tok) >= 2]
    if not tokens:
        return False
    words = set(t.split())
    return all(tok in words for tok in tokens)


def _too_similar_to_original(text: str, q: str) -> bool:
    """True if `text` is basically the original question with only cosmetic edits (no added clause)."""
    t_tokens = _norm(text).split()
    q_tokens = _norm(q).split()
    # If rewrite adds fewer than 4 new content tokens vs original, treat as paraphrase.
    q_set = set(q_tokens)
    new_tokens = [tok for tok in t_tokens if tok not in q_set and len(tok) >= 3]
    return len(new_tokens) < 4


async def call_one(client, deployment, sem, task, max_retries=5):
    avoid_terms: List[str] = task.get("avoid_terms", [])
    async with sem:
        last = ""
        for attempt in range(max_retries):
            try:
                # On retry for T3, swap topic to diversify
                prompt = task["prompt"]
                if attempt > 0 and task["ptype"] == 3:
                    new_topic = random.choice(DISTRACTION_TOPICS)
                    q = task.get("q", "")
                    avoid = task["kw"].get("avoid", "")
                    user = PERT_USER_TYPE3.format(q=q, topic=new_topic, avoid=avoid)
                    prompt = [
                        {"role": "system", "content": [{"type": "input_text", "text": PERT_SYSTEM}]},
                        {"role": "user", "content": [{"type": "input_text", "text": user}]},
                    ]
                resp = await client.responses.create(
                    model=deployment,
                    input=prompt,
                )
                txt = getattr(resp, "output_text", None) or ""
                out = clean_output(txt)
                if not out:
                    continue
                if _leaks(out, avoid_terms):
                    continue
                if task["ptype"] in (1, 2):
                    if _is_hedged(out):
                        continue
                    if _too_similar_to_original(out, task.get("q", "")):
                        continue
                    if task["ptype"] == 1:
                        wa = task["kw"].get("wa", "")
                        if not _contains_wrong_belief(out, wa):
                            continue
                return out
            except Exception as ex:
                if attempt == max_retries - 1:
                    log.warning(f"giving up on query_idx={task['query_idx']} slot={task['slot_idx']}: {ex}")
                    return ""
                await asyncio.sleep(min(30, 2 ** attempt) + random.random())
        return last


async def run_all_streaming(tasks, queries, concurrency, deployment, out_path):
    """Stream-write: as soon as all 5 slots for a query_idx complete, write that line."""
    client = openai.AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL", None),
    )

    # Group tasks by query_idx
    by_q: Dict[int, List[Dict]] = {}
    for t in tasks:
        by_q.setdefault(t["query_idx"], []).append(t)

    q_lookup = {e["query_idx"]: e for e in queries}

    sem = asyncio.Semaphore(concurrency)
    done_cnt = 0
    t0 = time.time()
    total = len(tasks)
    file_lock = asyncio.Lock()
    pending_per_q: Dict[int, int] = {qid: len(ts) for qid, ts in by_q.items()}
    results_per_q: Dict[int, Dict[int, str]] = {qid: {} for qid in by_q}

    fout = open(out_path, "a", buffering=1)  # line-buffered append

    async def do_task(task):
        nonlocal done_cnt
        out = await call_one(client, deployment, sem, task)
        qid = task["query_idx"]
        results_per_q[qid][task["slot_idx"]] = out
        pending_per_q[qid] -= 1
        done_cnt += 1
        if pending_per_q[qid] == 0:
            # Assemble and write this query
            e = q_lookup[qid]
            slots = sorted(by_q[qid], key=lambda t: t["slot_idx"])
            perts = []
            for t in slots:
                o = results_per_q[qid].get(t["slot_idx"], "")
                ptype = t["ptype"]
                kw = t["kw"]
                if ptype == 3:
                    perts.append({
                        "perturbation_type": 3,
                        "perturbation_text": o,
                        "perturbed_query": f"{e['question']} {o}" if o else e["question"],
                        "topic": kw.get("topic", ""),
                    })
                else:
                    # T1/T2: LLM returns a full rewritten interrogative sentence
                    pq = o.strip() if o else e["question"]
                    if pq and not pq.endswith(("?", ".", "!")):
                        pq = pq + "?"
                    rec = {
                        "perturbation_type": ptype,
                        "perturbation_text": pq,
                        "perturbed_query": pq,
                    }
                    if ptype == 1:
                        rec["wrong_belief"] = kw.get("wa")
                    if ptype == 2:
                        rec["t2_variant"] = kw.get("t2_variant")
                    perts.append(rec)
            line = json.dumps({
                "query_idx": qid,
                "question": e["question"],
                "correct_answer": e["correct_answer"],
                "all_answers": e["all_answers"],
                "perturbations": perts,
            }, ensure_ascii=False)
            async with file_lock:
                fout.write(line + "\n")
            # free memory
            del results_per_q[qid]
            del by_q[qid]
        if done_cnt % 200 == 0 or done_cnt == total:
            dt = time.time() - t0
            rate = done_cnt / max(dt, 1e-6)
            eta = (total - done_cnt) / max(rate, 1e-6)
            log.info(f"{done_cnt}/{total} ({rate:.1f}/s, eta {eta/60:.1f}m)")

    await asyncio.gather(*[do_task(t) for t in tasks])
    fout.close()
    await client.close()


async def run_all(tasks, concurrency, deployment):
    client = openai.AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL", None),
    )

    sem = asyncio.Semaphore(concurrency)
    total = len(tasks)
    results: List[str] = [""] * total
    done_cnt = 0
    t0 = time.time()

    async def worker(i, task):
        nonlocal done_cnt
        out = await call_one(client, deployment, sem, task)
        results[i] = out
        done_cnt += 1
        if done_cnt % 200 == 0 or done_cnt == total:
            dt = time.time() - t0
            rate = done_cnt / max(dt, 1e-6)
            eta = (total - done_cnt) / max(rate, 1e-6)
            log.info(f"{done_cnt}/{total} ({rate:.1f}/s, eta {eta/60:.1f}m)")

    await asyncio.gather(*[worker(i, t) for i, t in enumerate(tasks)])
    await client.close()
    return results


def assemble(queries, tasks, raw_outputs, out_path):
    # Group by query_idx
    by_q: Dict[int, List[Any]] = {}
    for t, out in zip(tasks, raw_outputs):
        by_q.setdefault(t["query_idx"], []).append((t["slot_idx"], t["ptype"], t["kw"], out))

    n_empty = 0
    with open(out_path, "w") as f:
        for e in queries:
            slots = sorted(by_q.get(e["query_idx"], []), key=lambda x: x[0])
            perts = []
            for slot_idx, ptype, kw, out in slots:
                if not out:
                    n_empty += 1
                if ptype == 3:
                    # type-3: append distractor sentence to original question
                    if out:
                        perturbed_q = f"{e['question']} {out}"
                    else:
                        perturbed_q = e["question"]
                    perts.append({
                        "perturbation_type": 3,
                        "perturbation_text": out,
                        "perturbed_query": perturbed_q,
                        "topic": kw.get("topic", ""),
                    })
                else:
                    perts.append({
                        "perturbation_type": ptype,
                        "perturbation_text": out,
                        "perturbed_query": out if out else e["question"],
                        **({"wrong_belief": kw.get("wa")} if ptype == 1 else {}),
                        **({"t2_variant": kw.get("t2_variant")} if ptype == 2 else {}),
                    })
            f.write(json.dumps({
                "query_idx": e["query_idx"],
                "question": e["question"],
                "correct_answer": e["correct_answer"],
                "all_answers": e["all_answers"],
                "perturbations": perts,
            }) + "\n")
    log.info(f"wrote {out_path}  empty_outputs={n_empty}/{len(tasks)}")


def load_training_queries():
    """NQ-Open train subset used for critic distillation."""
    from datasets import load_dataset
    log.info("Loading NQ-Open train ...")
    nq = load_dataset("google-research-datasets/nq_open")
    out = []
    for it in nq["train"]:
        if not it["answer"]:
            continue
        out.append({"query_idx": len(out), "question": it["question"],
                    "correct_answer": it["answer"][0], "all_answers": it["answer"],
                    "source": "nq_open"})
    log.info(f"NQ-Open train: {len(out)}")
    random.seed(42)
    random.shuffle(out)
    max_train = int(os.environ.get("MAX_TRAIN_QUERIES", "10000"))
    out = out[:max_train]
    # re-index after shuffle
    for i, e in enumerate(out):
        e["query_idx"] = i
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--concurrency", type=int, default=200)
    ap.add_argument("--smoke", type=int, default=0, help="if >0, only do first N queries")
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5"))
    ap.add_argument("--data", choices=["nq_val", "training"], default="nq_val")
    args = ap.parse_args()

    log.info(f"Model: {args.model} | Data: {args.data}")
    if args.data == "training":
        queries = load_training_queries()
        log.info(f"Loaded {len(queries)} NQ training queries")
    else:
        queries = load_nq_validation()
        log.info(f"Loaded {len(queries)} NQ-Open validation queries")
    if args.smoke > 0:
        queries = queries[:args.smoke]
        log.info(f"SMOKE mode: truncated to {len(queries)}")

    entity_pool = build_wrong_pool(queries)
    log.info(f"entity pool: {len(entity_pool)}")

    # Resume: skip query_idx already present in output file
    done_qids = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for l in f:
                try:
                    done_qids.add(json.loads(l)["query_idx"])
                except Exception:
                    pass
        log.info(f"Resume: {len(done_qids)} queries already in output, skipping")
    queries = [q for q in queries if q["query_idx"] not in done_qids]
    log.info(f"Remaining queries: {len(queries)}")

    tasks = build_tasks(queries, entity_pool)
    log.info(f"total API calls: {len(tasks)}")

    if tasks:
        asyncio.run(run_all_streaming(tasks, queries, args.concurrency, args.model, args.output))
    log.info("done.")


if __name__ == "__main__":
    main()
