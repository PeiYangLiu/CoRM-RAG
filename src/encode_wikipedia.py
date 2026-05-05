"""
Encode all Wikipedia passages with Contriever (multi-GPU).
Each GPU encodes a shard of passages, results saved to separate files.
"""
import os, json, time, logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("encode_wiki")


class PassageDataset(Dataset):
    def __init__(self, passages, tokenizer, max_length=256):
        self.passages = passages
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.passages)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.passages[idx], max_length=self.max_length,
            truncation=True, padding="max_length", return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (torch.sum(token_embeddings * input_mask_expanded, 1) /
            torch.clamp(input_mask_expanded.sum(1), min=1e-9))


def encode_shard(gpu_id, passages, tokenizer, model_name, output_path, batch_size=512):
    """Encode a shard of passages on a single GPU."""
    device = torch.device(f"cuda:{gpu_id}")
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    dataset = PassageDataset(passages, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    all_embeddings = []
    t0 = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            emb = mean_pooling(outputs, batch["attention_mask"])
            # Normalize for cosine similarity
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            all_embeddings.append(emb.cpu().numpy())

            if (batch_idx + 1) % 100 == 0:
                done = min((batch_idx + 1) * batch_size, len(passages))
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (len(passages) - done) / rate / 60
                logger.info(f"[GPU {gpu_id}] {done:,}/{len(passages):,} "
                           f"({100*done/len(passages):.1f}%) "
                           f"{rate:.0f} passages/s, ETA {eta:.0f}min")

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    np.save(output_path, embeddings)
    elapsed = time.time() - t0
    logger.info(f"[GPU {gpu_id}] Done: {len(passages):,} passages in {elapsed/60:.1f}min "
               f"→ {output_path} ({embeddings.shape})")
    return embeddings.shape


def load_wikipedia(max_articles=None):
    """Load Wikipedia and chunk into ~200-word passages."""
    from datasets import load_dataset

    logger.info("Loading Wikipedia (streaming) ...")
    try:
        wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    except Exception:
        logger.info("Falling back to alternate wikipedia dataset name ...")
        wiki = load_dataset("wikipedia", "20220301.en", split="train",
                          streaming=True, trust_remote_code=True)

    passages = []
    titles = []
    t0 = time.time()
    for i, article in enumerate(wiki):
        text = article["text"]
        title = article["title"]
        words = text.split()
        for j in range(0, len(words), 200):
            chunk = " ".join(words[j:j+200])
            if len(chunk.split()) < 30:
                continue
            passages.append(f"{title}: {chunk}")

        if max_articles and i + 1 >= max_articles:
            break

        if (i + 1) % 100000 == 0:
            elapsed = time.time() - t0
            logger.info(f"  {i+1:,} articles → {len(passages):,} passages ({elapsed:.0f}s)")

    logger.info(f"Wikipedia: {len(passages):,} passages from {i+1:,} articles")
    return passages


def main():
    output_dir = os.environ.get("OUTPUT_DIR", "./output")
    os.makedirs(output_dir, exist_ok=True)

    num_gpus = torch.cuda.device_count()
    logger.info(f"GPUs available: {num_gpus}")

    model_name = "facebook/contriever-msmarco"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load Wikipedia
    passages = load_wikipedia()

    # Save passages text (needed later for retrieval)
    passages_path = os.path.join(output_dir, "wiki_passages.jsonl")
    logger.info(f"Saving passages text to {passages_path} ...")
    with open(passages_path, "w") as f:
        for p in passages:
            f.write(json.dumps({"text": p}) + "\n")
    logger.info(f"Saved {len(passages):,} passages")

    # Split across GPUs and encode in parallel
    shard_size = (len(passages) + num_gpus - 1) // num_gpus
    children = []

    for gpu_id in range(num_gpus):
        start = gpu_id * shard_size
        end = min(start + shard_size, len(passages))
        shard_passages = passages[start:end]
        shard_output = os.path.join(output_dir, f"embeddings_shard_{gpu_id}.npy")

        pid = os.fork()
        if pid == 0:
            try:
                encode_shard(gpu_id, shard_passages, tokenizer, model_name,
                           shard_output, batch_size=512)
            except Exception as e:
                logger.error(f"[GPU {gpu_id}] FAILED: {e}")
                import traceback; traceback.print_exc()
                os._exit(1)
            os._exit(0)
        else:
            children.append((pid, gpu_id))
            logger.info(f"Launched GPU {gpu_id}: passages {start:,}-{end:,} ({len(shard_passages):,})")

    # Wait for all
    failed = 0
    for pid, gpu_id in children:
        _, status = os.waitpid(pid, 0)
        if os.WEXITSTATUS(status) != 0:
            logger.error(f"GPU {gpu_id} (PID {pid}) failed!")
            failed += 1
        else:
            logger.info(f"GPU {gpu_id} (PID {pid}) done")

    if failed > 0:
        logger.error(f"{failed} GPUs failed!")
        # Don't exit(1) to avoid retry
    else:
        # Merge embeddings
        logger.info("Merging embeddings ...")
        shards = []
        for gpu_id in range(num_gpus):
            shard_path = os.path.join(output_dir, f"embeddings_shard_{gpu_id}.npy")
            shards.append(np.load(shard_path))
        all_emb = np.concatenate(shards, axis=0)
        merged_path = os.path.join(output_dir, "wiki_embeddings.npy")
        np.save(merged_path, all_emb)
        logger.info(f"Merged: {all_emb.shape} → {merged_path} "
                   f"({all_emb.nbytes / 1e9:.1f} GB)")

        # Clean up shards
        for gpu_id in range(num_gpus):
            os.remove(os.path.join(output_dir, f"embeddings_shard_{gpu_id}.npy"))

    logger.info("Done!")


if __name__ == "__main__":
    main()
