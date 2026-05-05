#!/usr/bin/env python3
"""
Train the Evidence Critic with held-out validation.
Supports multi-node DDP and checkpoint resume for preemption recovery.
"""
import os, sys, json, time, random, logging, argparse, glob, re

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("train_critic")


# ═══════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════

class EvidenceCritic(nn.Module):
    def __init__(self, backbone="microsoft/deberta-v3-large"):
        super().__init__()
        # Keep backbone weights in fp32; autocast handles mixed precision.
        self.encoder = AutoModel.from_pretrained(backbone)
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        hidden = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        # Ensure dtype matches head weights (robust under autocast / pure-eval paths).
        cls = cls.to(self.head[-1].weight.dtype)
        return self.head(cls).squeeze(-1)  # (batch,)

    def predict_robustness(self, input_ids, attention_mask):
        return torch.sigmoid(self.forward(input_ids, attention_mask))


# ═══════════════════════════════════════════════════
# Loss
# ═══════════════════════════════════════════════════

class HybridLoss(nn.Module):
    """Listwise ranking (CE) + pointwise confidence calibration (BCE)."""
    def __init__(self, tau=1.0):
        super().__init__()
        self.tau = tau
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, labels, train_mode, docs_per_query=None):
        """
        logits: (batch,) raw scores
        labels: (batch,) robustness scores in [0,1]
        train_mode: list of 'listwise+pointwise' or 'pointwise_only'
        docs_per_query: int, number of docs per query (1 + neg_per_pos).
                        Required for correct per-query listwise KL.
        """
        # Cast to float32 for numerical stability
        logits = logits.float()
        labels = labels.float()

        # Pointwise BCE for ALL samples
        loss_calib = self.bce(logits, labels).mean()

        # Listwise CE — computed per query/listwise group.
        if docs_per_query is not None and docs_per_query > 1:
            n_queries = logits.shape[0] // docs_per_query
            if n_queries > 0:
                # Reshape to (n_queries, docs_per_query)
                lw_logits = logits[:n_queries * docs_per_query].view(n_queries, docs_per_query)
                lw_labels = labels[:n_queries * docs_per_query].view(n_queries, docs_per_query)

                # Only include queries with listwise mode (check first doc per query)
                lw_modes = train_mode[::docs_per_query][:n_queries]
                lw_mask = torch.tensor([m == "listwise+pointwise" for m in lw_modes],
                                       device=logits.device)

                if lw_mask.any():
                    lw_logits = lw_logits[lw_mask]   # (n_lw, docs_per_query)
                    lw_labels = lw_labels[lw_mask]

                    label_sum = lw_labels.sum(dim=1, keepdim=True)
                    valid = label_sum.squeeze(1) > 0
                    if valid.any():
                        lw_logits = lw_logits[valid]
                        lw_labels = lw_labels[valid]
                        label_sum = label_sum[valid]
                        teacher = lw_labels / label_sum.clamp_min(1e-8)
                        student = torch.log_softmax(lw_logits / self.tau, dim=1)
                        loss_rank = torch.sum(-teacher * student, dim=1).mean()
                    else:
                        loss_rank = torch.tensor(0.0, device=logits.device)
                else:
                    loss_rank = torch.tensor(0.0, device=logits.device)
            else:
                loss_rank = torch.tensor(0.0, device=logits.device)
        else:
            loss_rank = torch.tensor(0.0, device=logits.device)

        return loss_calib + loss_rank, loss_calib.item(), loss_rank.item()


# ═══════════════════════════════════════════════════
# Dataset with online negative sampling
# ═══════════════════════════════════════════════════

class CriticDataset(Dataset):
    """
    Lazy-loading dataset. Reads from train_groups.jsonl using byte-offset index.
    Each line in groups file = one JSON object with all docs for a query group.
    Each __getitem__ returns 1 positive + N negatives from the same query group.
    """
    def __init__(self, index_entries, groups_path, tokenizer, neg_per_pos=10, max_length=512):
        self.groups_path = groups_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.neg_per_pos = neg_per_pos
        self._file = None

        self.positives = []
        self.index = index_entries
        for i, entry in enumerate(index_entries):
            if entry["n_pos"] > 0:
                for _ in range(entry["n_pos"]):
                    self.positives.append(i)

        logger.info(f"Dataset: {len(self.positives):,} positive samples, "
                    f"{len(self.index):,} groups")

    def _get_file(self):
        if self._file is None:
            self._file = open(self.groups_path, 'r')
        return self._file

    def _read_group(self, idx):
        entry = self.index[idx]
        f = self._get_file()
        f.seek(entry["offset"])
        return json.loads(f.read(entry["length"]))

    def __len__(self):
        return len(self.positives)

    def __getitem__(self, idx):
        group = self._read_group(self.positives[idx])
        question = group["question"]

        pos_docs = [d for d in group["docs"] if d["score"] > 0]
        neg_docs = [d for d in group["docs"] if d["score"] == 0]
        pos_doc = random.choice(pos_docs)

        if len(neg_docs) > self.neg_per_pos:
            neg_docs = random.sample(neg_docs, self.neg_per_pos)

        all_docs = [pos_doc] + neg_docs
        texts_a = [question] * len(all_docs)
        texts_b = [d["text"] for d in all_docs]
        scores = [d["score"] for d in all_docs]
        modes = [group["train_mode"]] * len(all_docs)

        enc = self.tokenizer(
            texts_a, texts_b,
            max_length=self.max_length,
            truncation=True, padding=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": torch.tensor(scores, dtype=torch.float32),
            "train_mode": modes,
        }


def collate_fn(batch):
    """Flatten nested batch and pad to longest sequence."""
    all_ids = [b["input_ids"] for b in batch]
    all_mask = [b["attention_mask"] for b in batch]
    labels = torch.cat([b["labels"] for b in batch], dim=0)
    modes = []
    for b in batch:
        modes.extend(b["train_mode"])

    # Pad all items to the max length in this batch
    max_len = max(ids.shape[1] for ids in all_ids)
    padded_ids, padded_mask = [], []
    for ids, mask in zip(all_ids, all_mask):
        pad_len = max_len - ids.shape[1]
        if pad_len > 0:
            ids = torch.nn.functional.pad(ids, (0, pad_len), value=0)
            mask = torch.nn.functional.pad(mask, (0, pad_len), value=0)
        padded_ids.append(ids)
        padded_mask.append(mask)

    return {
        "input_ids": torch.cat(padded_ids, dim=0),
        "attention_mask": torch.cat(padded_mask, dim=0),
        "labels": labels,
        "train_mode": modes,
    }


# ═══════════════════════════════════════════════════
# Pre-tokenized parquet dataset (preferred path)
# ═══════════════════════════════════════════════════

class CriticParquetDataset(Dataset):
    """
    Reads pre-tokenized groups from parquet shards (output of
    scripts/pretokenize_critic.py). Each row of the underlying Arrow table
    represents one (query, 100 docs) group with:
        key: str
        train_mode: str
        pos_mask: list<bool>
        scores: list<float32>
        input_ids: list<list<int32>>   (one tokenized [CLS] q [SEP] d [SEP] per doc)

    __getitem__ samples 1 random positive doc + neg_per_pos random negatives,
    builds a padded (k, L) tensor, and returns the soft robustness labels.
    No tokenization or large-file random IO at training time.

    Implementation note: uses pyarrow.parquet.read_table directly (no HF datasets
    "Generating train split" conversion step). Each shard is loaded into an
    in-memory Arrow Table at init; DataLoader workers share via fork CoW.
    """
    def __init__(self, shard_files, neg_per_pos=10, val_qis=None, mode="train",
                 _shared_state=None):
        """
        Args:
            shard_files: list of parquet paths.
            neg_per_pos: negatives per positive.
            val_qis: set of qi ids in val split.
            mode: 'train' or 'val'.
            _shared_state: optional dict from a sibling dataset to avoid loading
                shards twice (re-uses 'hf_ds', 'keys_all', 'pos_idx_per_row',
                'neg_idx_per_row', 'train_modes').
        """
        from datasets import load_dataset
        self.neg_per_pos = neg_per_pos
        shard_files = sorted(shard_files)

        if _shared_state is not None:
            self.hf_ds = _shared_state["hf_ds"]
            self._pm_flat = _shared_state["pm_flat"]
            self._pm_offsets = _shared_state["pm_offsets"]
            self._score_flat = _shared_state["score_flat"]
            self._score_offsets = _shared_state["score_offsets"]
            self._train_modes = _shared_state["_train_modes"]
            keys_all = _shared_state["keys_all"]
            n_total = len(self.hf_ds)
        else:
            logger.info(f"[{mode}] load_dataset('parquet', {len(shard_files)} shards) ...")
            t0 = time.time()
            self.hf_ds = load_dataset(
                "parquet",
                data_files={"train": shard_files},
                split="train",
                keep_in_memory=False,
            )
            logger.info(f"[{mode}] loaded in {time.time()-t0:.1f}s, "
                        f"{len(self.hf_ds):,} rows, "
                        f"arrow files: {[f['filename'] for f in self.hf_ds.cache_files][:2]}...")
            n_total = len(self.hf_ds)

            # Vectorized pos_mask extraction via pyarrow. Row lengths may be variable,
            # so we keep a flat bool array + per-row offsets instead of reshaping to 2D.
            t0 = time.time()
            import numpy as np
            pa_table = self.hf_ds.data.table
            pm_col = pa_table.column("pos_mask").combine_chunks()
            self._pm_flat = pm_col.values.to_numpy(zero_copy_only=False).astype(bool, copy=False)
            self._pm_offsets = pm_col.offsets.to_numpy().astype(np.int64, copy=False)
            if "scores" not in pa_table.column_names:
                raise ValueError("Tokenized critic shards must include a 'scores' column. "
                                 "Re-run scripts/pretokenize_critic.py with the updated code.")
            score_col = pa_table.column("scores").combine_chunks()
            self._score_flat = score_col.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
            self._score_offsets = score_col.offsets.to_numpy().astype(np.int64, copy=False)
            keys_all = pa_table.column("key").to_pylist()
            train_modes_all = pa_table.column("train_mode").to_pylist()
            logger.info(f"[{mode}] built pm_flat ({len(self._pm_flat):,} bools), "
                        f"scores ({len(self._score_flat):,} floats) + keys in {time.time()-t0:.1f}s")

            self._train_modes = train_modes_all
            self._shared_state = {
                "hf_ds": self.hf_ds,
                "pm_flat": self._pm_flat,
                "pm_offsets": self._pm_offsets,
                "score_flat": self._score_flat,
                "score_offsets": self._score_offsets,
                "_train_modes": self._train_modes,
                "keys_all": keys_all,
            }

        # Flat list of (global_row_idx, pos_doc_idx) for split-filtered positives,
        # built via fully vectorized numpy: no Python-level per-row loop.
        import numpy as np
        t0 = time.time()
        pm_flat = self._pm_flat
        pm_offsets = self._pm_offsets
        row_lens = np.diff(pm_offsets).astype(np.int64, copy=False)
        # map each flat position → (row, col within row)
        row_ids_flat = np.repeat(np.arange(n_total, dtype=np.int64), row_lens)
        col_ids_flat = np.arange(int(pm_offsets[-1]), dtype=np.int64) - np.repeat(pm_offsets[:-1], row_lens)
        pos_rows_arr = row_ids_flat[pm_flat]
        pos_cols_arr = col_ids_flat[pm_flat]
        if val_qis is not None:
            qi_per_row = np.fromiter(
                (int(k.split("_", 1)[0]) for k in keys_all),
                dtype=np.int64, count=n_total,
            )
            val_qis_arr = np.fromiter(val_qis, dtype=np.int64, count=len(val_qis))
            in_val_row = np.isin(qi_per_row, val_qis_arr)
            keep_row = in_val_row if mode == "val" else ~in_val_row
            keep = keep_row[pos_rows_arr]
            pos_rows_arr = pos_rows_arr[keep]
            pos_cols_arr = pos_cols_arr[keep]
        elif mode == "val":
            pos_rows_arr = pos_rows_arr[:0]
            pos_cols_arr = pos_cols_arr[:0]
        self.positives = np.stack([pos_rows_arr, pos_cols_arr], axis=1).astype(np.int32, copy=False)
        logger.info(f"[{mode}] built positives index in {time.time()-t0:.1f}s: shape={self.positives.shape}")

        logger.info(f"ParquetDataset[{mode}]: {len(self.positives):,} positive samples, "
                    f"{n_total:,} groups (mmap'd arrow cache, {len(shard_files)} shards)")

    def _row_input_ids(self, global_row_idx):
        """Zero-copy mmap'd random access via HF datasets."""
        return self.hf_ds[int(global_row_idx)]["input_ids"]

    def __len__(self):
        return len(self.positives)

    def __getitem__(self, idx):
        row_idx, pdi = self.positives[idx]
        row_idx = int(row_idx); pdi = int(pdi)
        all_ids = self._row_input_ids(row_idx)
        # Compute neg_pool on-the-fly from pm_flat slice (fast, no pre-materialized lists).
        s = int(self._pm_offsets[row_idx]); e = int(self._pm_offsets[row_idx + 1])
        mask = self._pm_flat[s:e]
        import numpy as np
        neg_pool = np.nonzero(~mask)[0]
        if len(neg_pool) > self.neg_per_pos:
            neg_chosen = random.sample(neg_pool.tolist(), self.neg_per_pos)
        else:
            neg_chosen = neg_pool.tolist()
        chosen = [pdi] + neg_chosen
        ids_list = [all_ids[i] for i in chosen]
        score_start = int(self._score_offsets[row_idx])
        scores = self._score_flat[score_start:score_start + len(mask)]
        labels = [float(scores[i]) for i in chosen]
        train_mode = self._train_modes[row_idx]

        max_len = max(len(x) for x in ids_list)
        ids_t = torch.zeros(len(chosen), max_len, dtype=torch.long)
        mask_t = torch.zeros(len(chosen), max_len, dtype=torch.long)
        for i, x in enumerate(ids_list):
            L = len(x)
            ids_t[i, :L] = torch.tensor(x, dtype=torch.long)
            mask_t[i, :L] = 1
        return {
            "input_ids": ids_t,
            "attention_mask": mask_t,
            "labels": torch.tensor(labels, dtype=torch.float32),
            "train_mode": [train_mode] * len(chosen),
        }


# ═══════════════════════════════════════════════════
# Checkpoint
# ═══════════════════════════════════════════════════

def save_checkpoint(model, optimizer, scheduler, epoch, step, best_metric, output_dir,
                    tag="latest", best_by="val_loss"):
    path = os.path.join(output_dir, f"checkpoint-{tag}")
    os.makedirs(path, exist_ok=True)
    state = {
        "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_metric": best_metric,
        "best_by": best_by,
        "best_loss": best_metric if best_by == "val_loss" else float("inf"),
    }
    torch.save(state, os.path.join(path, "state.pt"))
    logger.info(f"Saved checkpoint: {path} (epoch={epoch}, step={step}, "
                f"best_{best_by}={best_metric:.4f})")


def load_checkpoint(model, optimizer, scheduler, output_dir, tag="latest", device="cpu",
                    best_by="val_loss"):
    """Returns (epoch, step, best_metric). best_metric is seeded from the checkpoint iff
    the saved metric matches the requested best_by; otherwise reset to a neutral extreme
    so any freshly-computed metric will register as 'best'."""
    worst = float("inf") if best_by == "val_loss" else float("-inf")
    path = os.path.join(output_dir, f"checkpoint-{tag}", "state.pt")
    if not os.path.exists(path):
        return 0, 0, worst
    state = torch.load(path, map_location=device)
    if hasattr(model, "module"):
        model.module.load_state_dict(state["model"])
    else:
        model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    prior_best_by = state.get("best_by", "val_loss")
    if prior_best_by == best_by and "best_metric" in state:
        best_metric = state["best_metric"]
    elif prior_best_by == "val_loss" and best_by == "val_loss":
        best_metric = state.get("best_loss", worst)
    else:
        # Different criterion than before — reset so new best_by takes over.
        best_metric = worst
        logger.info(f"Resuming with new --save_best_by={best_by} (was {prior_best_by}); "
                    f"resetting best_metric to {worst}")
    logger.info(f"Resumed from {path} (epoch={state['epoch']}, step={state['step']}, "
                f"best_{best_by}={best_metric})")
    return state["epoch"], state["step"], best_metric


# ═══════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════

def train_fold(fold_id, train_ds, val_ds, args):
    # DDP setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if args.distributed:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    is_main = (not args.distributed) or (dist.get_rank() == 0)

    output_dir = os.path.join(args.output_dir, f"fold_{fold_id}")
    os.makedirs(output_dir, exist_ok=True)

    if is_main:
        logger.info(f"=== Fold {fold_id} ===")
        logger.info(f"Train samples: {len(train_ds):,}, Val samples: {len(val_ds):,}")

    # Model — only rank 0 downloads from HuggingFace, others wait
    if args.distributed:
        if local_rank == 0:
            model = EvidenceCritic(backbone=args.backbone).to(device)
        dist.barrier()
        if local_rank != 0:
            model = EvidenceCritic(backbone=args.backbone).to(device)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    else:
        model = EvidenceCritic(backbone=args.backbone).to(device)

    # Dataset already constructed by main()
    train_sampler = DistributedSampler(train_ds, shuffle=True) if args.distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if args.distributed else None

    _dl_kwargs = dict(
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    if args.num_workers > 0:
        _dl_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                              shuffle=(train_sampler is None), collate_fn=collate_fn,
                              **_dl_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler,
                            shuffle=False, collate_fn=collate_fn,
                            **_dl_kwargs)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.06 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    loss_fn = HybridLoss(tau=args.tau)

    # Mixed precision dtype: bf16 (default, for A100/H100) or fp16 (for V100)
    amp_dtype_str = (args.amp_dtype or "bf16").lower()
    if amp_dtype_str in ("bf16", "bfloat16"):
        amp_dtype = torch.bfloat16
        use_scaler = False
    elif amp_dtype_str in ("fp16", "float16", "half"):
        amp_dtype = torch.float16
        use_scaler = True
    else:
        raise ValueError(f"Unsupported --amp_dtype: {args.amp_dtype}")
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None
    if is_main:
        logger.info(f"AMP dtype: {amp_dtype_str} (GradScaler={'on' if use_scaler else 'off'})")

    # Resume
    start_epoch, start_step, best_metric = load_checkpoint(
        model, optimizer, scheduler, output_dir, device=device,
        best_by=args.save_best_by)

    def _is_better(new, cur):
        return new < cur if args.save_best_by == "val_loss" else new > cur

    if is_main:
        logger.info(f"Training loop start: total_steps={total_steps}, "
                    f"steps_per_epoch={len(train_loader)}, warmup={warmup_steps}")
        sys.stdout.flush(); sys.stderr.flush()

    # Run pre-training eval at step 0
    def run_eval(epoch_tag, global_step, save_ckpt=True, epoch_for_ckpt=0):
        nonlocal best_metric
        model.eval()
        val_loss_local = 0.0
        val_n_local = 0
        pos_scores_local = []
        neg_scores_local = []
        ndcg_sum_local = 0.0
        mrr_sum_local = 0.0
        n_ranked_queries_local = 0
        dpq = 1 + args.neg_per_pos

        with torch.no_grad():
            for _vi, batch in enumerate(val_loader):
                if args.val_max_batches > 0 and _vi >= args.val_max_batches:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                modes = batch["train_mode"]
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    logits = model(input_ids, attention_mask)
                    loss, _, _ = loss_fn(logits, labels, modes, docs_per_query=dpq)
                val_loss_local += loss.item()
                val_n_local += 1
                scores = torch.sigmoid(logits).float().cpu().numpy()
                lbls = labels.float().cpu().numpy()
                n_q = len(scores) // dpq
                for qi in range(n_q):
                    s = scores[qi*dpq:(qi+1)*dpq]
                    l = lbls[qi*dpq:(qi+1)*dpq]
                    pos_scores_local.append(float(s[0]))
                    neg_scores_local.extend(s[1:].tolist())
                    ranked = sorted(range(len(s)), key=lambda i: s[i], reverse=True)
                    dcg = sum(l[ranked[k]] / np.log2(k+2) for k in range(min(3, len(ranked))))
                    ideal = sorted(l, reverse=True)
                    idcg = sum(ideal[k] / np.log2(k+2) for k in range(min(3, len(ideal))))
                    ndcg_sum_local += dcg / max(idcg, 1e-8)
                    for k, idx in enumerate(ranked):
                        if l[idx] > 0:
                            mrr_sum_local += 1.0 / (k + 1)
                            break
                    n_ranked_queries_local += 1

        if args.distributed:
            pos_sum_local = float(sum(pos_scores_local))
            pos_sqsum_local = float(sum(x*x for x in pos_scores_local))
            pos_cnt_local = len(pos_scores_local)
            neg_sum_local = float(sum(neg_scores_local))
            neg_sqsum_local = float(sum(x*x for x in neg_scores_local))
            neg_cnt_local = len(neg_scores_local)
            t = torch.tensor([
                val_loss_local, float(val_n_local),
                ndcg_sum_local, mrr_sum_local, float(n_ranked_queries_local),
                pos_sum_local, pos_sqsum_local, float(pos_cnt_local),
                neg_sum_local, neg_sqsum_local, float(neg_cnt_local),
            ], device=device, dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            (val_loss_sum, val_n_sum, ndcg_sum, mrr_sum, n_ranked_queries,
             pos_sum, pos_sqsum, pos_cnt,
             neg_sum, neg_sqsum, neg_cnt) = [x.item() for x in t]
            pos_mean = pos_sum / max(pos_cnt, 1)
            pos_std = (pos_sqsum / max(pos_cnt, 1) - pos_mean*pos_mean) ** 0.5
            neg_mean = neg_sum / max(neg_cnt, 1)
            neg_std = (neg_sqsum / max(neg_cnt, 1) - neg_mean*neg_mean) ** 0.5
            avg_val = val_loss_sum / max(val_n_sum, 1)
        else:
            avg_val = val_loss_local / max(val_n_local, 1)
            ndcg_sum = ndcg_sum_local
            mrr_sum = mrr_sum_local
            n_ranked_queries = n_ranked_queries_local
            pos_arr_full = np.array(pos_scores_local)
            neg_arr_full = np.array(neg_scores_local)
            pos_mean = float(pos_arr_full.mean()) if len(pos_arr_full) else 0.0
            pos_std = float(pos_arr_full.std()) if len(pos_arr_full) else 0.0
            neg_mean = float(neg_arr_full.mean()) if len(neg_arr_full) else 0.0
            neg_std = float(neg_arr_full.std()) if len(neg_arr_full) else 0.0

        if is_main:
            gap = pos_mean - neg_mean
            ndcg = ndcg_sum / max(n_ranked_queries, 1)
            mrr = mrr_sum / max(n_ranked_queries, 1)
            logger.info(f"Fold {fold_id} {epoch_tag} | Val Loss={avg_val:.4f} | "
                        f"NDCG@3={ndcg:.4f} MRR={mrr:.4f} | "
                        f"Pos={pos_mean:.4f}±{pos_std:.4f} "
                        f"Neg={neg_mean:.4f}±{neg_std:.4f} Gap={gap:.4f}")
            if save_ckpt:
                metric_val = {"val_loss": avg_val, "ndcg": ndcg, "mrr": mrr}[args.save_best_by]
                if _is_better(metric_val, best_metric):
                    best_metric = metric_val
                    save_checkpoint(model, optimizer, scheduler,
                                    epoch_for_ckpt, global_step,
                                    best_metric, output_dir, tag="best",
                                    best_by=args.save_best_by)
        model.train()
        if args.distributed:
            # Broadcast best_metric from rank 0 so non-main ranks stay in sync on resume.
            t = torch.tensor([best_metric], device=device, dtype=torch.float64)
            dist.broadcast(t, src=0)
            best_metric = float(t.item())
            dist.barrier()
        return avg_val

    # Initial eval at step 0 (only if not resuming mid-training)
    if start_step == 0 and start_epoch == 0:
        if is_main:
            logger.info("Running initial eval at step 0 ...")
        run_eval("Step0", global_step=0, save_ckpt=False)

    global_step = start_step
    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0
        n_batches = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            # Early progress logging (before log_every kicks in) — catches DDP/IO hangs
            if is_main and global_step < 10:
                logger.info(f"[early] step {global_step} batch_idx={batch_idx} "
                            f"batch_size={batch['input_ids'].shape[0]} "
                            f"t_since_start={time.time()-t0:.1f}s")
                sys.stderr.flush()

            # Skip already-done steps on resume
            if epoch == start_epoch and batch_idx < start_step % len(train_loader):
                continue

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            modes = batch["train_mode"]

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits = model(input_ids, attention_mask)
                dpq = 1 + args.neg_per_pos  # docs per query
                loss, l_calib, l_rank = loss_fn(logits, labels, modes, docs_per_query=dpq)
                # Save scores before backward frees the graph
                batch_scores_detached = torch.sigmoid(logits.detach())

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if is_main and global_step % args.log_every == 0:
                avg = epoch_loss / n_batches
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                dpq = 1 + args.neg_per_pos
                samples_per_sec = n_batches * args.batch_size * dpq / elapsed
                if args.distributed:
                    samples_per_sec *= dist.get_world_size()
                # Quick score gap on current batch
                with torch.no_grad():
                    n_q = batch_scores_detached.shape[0] // dpq
                    if n_q > 0:
                        reshaped = batch_scores_detached[:n_q*dpq].view(n_q, dpq)
                        pos_s = reshaped[:, 0].mean().item()
                        neg_s = reshaped[:, 1:].mean().item()
                        gap_str = f"pos={pos_s:.4f} neg={neg_s:.4f} gap={pos_s-neg_s:.4f}"
                    else:
                        gap_str = ""
                logger.info(f"Fold {fold_id} Epoch {epoch} Step {global_step} | "
                            f"Loss={avg:.4f} (calib={l_calib:.4f} rank={l_rank:.4f}) | "
                            f"LR={lr:.2e} | {samples_per_sec:.0f} samples/s | {gap_str}")
            # Save checkpoint periodically
            if is_main and global_step % args.save_every == 0:
                save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                               best_metric, output_dir, tag="latest",
                               best_by=args.save_best_by)

            # Intra-epoch eval.
            if args.eval_every_steps > 0 and global_step % args.eval_every_steps == 0:
                run_eval(f"Step {global_step}", global_step=global_step,
                         save_ckpt=True, epoch_for_ckpt=epoch)

            if args.max_steps and global_step >= args.max_steps:
                if is_main:
                    logger.info(f"Reached --max_steps={args.max_steps}, stopping early.")
                break

        if args.max_steps and global_step >= args.max_steps:
            break

        # End of epoch: validate with ranking metrics (distributed).
        # run_eval handles logging + checkpoint-best selection by --save_best_by.
        run_eval(f"Epoch {epoch}", global_step=global_step,
                 save_ckpt=True, epoch_for_ckpt=epoch)

        if is_main:
            # Save per-epoch checkpoint (never overwritten) + latest
            save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                           best_metric, output_dir, tag=f"epoch{epoch}",
                           best_by=args.save_best_by)
            save_checkpoint(model, optimizer, scheduler, epoch + 1, global_step,
                           best_metric, output_dir, tag="latest",
                           best_by=args.save_best_by)

        # Barrier so non-main ranks wait for rank 0 checkpoint write
        if args.distributed:
            dist.barrier()

    if args.distributed:
        dist.destroy_process_group()

    if is_main:
        logger.info(f"Fold {fold_id} done. Best {args.save_best_by}: {best_metric:.4f}")


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None,
                        help="Path to training JSONL file (group-per-line format)")
    parser.add_argument("--index", default=None,
                        help="Path to line-offset index JSON")
    parser.add_argument("--tokenized_dir", default=None,
                        help="Dir of pre-tokenized parquet shards (preferred). "
                             "Output of scripts/pretokenize_critic.py")
    parser.add_argument("--output_dir", default="./checkpoints/critic")
    parser.add_argument("--backbone", default="microsoft/deberta-v3-large")
    parser.add_argument("--fold", type=int, default=None, help="Which fold (0-4). None=95/5 split")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=0,
                        help="If > 0, stop training after this many global steps (smoke test).")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--neg_per_pos", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--eval_every_steps", type=int, default=0,
                        help="If > 0, run validation every N global steps (in addition to epoch-end).")
    parser.add_argument("--save_best_by", default="ndcg",
                        choices=["val_loss", "ndcg", "mrr"],
                        help="Metric that determines checkpoint-best. "
                             "'val_loss': save when val_loss decreases. "
                             "'ndcg' / 'mrr' (recommended): save when ranking metric increases. "
                             "Val loss and ranking metric can diverge during late training "
                             "(calibration drift vs ranking improvement); retrieval quality is "
                             "driven by ranking metric, so this defaults to ndcg.")
    parser.add_argument("--val_max_batches", type=int, default=0,
                        help="If > 0, cap validation at this many batches (for fast iteration).")
    parser.add_argument("--amp_dtype", default="bf16",
                        help="Mixed precision dtype: 'bf16' (A100/H100) or 'fp16' (V100).")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_dir", default=None, help="Dir with eval sets for test-during-training")
    parser.add_argument("--val_frac", type=float, default=0.05, help="Held-out qi fraction when --fold not set")
    parser.add_argument("--distributed", action="store_true")
    args = parser.parse_args()

    # Auto-detect distributed
    if "LOCAL_RANK" in os.environ:
        args.distributed = True

    is_main = (not args.distributed) or (int(os.environ.get("RANK", 0)) == 0 and
               int(os.environ.get("LOCAL_RANK", 0)) == 0)

    # ──────────────────────────────────────────────
    # Build all_qis (for val split) from either source
    # ──────────────────────────────────────────────
    if args.tokenized_dir:
        # Read keys from parquet shards (fast: only one column)
        import pyarrow.parquet as pq
        shard_files = sorted(glob.glob(os.path.join(args.tokenized_dir, "shard-*.parquet")))
        if not shard_files:
            raise FileNotFoundError(f"No shard-*.parquet found in {args.tokenized_dir}")
        if is_main:
            logger.info(f"Found {len(shard_files)} parquet shards in {args.tokenized_dir}")
        t0 = time.time()
        all_keys = []
        for sf in shard_files:
            tbl = pq.read_table(sf, columns=["key"])
            all_keys.extend(tbl.column("key").to_pylist())
        if is_main:
            logger.info(f"Loaded {len(all_keys):,} group keys in {time.time()-t0:.1f}s")
        all_qis = sorted({int(k.split("_", 1)[0]) for k in all_keys})
    elif args.data:
        groups_path = args.data
        index_path = args.index or os.path.join(os.path.dirname(groups_path), "train_index.json")
        if is_main:
            logger.info(f"Loading index from {index_path} ...")
        t0 = time.time()
        with open(index_path) as f:
            all_index = json.load(f)
        if is_main:
            logger.info(f"Loaded {len(all_index):,} groups in {time.time()-t0:.1f}s")
        def qi_of(entry):
            return int(entry["key"].split("_", 1)[0])
        all_qis = sorted({qi_of(e) for e in all_index})
    else:
        raise ValueError("Provide --tokenized_dir (preferred) or --data (jsonl)")

    rng = random.Random(42)
    rng.shuffle(all_qis)

    if args.fold is not None:
        folds = [[] for _ in range(args.n_folds)]
        for i, qi in enumerate(all_qis):
            folds[i % args.n_folds].append(qi)
        val_qis = set(folds[args.fold])
    else:
        n_val = max(1, int(args.val_frac * len(all_qis)))
        val_qis = set(all_qis[:n_val])

    if is_main:
        logger.info(f"Split by qi: {len(all_qis)-len(val_qis):,} train qis / {len(val_qis):,} val qis")

    # ──────────────────────────────────────────────
    # Build datasets
    # ──────────────────────────────────────────────
    if args.tokenized_dir:
        train_ds = CriticParquetDataset(shard_files, neg_per_pos=args.neg_per_pos,
                                        val_qis=val_qis, mode="train")
        val_ds = CriticParquetDataset(shard_files, neg_per_pos=args.neg_per_pos,
                                      val_qis=val_qis, mode="val",
                                      _shared_state=train_ds._shared_state)
    else:
        # JSONL path
        tokenizer = AutoTokenizer.from_pretrained(args.backbone)
        def qi_of(entry):
            return int(entry["key"].split("_", 1)[0])
        train_idx = [e for e in all_index if qi_of(e) not in val_qis]
        val_idx = [e for e in all_index if qi_of(e) in val_qis]
        train_ds = CriticDataset(train_idx, args.data, tokenizer, neg_per_pos=args.neg_per_pos)
        val_ds = CriticDataset(val_idx, args.data, tokenizer, neg_per_pos=args.neg_per_pos)

    if is_main:
        logger.info(f"Train: {len(train_ds):,} positives, Val: {len(val_ds):,} positives")

    train_fold(args.fold or 0, train_ds, val_ds, args)


if __name__ == "__main__":
    main()
