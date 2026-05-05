# CoRM-RAG

Code for **"Beyond Semantic Relevance: Counterfactual Risk Minimization for Robust Retrieval-Augmented Generation"**.

CoRM-RAG aligns retrieval with decision safety rather than mere semantic similarity. We introduce a **Cognitive Perturbation Protocol** to simulate user biases during training, distilled into a lightweight **Evidence Critic** that scores documents by their robustness to query perturbations.

## Repository Layout

```
src/
  encode_wikipedia.py              # Encode Wikipedia passages with Contriever; build FAISS index
  gen_perturbations_api.py         # Generate cognitive perturbations of queries via OpenAI-compatible API
  gen_perturbations_distributed.py # Multi-node multi-GPU perturbation generation with vLLM
  retrieve_perturbed_distributed.py# Retrieve top-K passages for each (query, perturbation) pair
  teacher_evaluation.py            # Teacher LLM evaluation over (query, doc, perturbation) triples
  train_critic.py                  # Train the Evidence Critic (DeBERTa backbone, listwise loss)
  run_evaluation.py                # End-to-end evaluation: retrieve -> rerank with critic -> answer
  run_*.sh                         # Driver scripts for each stage

scripts/
  preprocess_training_data.py      # Build training groups from teacher-evaluation outputs
  build_teacher_pool.py            # Merge clean/perturbed retrieval candidates for teacher scoring
  build_train_expanded.py          # Construct paper-style listwise training records
  build_and_slice.py               # Shard training data for distributed training
  pretokenize_critic.py            # Pre-tokenize training data for fast loading
```

## Pipeline

1. **Index Wikipedia**: `python src/encode_wikipedia.py`
2. **Generate perturbations**: `bash src/run_distributed.sh` (or `src/gen_perturbations_api.py` for API-based).
3. **Retrieve perturbed**: `bash src/run_retrieve_perturbed.sh`
4. **Teacher evaluation**: `python scripts/build_teacher_pool.py && RETRIEVAL_FILE_NAME=retrieval_teacher_pool.jsonl bash src/run_teacher.sh`
5. **Build training data**: `python scripts/build_train_expanded.py && python scripts/preprocess_training_data.py && python scripts/pretokenize_critic.py`
6. **Train critic**: `bash src/run_train_critic.sh` (or use the released checkpoint below).
7. **Evaluate**: `CRITIC_PATH=/path/to/state.pt bash src/run_eval.sh`

## Environment

```
torch >= 2.1
transformers >= 4.40
faiss-gpu
vllm >= 0.5
openai >= 1.30
datasets
huggingface_hub (for downloading the released checkpoint)
numpy, scipy, scikit-learn
```

Set the following environment variables as needed:

| Var | Purpose |
| --- | --- |
| `DATA_SRC` | Root of the data directory (default `./data`) |
| `OUTPUT_DIR` | Where outputs are written (default `./output`) |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` | OpenAI-compatible API endpoint for perturbation generation |
| `BACKBONE_PATH` | Critic backbone (default `microsoft/deberta-v3-large`) |
| `CRITIC_PATH` | Direct path to an Evidence Critic checkpoint `state.pt` |
| `CRITIC_CKPT_DIR` | Critic checkpoint directory used at evaluation time |

## Data

Released code targets one decision-making benchmark; other benchmarks reported in the paper follow the same pipeline. Place input files under `${DATA_SRC}` (default `./data`).

## Released Checkpoint

The released Evidence Critic checkpoint is hosted on Hugging Face:

<https://huggingface.co/PeiyangLiu/CoRM-RAG>

Download it with:

```bash
huggingface-cli download PeiyangLiu/CoRM-RAG \
  critic-v12-mixed/checkpoint-latest/state.pt \
  --local-dir checkpoints/hf
```

Then evaluate with:

```bash
CRITIC_PATH=checkpoints/hf/critic-v12-mixed/checkpoint-latest/state.pt bash src/run_eval.sh
```

## Citation

If you use CoRM-RAG, please cite:

```bibtex
@misc{liu2026cormrag,
  title={Beyond Semantic Relevance: Counterfactual Risk Minimization for Robust Retrieval-Augmented Generation},
  author={Peiyang Liu and Qiang Yan and Ziqiang Cui and Di Liang and Xi Wang and Wei Ye},
  year={2026},
  eprint={2605.01302},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2605.01302}
}
```

## License

Released for academic research use.
