# Running UC-ASR on Kaggle GPU

The local dev box is CPU-only, so full-scale runs with a real LLM happen on Kaggle's free GPU. This guide covers setup, the `kaggle` scale profile, and the session/quota limits you need to plan around.

## Why Kaggle works here
- Free **NVIDIA T4 (16GB) x2** or **P100 (16GB)** with CUDA, so `bitsandbytes` 4-bit quantization runs (impossible on the CPU box).
- Qwen2.5-3B in 4-bit uses ~2-3GB VRAM; Mistral-7B in 4-bit ~5-6GB. Either fits a single T4 comfortably.

## Limits to plan around
- **~30 GPU-hours / week** quota.
- **12-hour** max session (headless "Save & Run All" commits run to completion within that window).
- Interactive sessions also stop if the tab is closed too long.
- **~20GB** working disk under `/kaggle/working` (saved as notebook output).
- **Internet is OFF by default** - toggle it on (phone-verified account required) for pip/data/model downloads, or attach data as a Kaggle Dataset for offline runs.

The permanent SQLite LLM cache (`cache/llm_responses.sqlite`, sha256-keyed) is the key to living within these limits: if a session dies mid-run, the next session resumes from cached reasoning traces instead of recomputing. Save the cache as notebook output and re-attach it to continue.

## Quickstart
1. Open `kaggle_run.ipynb` on Kaggle (upload it, or create a notebook and paste the cells).
2. **Settings -> Accelerator -> GPU T4 x2** (or P100).
3. **Settings -> Internet -> On.**
4. Edit the `REPO_URL` cell to point at your GitHub mirror of this repo (or use the Kaggle-Dataset copy cell).
5. Run all cells. The `rerank` stage is the slow part (real LLM generation); everything else is fast.

## The `kaggle` scale profile
Selected with `--scale kaggle`. It differs from `full` by sizing the LLM workload to one session:

| Knob | `full` | `kaggle` | Why |
|---|---|---|---|
| `llm.backend` | hf | **hf** | real model |
| `llm.hf_model` | Qwen2.5-3B | **Qwen2.5-3B** | fast default; swap to Mistral-7B to compare |
| `llm.load_in_4bit` | true | **true** | CUDA available on Kaggle |
| `preprocessing.k_core` | 5 | **5** | real 5-core |
| `preprocessing.n_users_subset` | 10000 | **10000** | full user target |
| `retrievers.top_k_candidates` | 50 | **50** | full K=50 |
| `uncertainty.uaro.n_passes` | 3 | **3** | N=3 @ T=0.7 |
| `evaluation.kaggle_max_eval_users` | - | **2000** | cap LLM-scored users to fit ~12h |
| `evaluation.kaggle_max_ablation_users` | - | **300** | keep the 10 ablations bounded |

Raise `kaggle_max_eval_users` toward `n_users_subset` across multiple sessions; the cache makes re-runs cheap.

## Comparing Qwen vs Mistral
Edit `scales.kaggle.llm.hf_model` in `config.yaml`:
- `Qwen/Qwen2.5-3B-Instruct` (faster, lighter)
- `mistralai/Mistral-7B-Instruct-v0.2` (stronger, ~2x slower)

Each model writes under its own cache key (the model id is part of the sha256), so the two runs don't collide and you can keep both result sets.

## Offline data (no internet)
1. Create a Kaggle Dataset containing `Grocery_and_Gourmet_Food.jsonl.gz` and `meta_Grocery_and_Gourmet_Food.jsonl.gz`.
2. Attach it to the notebook.
3. `load_config` auto-detects an attached dataset that holds the raw review file and repoints `paths.raw_dir` at it. To pin it explicitly: `os.environ['UC_ASR_KAGGLE_DATASET'] = '<slug>'`.

Note: the HF model download still needs internet unless you also attach the model weights as a dataset.

## Path handling on Kaggle
`load_config` detects the Kaggle kernel and sets `project.root = /kaggle/working`, so all `paths.*` resolve to the writable output dir automatically. Opt out with `UC_ASR_NO_KAGGLE_PATHS=1`.

## What to download when done
After **Save Version**, keep: `results/` (tables, stats, figures), `data/processed/*.parquet`, and `cache/llm_responses.sqlite` (so you can resume or extend without re-spending GPU-hours).
