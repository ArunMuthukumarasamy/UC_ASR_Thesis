# UC-ASR Research Codebase — Agent Guide

**Thesis chapter:** "UC-ASR: Uncertainty-Conditioned Adaptive Semantic Reranking for LLM-Based Recommendation"
**Dataset:** McAuley-Lab Amazon Reviews 2023 — Grocery and Gourmet Food.
**Workspace root:** `d:\UC_ASR_Thesis` (all paths relative to this).

## Environment (verified at setup)
- Windows, PowerShell, Python 3.12.10, pip 25.0.1.
- **CPU-ONLY machine. No NVIDIA GPU. No Ollama installed.**
- Consequences the code MUST handle gracefully:
  - `bitsandbytes` 4-bit quantization requires CUDA → NOT available. Code must detect this and fall back to CPU `float32`/`bfloat16` without crashing.
  - Running Qwen2.5-3B / Mistral-7B on CPU is viable but slow. The LLM engine MUST be fully functional, but the pipeline MUST also support a `smoke`/`demo` scale (small user subset, fewer candidates) so the whole pipeline runs end-to-end quickly for verification, AND a `full` scale for the real experiment.
  - Everything LLM-related MUST be cached in `cache/llm_responses.sqlite` so reruns are cheap.
  - The pipeline must never hard-require a GPU or Ollama; detect-and-fallback everywhere.

## Folder structure (already created)
```
data/raw/            # downloaded .jsonl.gz
data/processed/      # parquet splits, candidates, stats
cache/               # llm_responses.sqlite + model caches
src/                 # the 6 modules
results/             # tables, stats, ablations
results/figures/     # 300-DPI PNGs
config.yaml          # master config (already written — read it, honor it)
requirements.txt     # already written
```

## Modules to implement (exact filenames)
1. `src/data_pipeline.py` — download (requests+tqdm streaming, HF datasets fallback), stream-parse (RAM<8GB), deterministic 10k-user subset, 5-core filter, chronological Leave-One-Out split → `train/val/test.parquet`, write `data/processed/dataset_stats.json`.
2. `src/retrievers.py` — Popularity, BPR-MF (dim=64, 50 ep), LightGCN (3 layers, dim=64, 50 ep), FAISS `IndexFlatIP` over LightGCN embeddings → top-50 candidates `data/processed/candidates_top50.parquet` with normalized retrieval confidence `E_{u,i}` in [0,1]; SBERT (`all-MiniLM-L6-v2`) reranker over top-50.
3. `src/llm_engine.py` — Think-Before-Recommend prompt engine; HF local (4bit/bfloat16 with CPU fallback) OR Ollama API; SQLite cache in `cache/llm_responses.sqlite` storing every reasoning trace + relevance score (1.0–5.0) permanently keyed by (model, prompt, temperature, seed/pass).
4. `src/uncertainty.py` —
   - UARO Score Variance: N=3 passes @ T=0.7 → mean `R`, sample variance `σ²`, confidence `c = 1/(1+σ²)`, normalized uncertainty `U`.
   - ATM Semantic Cluster Ratio `CR = max_m |s_m| / N`: greedy clustering over N traces using bidirectional NLI entailment / SBERT sim ≥ 0.85.
   - ATM Bayesian top-k threshold `t(k)`: `x_i ~ N(μ_i, σ_i²+β²)`, solve `Σ P(x_i>t(k)) = k` (k=10) by binary search; boundary prob `p_i^{top-k} = P(x_i>t(k))`.
5. `src/rerankers.py` — on the SAME top-50: (a) Single-pass LLM (N=1,T=0.0); (b) UARO (N=3, `score = R - λU + γE`); (c) ATM-style adaptive (uncertain set `ε < p < 1-ε`, listwise batches m=10); (d) **UC-ASR**: fuse `c, CR`, top-k boundary certainty → reliability `A_{u,i}`; 3-tier gate (Confident `A≥τ_high` freeze/0 calls; Ambiguous `τ_low≤A<τ_high` selective listwise m=10; Unstable `A<τ_low` +2 deep samples w/ full metadata); `FinalScore = αR + βCR + δE − γU` (tuned on val).
6. `src/evaluate.py` + `run_pipeline.py` — 3 seeds (42,100,2026); `results/thesis_main_table.md`+`.csv` (cols: Method | Recall@10 | NDCG@10 | MRR | Avg. LLM calls | Latency | Stability | CR) for all 8 methods; 10 ablation studies → `results/ablation_results.md`; paired Wilcoxon signed-rank → `results/statistical_significance.md`; 4 publication-ready 300-DPI white-background figures in `results/figures/`.

## Engineering requirements
- Read ALL hyperparameters from `config.yaml`. No magic numbers in code.
- Deterministic: seed numpy/torch/random from config.
- `run_pipeline.py` must expose a CLI with `--scale {smoke,full}` and `--stage {data,retrieve,rerank,eval,all}` so stages can run independently and a fast smoke run proves correctness end-to-end.
- Keep RAM < 8GB in the data stage (stream, don't load everything).
- Every module runnable standalone (`python -m src.<module>`) and importable.
- Log progress clearly; fail loudly with actionable messages when a download/model is unavailable.

## Verification bar
- `pip install -r requirements.txt` succeeds (or documents any CPU-only skips).
- `python run_pipeline.py --scale smoke --stage all` runs end-to-end on a tiny subset and produces the tables, stats JSON, and at least placeholder figures WITHOUT a GPU and WITHOUT Ollama (using a small HF model on CPU, or a deterministic mock LLM if even that is too slow — but the real HF path must exist and be selectable).
- All output files land in the exact paths named above.
