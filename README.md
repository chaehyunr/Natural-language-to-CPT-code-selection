# Natural language to CPT code selection

Code and data for *Natural language to CPT code selection: A multi-stage framework for clinical research queries*.

Given a natural-language cohort query (e.g. *"all shoulder arthroplasty, excluding revision"*), the pipeline returns the set of CPT codes that defines the cohort:

```
query ──► Retrieve ──► Rerank ──► LLM Select ──► S(q)
          mxbai          Qwen3-Reranker-4B   Llama3-8B, label ≥ τ (τ = 3)
          top 30         top 20
```

Every returned code comes from the indexed CPT corpus, so the pipeline cannot output a code that does not exist.

## Repository layout

| File | Paper section | Role |
|---|---|---|
| `config.py` | Experimental setup | Paths, model names, K_retrieve = 30, K_rerank = 20, RRF k = 60, τ = 3, seeds |
| `build_index.py` | CPT code corpus | Embeds the 8,745 CPT descriptions into a local Qdrant index |
| `stage1_retrieve.py` | Retrieval (Eqs 2–3) | bge / mxbai / ClinicalBERT (dense), BM25 (sparse), mxbai + BM25 (RRF) |
| `stage2_rerank.py` | Reranking (Eq 4) | BGE-reranker-v2-m3, MedCPT cross-encoder, Qwen3-Reranker-4B |
| `stage3_select.py` | LLM selection (Eqs 5–6, Table 1) | 3/2/1 labeling prompt, label parser, threshold τ |
| `baseline_direct_llm.py` | Direct LLM baseline | LLM generates 20 candidates, then labels them with the stage-3 prompt |
| `metrics.py` | Evaluation metrics (Eqs 7–9) | Recall, Precision, Exclusion leakage, MAP |
| `aggregate_runs.py` | Experimental setup | Mean (SD) over the three seeds |
| `data/cpt_query_validation_dataset.csv` | Evaluation dataset | 145 physician-authored queries (127 used, see below) |

## Setup

```bash
pip install -r requirements.txt
```

All experiments in the paper were run on a single NVIDIA H100 GPU with Python 3.10. Retrieval runs on CPU; reranking with Qwen3-Reranker-4B and LLM selection need a GPU.

The selection models are gated on Hugging Face (`meta-llama/Meta-Llama-3-8B-Instruct`, `Qwen/Qwen3-8B`). Either log in with `huggingface-cli login`, or pass local copies with `--llama3_path` / `--qwen3_path`.

## Data

**Evaluation dataset** — `data/cpt_query_validation_dataset.csv`

| Column | Content |
|---|---|
| `ID`, `Query`, `Category`, `Complexity` | query metadata |
| `Included CPT Codes` | ground-truth codes G(q) |
| `Excluded CPT Codes` | explicitly excluded codes X(q) |
| `Has Exclusion` | `Y` if the query carries exclusion criteria |
| `Notes` | annotator rationale |

The file holds all 145 authored queries. Queries with more than 10 ground-truth codes (18) are removed at load time (`MAX_TRUE_CODES` in `config.py`), leaving the 127 queries reported in the paper, 9 of which carry exclusion criteria covering 30 excluded codes.

**CPT corpus** — not distributed. CPT descriptions are copyrighted by the American Medical Association. Place a spreadsheet at `data/web_cpt_codes.xlsx` with columns `code`, `system`, `display`; rows with `system == "CPT4"` are used (8,745 codes in our release). See `data/README.md`.

## Reproducing the results

```bash
# once: build the Qdrant index (db/, 4 collections)
python build_index.py

# Table 2 — Retriever rows (K = 30)
python stage1_retrieve.py --save_dir results/retrieval

# Table 2 — Reranker rows (K = 20, reranking the mxbai candidates)
python stage2_rerank.py --predictions results/retrieval/predictions.csv --save_dir results/rerank

# Table 2 — LLM rows (three seeds, then mean (SD))
for s in 0 1 2; do
  python stage3_select.py --predictions results/rerank/predictions.csv --seed $s --save_dir results/selection/run_$s
done
python aggregate_runs.py results/selection/run_0 results/selection/run_1 results/selection/run_2

# Direct LLM baseline (Table 3, hallucination rate)
for s in 0 1 2; do
  python baseline_direct_llm.py --seed $s --save_dir results/direct_llm/run_$s \
      --deleted_codes data/deleted_cpt_codes.txt   # optional, see "Hallucination check"
done
python aggregate_runs.py results/direct_llm/run_0 results/direct_llm/run_1 results/direct_llm/run_2
```

Add `--max_rows 3` to any stage for a quick check.

## Outputs

Each stage writes to its `--save_dir`:

| File | Content |
|---|---|
| `predictions.csv` | one row per (method, query). `retrieved_codes` is the stage output; stage 3 also stores `pre_selection_codes` (the 20 reranked candidates) and `label_map` (code → label) |
| `per_query_metrics.csv` | metrics for every query |
| `summary_metrics.csv` | macro-averaged `recall`, `precision`, `leakage`, `map` (ranked stages only), `mean_n_pred`; the direct LLM baseline adds `hallucination_rate` |

`method` names: `mxbai_dense`, `hybrid_mxbai_bm25_rrf`, … (stage 1), `mxbai_dense__Qwen3-4B` (stage 2), `mxbai_dense__Qwen3-4B__sel_llama3` (stage 3), `direct_llm__llama3` (baseline).

## Evaluation details

- All metrics are computed per query and macro-averaged.
- Retrieval and reranking are evaluated on the top-K list (precision denominator K). Selection and the direct LLM baseline are evaluated on the returned set (precision denominator |P(q)|; an empty set counts as precision 0).
- Leakage = |X(q) ∩ P(q)| / |X(q)|, averaged over the 9 queries with exclusion criteria only (`n_exclusion_queries`).
- MAP is reported for retrieval and reranking only, since the selection output is unordered.
- **Hallucination check** (direct LLM baseline). A generated code is hallucinated if it does not have the CPT format (five digits, or four digits followed by F, T, U, M or A) or is absent from the reference set of current CPT codes plus codes deleted in past releases. Pass the deleted-code list with `--deleted_codes`; without it, the reference set is the current corpus only. `hallucination_rate` is the share of generated codes that are hallucinated, pooled over the queries of a run.

## Implementation notes

- **Determinism.** Retrieval and reranking are deterministic. Selection samples at temperature 0.7 with seeds 0, 1, 2 (`transformers.set_seed` once per model load).
- **Prompt variant.** The exclusion-aware prompt (Table 1) is used when the query's `Has Exclusion` annotation is `Y`; otherwise the base prompt is used. The clause shown to the model is extracted from the query text by pattern matching; for the one exclusion query without an explicit clause ("all primary major joint arthroplasty …"), the field reads `see query`.
- **Offline thresholding.** The GPU run stores only the label map; `--tau_list 3,2` evaluates several thresholds from the same run. The paper reports τ = 3.
- **Unlabeled candidates.** Candidates the LLM leaves out of its JSON answer are assigned label 2 (withheld at τ = 3).
- Prompts are truncated to 4,096 tokens and generation is limited to 512 new tokens.

## Provenance

Each source file starts with a *Changes from the original code* section listing what was changed, added or removed relative to the code that produced the paper's results, with the paper section each change follows. Inline `[changed]` / `[added]` comments mark the corresponding lines.

## Citation

```
TBD
```
