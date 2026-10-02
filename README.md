# Natural language to CPT code selection

Code for *Natural language to CPT code selection: A multi-stage framework for clinical research queries*.

Given a natural-language cohort query, the pipeline returns the set of CPT codes that defines the cohort.

## Pipeline

```
CPT corpus (8,745 codes)
    │
    ▼
[0] build_index.py          Embed every CPT description → local Qdrant index (run once)


Cohort query  e.g. "all shoulder arthroplasty, excluding revision"
    │
    ▼
[1] stage1_retrieve.py      mxbai-embed-large-v1 dense retrieval → top 30 candidates
    │
    ▼
[2] stage2_rerank.py        Qwen3-Reranker-4B scores each (query, description) pair → top 20
    │
    ▼
[3] stage3_select.py        Llama3-8B labels each candidate
    │                         3 = Must Include / 2 = Normal / 1 = Exclude
    │                       returns the codes with label ≥ τ (τ = 3)
    ▼
[4] aggregate_runs.py       Mean (SD) over three runs (seeds 0, 1, 2)


Direct LLM baseline (no retrieval, no reranking)
    │
    ├─[B] baseline_direct_llm.py   Llama3-8B generates 20 codes from the query
    │                              → labels them with the stage-3 prompt and τ
    │                              → flags generated codes that do not exist
    ▼
    aggregate_runs.py
```

## Code

| File | Role |
|---|---|
| `config.py` | Paths, model names and experiment settings (K = 30 / 20, RRF k = 60, τ = 3, seeds, temperature 0.7) |
| `build_index.py` | Builds one Qdrant collection per encoder: bge, mxbai, ClinicalBERT (dense) and BM25 (sparse) |
| `stage1_retrieve.py` | Retrieval with the four encoders and mxbai + BM25 fused by reciprocal rank fusion |
| `stage2_rerank.py` | Reranking with BGE-reranker-v2-m3, MedCPT cross-encoder and Qwen3-Reranker-4B |
| `stage3_select.py` | LLM selection: the two labeling prompts, label parser and threshold τ |
| `baseline_direct_llm.py` | Direct LLM baseline: candidate generation, labeling and hallucination check |
| `metrics.py` | Recall, precision, exclusion leakage and MAP, shared by every stage |
| `aggregate_runs.py` | Mean (SD) of each metric across the three runs |
| `data/cpt_query_validation_dataset.csv` | 145 physician-authored queries with included and excluded CPT codes (127 used) |

The CPT corpus (`data/web_cpt_codes.xlsx`) and the index (`db/`) are not included; see `data/README.md`.

## Requirements

Python 3.10. Reranking with Qwen3-Reranker-4B and LLM selection need a GPU.

```bash
pip install -r requirements.txt
```

```
qdrant-client>=1.10
fastembed>=0.3
sentence-transformers==4.0.2
pandas>=2.0
numpy>=1.24
openpyxl>=3.1
torch>=2.0
transformers>=4.51
accelerate>=1.0
```
