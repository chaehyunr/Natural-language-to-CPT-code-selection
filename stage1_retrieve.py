"""
Stage 1 — Retrieval (paper, "Retrieval").

Scores every CPT code against the query and keeps the top K_retrieve = 30 as C(q).

Methods (names are used as the `method` column downstream):
  bge_dense, mxbai_dense, clinicalbert_dense   dense cosine similarity (Eq 2)
  bm25_sparse                                  lexical BM25
  hybrid_mxbai_bm25_rrf                        mxbai + BM25 fused by RRF (Eq 3)

Usage:
  python stage1_retrieve.py                            # all five methods (Table 2)
  python stage1_retrieve.py --methods mxbai_dense      # pipeline retriever only

Outputs (save_dir):
  predictions.csv       one row per (method, query); retrieved_codes ranked
  per_query_metrics.csv
  summary_metrics.csv   Recall / Precision / Leakage / MAP @ K   (Table 2, Retriever rows)

Changes from the original code
(SunnyBrook/Final/retrievers.py + run_retrieval_eval.py, merged into this file)
------------------------------------------------------------------------------
[changed] excluded_codes is now carried in every prediction row (base_record). The
          original dropped it, so leakage had to be recomputed afterwards by joining
          the dataset on query text. Retrieved codes are unchanged.
[changed] --run_methods / --dense_models / --sparse_models / --hybrid_pairs replaced by
          one --methods list. The default runs the five methods of Table 2.
[changed] Metrics are evaluated at K = 30 only (paper: "Retrieval is evaluated at K = 30");
          the original evaluated K = 10 and 30.
[changed] Output file all_predictions.csv -> predictions.csv.
[removed] Convex-combination fusion and score threshold (not in the paper), SPLADE
          pairs, semantic/distance metrics, column defaults of the old dataset.
"""

from __future__ import annotations

import argparse
import atexit
from pathlib import Path

import pandas as pd

from build_index import load_dense_encoder, load_sparse_encoder
from config import (
    DB_DIR, DENSE_MODELS, EXCLUDED_COL, HAS_EXCLUSION_COL, INCLUDED_COL, K_RETRIEVE,
    MAX_TRUE_CODES, PREFETCH_LIMIT, QUERY_COL, RESULTS_DIR, RRF_K, VAL_DATA,
)
from metrics import ensure_code_list, evaluate, normalize_cpt_code, save_outputs

ALL_METHODS = ["bge_dense", "mxbai_dense", "clinicalbert_dense", "bm25_sparse", "hybrid_mxbai_bm25_rrf"]


# ---------------------------------------------------------------------
# Evaluation data
# ---------------------------------------------------------------------

# [changed] from retrievers.load_eval_data(); same filter, columns fixed to the validation set
def load_queries(path=VAL_DATA, max_true_codes: int | None = MAX_TRUE_CODES,
                 max_rows: int | None = None) -> pd.DataFrame:
    """Load the query set; drop queries with more than `max_true_codes` ground-truth codes (145 -> 127)."""
    df = pd.read_csv(path)
    if max_true_codes is not None:
        n_true = df[INCLUDED_COL].apply(lambda x: len(ensure_code_list(x, deduplicate=True)))
        df = df[n_true <= max_true_codes].reset_index(drop=True)
    if max_rows is not None:
        df = df.head(max_rows)
    print(f"[data] {path}: {len(df)} queries")
    return df


def build_query(row: pd.Series) -> str:
    """The query string passed to every stage (lower-cased)."""
    return str(row.get(QUERY_COL, "")).strip().lower()


def base_record(row: pd.Series) -> dict:
    """
    Query-level fields carried through every stage.
    [changed] from retrievers.make_result_row(); excluded_codes added (Eq 8 needs X(q)).
    """
    return {
        "id": row.get("ID", ""),
        "category": row.get("Category", ""),
        "complexity": row.get("Complexity", ""),
        "has_exclusion": row.get(HAS_EXCLUSION_COL, ""),
        "query": build_query(row),
        "answer_codes": ensure_code_list(row.get(INCLUDED_COL), deduplicate=True),
        "excluded_codes": ensure_code_list(row.get(EXCLUDED_COL), deduplicate=True),   # [added]
    }


# ---------------------------------------------------------------------
# Retrievers
# ---------------------------------------------------------------------

def _to_codes(hits) -> list[tuple[str, float]]:
    out = []
    for h in hits:
        code = normalize_cpt_code((h.payload or {}).get("cpt_code"))
        if code is not None:
            out.append((code, float(h.score)))
    return out


class DenseRetriever:
    """Eq 2: cosine similarity between the query embedding and stored code embeddings."""
    def __init__(self, client, alias: str, encoder, top_k: int = K_RETRIEVE):
        self.client, self.alias, self.encode, self.top_k = client, alias, encoder, top_k

    def search(self, query: str, limit: int | None = None) -> list[tuple[str, float]]:
        hits = self.client.query_points(collection_name=self.alias, query=self.encode(query),
                                        using="dense", limit=limit or self.top_k).points
        return [(c, round(s, 4)) for c, s in _to_codes(hits)]


class SparseRetriever:
    def __init__(self, client, alias: str, encoder, top_k: int = K_RETRIEVE):
        self.client, self.alias, self.encode, self.top_k = client, alias, encoder, top_k

    def search(self, query: str, limit: int | None = None) -> list[tuple[str, float]]:
        hits = self.client.query_points(collection_name=self.alias, query=self.encode(query),
                                        using="sparse", limit=limit or self.top_k).points
        return [(c, round(s, 4)) for c, s in _to_codes(hits)]


class RRFRetriever:
    """
    Eq 3: RRF(c) = Σ_m 1 / (k + rank_m(c)), k = 60. A code absent from one list adds
    nothing for that list. Each source contributes its top PREFETCH_LIMIT (60) codes.
    [changed] replaces HybridRetriever(fusion_method="rrf"); the "cc" branch was removed.
    """

    def __init__(self, dense: DenseRetriever, sparse: SparseRetriever, top_k: int = K_RETRIEVE,
                 prefetch_limit: int = PREFETCH_LIMIT, rrf_k: int = RRF_K):
        self.dense, self.sparse = dense, sparse
        self.top_k, self.prefetch_limit, self.rrf_k = top_k, prefetch_limit, rrf_k

    def search(self, query: str) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}
        for ranking in (self.dense.search(query, self.prefetch_limit),
                        self.sparse.search(query, self.prefetch_limit)):
            for rank, (code, _) in enumerate(ranking, start=1):
                scores[code] = scores.get(code, 0.0) + 1.0 / (self.rrf_k + rank)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: self.top_k]
        return [(c, round(s, 6)) for c, s in ranked]


def build_retrievers(client, methods: list[str], top_k: int) -> dict:
    encoders: dict = {}

    def enc(alias):
        if alias not in encoders:
            print(f"[load] {alias}")
            encoders[alias] = load_dense_encoder(alias) if alias in DENSE_MODELS else load_sparse_encoder(alias)
        return encoders[alias]

    out = {}
    for m in methods:
        if m.endswith("_dense"):
            a = m[: -len("_dense")]
            out[m] = DenseRetriever(client, a, enc(a), top_k)
        elif m.endswith("_sparse"):
            a = m[: -len("_sparse")]
            out[m] = SparseRetriever(client, a, enc(a), top_k)
        elif m.startswith("hybrid_") and m.endswith("_rrf"):
            d, s = m[len("hybrid_"): -len("_rrf")].split("_")
            out[m] = RRFRetriever(DenseRetriever(client, d, enc(d), top_k),
                                  SparseRetriever(client, s, enc(s), top_k), top_k)
        else:
            raise ValueError(f"Unknown method: {m}. Available: {ALL_METHODS}")
    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Stage 1: retrieve CPT candidates.")
    ap.add_argument("--data", default=str(VAL_DATA))
    ap.add_argument("--save_dir", default=str(RESULTS_DIR / "retrieval"))
    ap.add_argument("--methods", default=",".join(ALL_METHODS))
    ap.add_argument("--k", type=int, default=K_RETRIEVE, help="candidates kept per query (K_retrieve)")
    ap.add_argument("--max_rows", type=int, default=None, help="quick test")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    queries = load_queries(args.data, max_rows=args.max_rows)

    from qdrant_client import QdrantClient
    client = QdrantClient(path=str(DB_DIR))
    atexit.register(client.close)
    retrievers = build_retrievers(client, methods, args.k)

    rows = []
    for name, retriever in retrievers.items():
        print(f"[run] {name}")
        for _, r in queries.iterrows():
            rec = base_record(r)
            hits = retriever.search(rec["query"])
            rows.append({"method": name, **rec,
                         "retrieved_codes": [c for c, _ in hits],
                         "retrieved_scores": dict(hits)})

    pred = pd.DataFrame(rows)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(save_dir / "predictions.csv", index=False)
    print(f"[SAVED] {save_dir / 'predictions.csv'}")

    per_query, summary = evaluate(pred, k_list=[args.k])
    save_outputs(per_query, summary, save_dir)


if __name__ == "__main__":
    main()
