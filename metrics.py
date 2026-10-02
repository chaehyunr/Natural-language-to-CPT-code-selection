"""
Evaluation metrics (paper, "Evaluation metrics", Eqs 7-9, and "Direct LLM baseline").

All metrics are computed per query and then macro-averaged across queries.

  Eq 7  Recall@K(q)    = |G(q) ∩ P(q)| / |G(q)|
        Precision@K(q) = |G(q) ∩ P(q)| / |P(q)|
  Eq 8  Leakage@K(q)   = |X(q) ∩ P(q)| / |X(q)|,   defined only when X(q) ≠ ∅ (n = 9 queries)
  Eq 9  MAP@K(q)       = 1 / min(|G(q)|, K) · Σ_{i=1..K} Precision@i(q) · 1[c_i ∈ G(q)]

G = ground-truth codes, X = excluded codes, P = codes returned by the stage being evaluated.
"For retrieval and reranking, P(q) is the top-K list, and for selection and the direct
LLM baseline, it is the full returned set." MAP is reported for retrieval and reranking
only, because the selection output is an unordered set.

Changes from the original code (SunnyBrook/Final/metrics.py)
-----------------------------------------------------------
[changed] Precision denominator is |P(q)| in both modes (Eq 7). The original divided by K
          for ranked lists. Identical in practice: retrieval always returns 30 codes and
          reranking 20, so |P(q)| = K.
[changed] MAP denominator is min(|G(q)|, K) (Eq 9). The original divided by |G(q)|.
          Identical here because |G(q)| <= 10 < K.
[changed] set_based=True/False replaced by k=None (set) / k=int (ranked top-K).
[changed] Output columns renamed: macro_recall -> recall, macro_precision -> precision,
          exclusion_leakage -> leakage, mean_map -> map; added mean_n_pred
          (paper, "Overall performance": 4.96 codes per query).
[added]   Hallucination check for the direct LLM baseline (is_hallucinated).
[removed] Semantic precision/recall/F1, SemDCG, SemnDCG, MACD, EMD, Any-Hit, F1,
          R-Precision, MRR, exclusion_recall, false_exclusion, and the Qdrant vector map
          used by the semantic metrics: none is reported in the paper.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Code parsing (unchanged from the original)
# ---------------------------------------------------------------------

def normalize_cpt_code(x: Any) -> str | None:
    """27130.0 -> "27130", " 0077u " -> "0077U", empty/NaN -> None."""
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s.upper()


def ensure_code_list(x: Any, deduplicate: bool = False) -> list[str]:
    """
    Parse a list, a "27130;27447" / "27130, 27447" string, or a
    CSV-saved "['27130', '27447']" string into a clean code list.
    """
    if x is None:
        return []
    try:
        if pd.isna(x):
            return []
    except (TypeError, ValueError):
        pass

    if isinstance(x, np.ndarray):
        items = x.tolist()
    elif isinstance(x, (list, tuple, set)):
        items = list(x)
    elif isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                items = list(parsed) if isinstance(parsed, (list, tuple, set)) else [parsed]
            except (SyntaxError, ValueError):
                items = re.split(r"[;,]", s.strip("[]"))
        else:
            items = re.split(r"[;,]", s)
    else:
        items = [x]

    codes = [c for c in (normalize_cpt_code(i) for i in items) if c is not None]
    if deduplicate:
        codes = list(dict.fromkeys(codes))
    return codes


def parse_label_map(x: Any) -> dict[str, int]:
    """label_map column (JSON or Python dict string) -> {code: label}."""
    if isinstance(x, dict):
        return {str(k): int(v) for k, v in x.items()}
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return {}
    s = str(x).strip()
    if not s:
        return {}
    try:
        d = json.loads(s)
    except Exception:
        try:
            d = ast.literal_eval(s)
        except Exception:
            return {}
    if not isinstance(d, dict):
        return {}
    out = {}
    for k, v in d.items():
        try:
            out[str(k)] = int(v)
        except (TypeError, ValueError):
            pass
    return out


def codes_at_tau(candidates: Any, label_map: dict[str, int], tau: int) -> list[str]:
    """
    Eq 6: S(q) = {c ∈ R(q) : label(c) >= tau}, kept in candidate (reranker) order.
    [changed] the original iterated over the label map, i.e. in the order of the LLM's
              JSON answer. The returned set is the same, so set metrics do not change.
    """
    return [c for c in ensure_code_list(candidates, deduplicate=True) if label_map.get(c, 0) >= tau]


# ---------------------------------------------------------------------
# Per-query metrics (Eqs 7-9)
# ---------------------------------------------------------------------

def query_metrics(pred: Any, true: Any, excluded: Any = None, k: int | None = None) -> dict[str, Any]:
    """
    Metrics for one query.
      k=None -> set mode    (selection, direct LLM): P(q) = all returned codes
      k=int  -> ranked mode (retrieval, reranking):  P(q) = top-k codes
    """
    pred_list = ensure_code_list(pred, deduplicate=True)
    true_set = set(ensure_code_list(true, deduplicate=True))
    excl_set = set(ensure_code_list(excluded, deduplicate=True))

    p = pred_list[:k] if k is not None else pred_list
    p_set = set(p)
    hits = p_set & true_set

    out = {
        "n_true": len(true_set),
        "n_pred": len(p),
        "n_hit": len(hits),
        # Eq 7
        "recall": len(hits) / len(true_set) if true_set else 0.0,
        # Eq 7. An empty returned set counts as precision 0 (reproduces Table 3).
        # [changed] denominator |P(q)| instead of K for ranked lists (see module docstring)
        "precision": len(hits) / len(p) if p else 0.0,
        "hit_codes": ";".join(sorted(hits)),
        # Eq 8. NaN when the query has no excluded codes, so averaging skips it.
        "n_excluded": len(excl_set),
        "leakage": len(excl_set & p_set) / len(excl_set) if excl_set else np.nan,
        "leaked_codes": ";".join(sorted(excl_set & p_set)),
    }

    if k is not None:
        # Eq 9 (ranked stages only)
        n_hit, ap = 0, 0.0
        for rank, code in enumerate(p, start=1):
            if code in true_set:
                n_hit += 1
                ap += n_hit / rank
        norm = min(len(true_set), k)          # [changed] was len(true_set)
        out["map"] = ap / norm if norm else 0.0

    return out


# ---------------------------------------------------------------------
# Hallucination check (paper, "Direct LLM baseline")
# ---------------------------------------------------------------------

# "A valid code is a five-digit number, or a four-digit number followed by the letter
#  F, T, U, M, or A."
CPT_FORMAT = re.compile(r"^(\d{5}|\d{4}[FTUMA])$")


def is_hallucinated(code: str, reference: set[str]) -> bool:
    """
    [added] A generated code is hallucinated if it fails either check:
      1. CPT format (CPT_FORMAT above);
      2. membership in the reference set = current CPT codes ∪ codes deleted in past releases.
    The paper's third rule (valid format but in a numeric range never assigned to any CPT
    section) is covered by check 2, since such a code appears in neither list.
    """
    c = normalize_cpt_code(code) or ""
    return not CPT_FORMAT.match(c) or c not in reference


# ---------------------------------------------------------------------
# DataFrame-level evaluation
# ---------------------------------------------------------------------

METADATA_COLS = ["id", "query", "category", "complexity", "has_exclusion"]


def evaluate(
    df: pd.DataFrame,
    k_list: Sequence[int] | None = None,
    pred_col: str = "retrieved_codes",
    true_col: str = "answer_codes",
    excluded_col: str = "excluded_codes",
    method_col: str = "method",
    group_cols: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate every row of `df`.
    k_list=None evaluates in set mode; otherwise each K is evaluated in ranked mode.
    `group_cols` are extra columns kept in the summary key (e.g. "tau").
    """
    rows = []
    ks = list(k_list) if k_list else [None]
    for _, r in df.iterrows():
        base = {c: r.get(c) for c in [method_col, *group_cols, *METADATA_COLS] if c in df.columns}
        for k in ks:
            m = query_metrics(r[pred_col], r[true_col], r.get(excluded_col), k=k)
            rows.append({**base, "k": k if k is not None else "set", **m})

    per_query = pd.DataFrame(rows)
    return per_query, summarize(per_query, [method_col, *group_cols, "k"])


def summarize(per_query: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """Macro average over queries. Leakage averages only the queries with exclusion criteria."""
    agg = {
        "n_queries": ("recall", "size"),
        "recall": ("recall", "mean"),
        "precision": ("precision", "mean"),
        "mean_n_pred": ("n_pred", "mean"),
        "leakage": ("leakage", "mean"),                 # NaN rows skipped -> denominator = 9 queries
        "n_exclusion_queries": ("leakage", "count"),
    }
    if "map" in per_query.columns:
        agg["map"] = ("map", "mean")
    return per_query.groupby(list(keys), as_index=False, sort=False).agg(**agg)


def save_outputs(per_query: pd.DataFrame, summary: pd.DataFrame, save_dir: str | Path) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    per_query.to_csv(save_dir / "per_query_metrics.csv", index=False)
    summary.to_csv(save_dir / "summary_metrics.csv", index=False)
    print(f"[SAVED] {save_dir / 'per_query_metrics.csv'}")
    print(f"[SAVED] {save_dir / 'summary_metrics.csv'}")
    print(summary.round(4).to_string(index=False))


def load_code_set(paths: Iterable[str | Path]) -> set[str]:
    """Read CPT codes from .txt (one per line), .csv or .xlsx (column `code`, else the first column)."""
    codes: set[str] = set()
    for p in paths:
        p = Path(p)
        if p.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(p)
        elif p.suffix.lower() == ".csv":
            df = pd.read_csv(p, dtype=str)
        else:
            df = pd.DataFrame({"code": p.read_text().split()})
        col = "code" if "code" in df.columns else df.columns[0]
        codes |= {c for c in (normalize_cpt_code(x) for x in df[col]) if c}
    return codes
