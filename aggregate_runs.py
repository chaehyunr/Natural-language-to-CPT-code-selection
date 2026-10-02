"""
Aggregate the three selection runs into mean (SD).

Paper, "Experimental setup": "The selection stage was run three times with identical
inputs, a sampling temperature of 0.7, and a different random seed for each run, and
results are reported as the mean (standard deviation) across the three runs."
Used for the LLM rows of Table 2 and for Table 3 (direct LLM baseline).

Usage:
  python aggregate_runs.py results/selection/run_0 results/selection/run_1 results/selection/run_2
  python aggregate_runs.py results/direct_llm/run_0 results/direct_llm/run_1 results/direct_llm/run_2

Changes from the original code
(SunnyBrook/Final/analyze_results.py, and llm_only_metrics.py from the local project)
------------------------------------------------------------------------------------
[changed] Reports the sample SD (ddof = 1). analyze_results.py reported a 95% CI half-width
          (t-table), which the paper does not use.
[changed] One script for both the pipeline and the direct LLM baseline. Recall, precision
          and leakage match llm_only_metrics.py exactly (checked on the same predictions):
          per-run macro averages, empty sets count as precision 0, leakage over the 9
          exclusion queries.
[added]   hallucination_rate, when present (direct LLM baseline).
[removed] Leakage recovery by joining the dataset on query text (excluded_codes now flows
          through the pipeline); K-ablation and prompt-ablation tables (not reported);
          Excel-sheet input of llm_only_metrics.py.
"""

import argparse
from pathlib import Path

import pandas as pd

METRICS = ["recall", "precision", "leakage", "mean_n_pred", "hallucination_rate"]
KEYS = ["method", "tau", "k"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", help="directories containing summary_metrics.csv")
    ap.add_argument("--out", default=None, help="output CSV (default: <parent of first run>/aggregate.csv)")
    args = ap.parse_args()

    frames = []
    for d in args.run_dirs:
        s = pd.read_csv(Path(d) / "summary_metrics.csv")
        s["run"] = Path(d).name
        frames.append(s)
    runs = pd.concat(frames, ignore_index=True)
    keys = [k for k in KEYS if k in runs.columns]
    metrics = [m for m in METRICS if m in runs.columns]

    g = runs.groupby(keys)
    agg = g[metrics].agg(["mean", "std"])            # pandas std uses ddof = 1
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg["n_runs"] = g["run"].nunique()
    agg = agg.reset_index()

    out = Path(args.out) if args.out else Path(args.run_dirs[0]).parent / "aggregate.csv"
    agg.to_csv(out, index=False)
    print(f"[SAVED] {out}\n")
    for _, r in agg.iterrows():
        label = "  ".join(f"{k}={r[k]}" for k in keys)
        vals = "  ".join(f"{m}={r[f'{m}_mean']:.3f} ({r[f'{m}_std']:.3f})" for m in metrics)
        print(f"{label}  n_runs={r['n_runs']}\n    {vals}")


if __name__ == "__main__":
    main()
