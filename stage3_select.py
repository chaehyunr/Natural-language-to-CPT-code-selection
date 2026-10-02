"""
Stage 3 — LLM selection (paper, "LLM selection", Eqs 5-6, Table 1).

The LLM labels each reranked candidate c in R(q):
  3 = Must Include : directly represents the queried procedure, no exclusion violated
  2 = Normal       : clinically related but not the queried procedure
  1 = Exclude      : violates the exclusion clause, or unrelated
and the pipeline returns S(q) = {c in R(q) : label(c) >= tau}, tau = 3.

Two prompt variants, chosen by the query's `has_exclusion` annotation:
  _PROMPT_WITH_EXCLUSION (Table 1) for the 9 queries with exclusion criteria,
  _PROMPT_BASE for the rest. Both share the same four worked examples.
The exclusion clause is extracted from the query text by pattern matching
(`extract_exclusion_clause`); if no clause is found the field reads "see query".

Only the label map is produced by the GPU run; tau is applied offline, so the
same output can be evaluated at several thresholds (--tau_list).

Usage (one run = one seed):
  python stage3_select.py --predictions results/rerank/predictions.csv \
      --models llama3,qwen3 --seed 0 --save_dir results/selection/run_0

Outputs (save_dir):
  predictions.csv       pre_selection_codes (R(q)), label_map, retrieved_codes (S(q) at --tau)
  per_query_metrics.csv
  summary_metrics.csv   Recall / Precision / Leakage per model and tau (set mode)

Changes from the original code
(SunnyBrook/Final/run_selection_eval.py + decompose_query.py, merged into this file)
-----------------------------------------------------------------------------------
Unchanged, character for character: the four worked examples, both prompts, the
exclusion-clause patterns, the label parser (incl. default label 2), the generation
settings (fp16, 4,096-token prompt cut, 512 new tokens, set_seed per model load) and
the retry fallback. 381 prompts built from the 127 queries were compared with the
original code and are identical.

[changed] Sampling (temperature 0.7) is the default, matching "Experimental setup".
          The original defaulted to greedy decoding and run_all.py passed
          --do_sample --temperature 0.7; use --greedy to turn sampling off.
[changed] decompose_query.py is folded into extract_exclusion_clause().
[changed] label_map is stored in reranker order (was the order of the LLM's JSON answer);
          S(q) is the same set.
[changed] method name drops the "__ours" suffix: mxbai_dense__Qwen3-4B__sel_llama3.
[removed] Prompt modes "nofew", "simple", "relative" and their prompts (prompt ablation is
          not reported), --input_k (K ablation not reported), BioMistral, hard-coded
          cluster model paths, automatic reranker selection, semantic metrics.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from config import (
    LLM_MODELS, MAX_NEW_TOKENS, MAX_PROMPT_TOKENS, RESULTS_DIR, TAU, TEMPERATURE,
)
from metrics import codes_at_tau, ensure_code_list, evaluate, parse_label_map, save_outputs
from stage2_rerank import load_cpt_descriptions

PIPELINE_METHOD = "mxbai_dense__Qwen3-4B"

# =============================================================
# Prompts (verbatim from the version that produced the paper results)
# =============================================================

_FEW_SHOT_EXAMPLES = """
Examples:

[Example 1 — Must Include (Label 3)]
Query: laparoscopic cholecystectomy
CPT 47562: laparoscopic cholecystectomy
Step 1. Surgical intent: remove gallbladder laparoscopically
Step 2. PM = direct   : description exactly matches the query procedure
                        (no semicolon qualifier, no with/without modifier)
Step 3. EC = ok       : no exclusion clause in query
Label: 3

[Example 2 — Normal (Label 2)]
Query: laparoscopic cholecystectomy
CPT 47563: laparoscopic cholecystectomy; with cholangiography
Step 1. Surgical intent: remove gallbladder laparoscopically
Step 2. PM = related  : stem "laparoscopic cholecystectomy" matches,
                        but semicolon qualifier adds an extra procedure
Step 3. EC = ok       : no exclusion clause in query
Label: 2

[Example 3 — Exclude via exclusion clause (Label 1)]
Query: knee arthroplasty (excluding revision)
CPT 27486: revision of total knee arthroplasty
Step 1. Surgical intent: primary knee replacement
Step 2. PM = related  : related to knee arthroplasty
Step 3. EC = violated : "revision" in description matches exclusion clause
Label: 1

[Example 4 — Exclude via unrelated procedure (Label 1)]
Query: laparoscopic cholecystectomy
CPT 27447: total knee arthroplasty
Step 1. Surgical intent: remove gallbladder laparoscopically
Step 2. PM = unrelated: completely different anatomical region and procedure
Step 3. EC = ok       : no exclusion clause in query
Label: 1
"""

# ── Base prompt (no exclusion) ────────────────────────────────
_PROMPT_BASE = """You are a medical coding expert evaluating CPT codes for surgical cohort construction.

Assign a relevance label to EACH CPT code using the following scale:
  3 = Must Include : The code directly represents the surgical procedure in the query.
  2 = Normal       : The code is related (e.g. a variant or ancillary procedure)
                     but does not directly represent the primary procedure.
  1 = Exclude      : The code is clinically unrelated to the query.

For each code, follow these steps:
  Step 1. Consider the surgical intent of the query.
  Step 2. Measure how directly the code represents the described procedure (PM).
  Step 3. Assign label 3 if PM is direct, 2 if related, 1 if unrelated.
{few_shot}
Now label the following:

Query: {query}

Candidate CPT codes:
{candidates}

Return ONLY a JSON object mapping each CPT code string to its integer label.
Format: {{"47562": 3, "47563": 2, "27447": 1}}
Do not include explanation. Use exactly the codes provided.

Labels:"""

# ── Exclusion-aware prompt ────────────────────────────────────
_PROMPT_WITH_EXCLUSION = """You are a medical coding expert evaluating CPT codes for surgical cohort construction.

Assign a relevance label to EACH CPT code using the following scale:
  3 = Must Include : The code directly represents the surgical procedure in the query
                     AND does NOT match the exclusion clause.
  2 = Normal       : The code is related but not primary,
                     AND does NOT match the exclusion clause.
  1 = Exclude      : The code MATCHES the exclusion clause: {exclusion_clause}
                     OR the code is clinically unrelated to the query.

IMPORTANT: If a code matches the exclusion clause, assign label 1 regardless of relevance.

For each code, follow these steps:
  Step 1. Consider the surgical intent of the query.
  Step 2. Check whether the code matches the exclusion clause (EC).
  Step 3. If EC violated → label 1. Otherwise assign 3 (direct) or 2 (related).
{few_shot}
Now label the following:

Query: {query}

Candidate CPT codes:
{candidates}

Return ONLY a JSON object mapping each CPT code string to its integer label.
Format: {{"47562": 3, "47563": 2, "47600": 1}}
Do not include explanation. Use exactly the codes provided.

Labels:"""


# =============================================================
# Exclusion clause extraction
# =============================================================

EXCLUSION_PATTERNS = [
    r'\(\s*excl(?:uding)?\s+(.+?)\)',      # (excluding ...)
    r'\bexcluding\s+(.+)$',
    r'\bexcept(?:\s+for)?\s+(.+)$',
    r'\bwithout\s+(.+)$',
    r'\bnot\s+including\s+(.+)$',
    r',\s*excl(?:uding)?\s+(.+)$',
    r'\bexcludes\s+(.+?)(?:\).*)?$',
]


def extract_exclusion_clause(query: str) -> Optional[str]:
    """
    'all knee arthroplasty excluding revision and removal' -> 'revision and removal'.
    Paper, "LLM selection": "Of the nine queries annotated with exclusion criteria, the
    extractor recovered a clause for eight, and the remaining query expressed its
    exclusion through the qualifier primary." For that query build_prompt() fills the
    clause field with "see query".
    [changed] was decompose_query.decompose_query()["exclusion_query"]; same patterns.
    """
    for pattern in EXCLUSION_PATTERNS:
        m = re.search(pattern, query, re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip(')')
    return None


def has_exclusion_flag(row: pd.Series, query: str) -> bool:
    """
    Paper: "the variant is chosen according to whether the query carries an exclusion
    annotation". Uses the `Has Exclusion` column (Y/N); pattern matching is used only if
    the column is absent.
    """
    if "has_exclusion" in row.index:
        return str(row["has_exclusion"]).strip().upper() in {"Y", "YES", "TRUE", "1"}
    return extract_exclusion_clause(query) is not None


# =============================================================
# Prompt building and response parsing
# =============================================================

def format_candidates(codes: List[str], code2text: Dict[str, str]) -> str:
    return "\n".join(f"{i}. {c}: {code2text.get(str(c), 'No description available')}"
                     for i, c in enumerate(codes, start=1))


def build_prompt(query: str, candidates: List[str], code2text: Dict[str, str], has_exclusion: bool) -> str:
    """Table 1 prompt for exclusion queries, base prompt otherwise; same four examples in both."""
    cand = format_candidates(candidates, code2text)
    if has_exclusion:
        clause = extract_exclusion_clause(query) or "see query"
        return _PROMPT_WITH_EXCLUSION.format(query=query, exclusion_clause=clause,
                                             candidates=cand, few_shot=_FEW_SHOT_EXAMPLES)
    return _PROMPT_BASE.format(query=query, candidates=cand, few_shot=_FEW_SHOT_EXAMPLES)


def parse_label_response(response_text: str, original_codes: List[str], default_label: int = 2) -> Dict[str, int]:
    """
    LLM response -> {code: label} for every candidate.
      1. first {...} JSON object;  2. fallback regex "12345 ... 3".
    Labels are clipped to [1, 3]; candidates the model did not label get `default_label` (2).
    """
    code_set = set(str(c) for c in original_codes)
    labels: Dict[str, int] = {}

    try:
        m = re.search(r"\{[^{}]+\}", response_text, re.DOTALL)
        if m:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    code = str(k).strip()
                    if code in code_set:
                        try:
                            labels[code] = max(1, min(3, int(v)))
                        except (ValueError, TypeError):
                            pass
    except (json.JSONDecodeError, TypeError):
        pass

    if not labels:
        for m in re.finditer(r"(\b\d{5}(?:F|T|U)?\b)[^\d]*(\d)", response_text):
            code = m.group(1)
            if code in code_set:
                try:
                    labels[code] = max(1, min(3, int(m.group(2))))
                except (ValueError, TypeError):
                    pass

    for c in original_codes:
        labels.setdefault(str(c), default_label)
    return labels


# =============================================================
# Local HuggingFace LLM
# =============================================================

class LocalLLM:
    """
    fp16 causal LM on one GPU. Paper, "Experimental setup": sampling temperature 0.7, a
    different random seed per run, prompts truncated to 4,096 tokens, 512 new tokens.
    [changed] was LocalLLMClient; do_sample now defaults to True (see module docstring).
    """

    def __init__(self, model_id: str, device: Optional[str] = None, do_sample: bool = True,
                 temperature: float = TEMPERATURE, top_p: float = 1.0, seed: Optional[int] = None,
                 max_new_tokens: int = MAX_NEW_TOKENS):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.do_sample, self.temperature, self.top_p = do_sample, temperature, top_p
        self.max_new_tokens = max_new_tokens

        if seed is not None:
            from transformers import set_seed
            set_seed(seed)                     # once per run -> reproducible sampling sequence

        print(f"[LLM] loading {model_id} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        cuda = self.device.startswith("cuda")
        kwargs: Dict[str, Any] = {"torch_dtype": torch.float16 if cuda else torch.float32}
        if cuda:
            kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
        if not cuda:
            self.model = self.model.to(self.device)
        self.input_device = next(self.model.parameters()).device

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=MAX_PROMPT_TOKENS).to(self.input_device)
        gen = dict(max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.pad_token_id,
                   do_sample=self.do_sample)
        if self.do_sample:
            gen.update(temperature=self.temperature, top_p=self.top_p)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, **gen)
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def unload(self) -> None:
        import gc
        del self.model, self.tokenizer
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def resolve_model(alias: str, paths: Dict[str, Optional[str]]) -> str:
    return paths.get(alias) or LLM_MODELS.get(alias, alias)


# =============================================================
# Selection
# =============================================================

def label_candidates(llm: LocalLLM, query: str, candidates: List[str], code2text: Dict[str, str],
                     has_exclusion: bool, retry: int = 2) -> Dict[str, int]:
    """One LLM call -> label for every candidate. If every attempt fails, all candidates get label 2."""
    prompt = build_prompt(query, candidates, code2text, has_exclusion)
    for attempt in range(1 + retry):
        try:
            return parse_label_response(llm.generate(prompt), candidates)
        except Exception as e:
            print(f"  [WARN] LLM call failed ({e})")
            if attempt < retry:
                time.sleep(1.0)
    return {str(c): 2 for c in candidates}


def label_map_json(label_map: Dict[str, int], candidates: List[str]) -> str:
    """[changed] stored in candidate (reranker) order; was the LLM answer order."""
    return json.dumps({c: label_map[c] for c in candidates if c in label_map})


def evaluate_label_maps(df: pd.DataFrame, tau_list: List[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply each tau to the stored label maps (Eq 6) and evaluate the resulting sets in set
    mode: precision over |S(q)|, no MAP (unordered output).
    """
    frames = []
    for tau in tau_list:
        d = df.copy()
        d["tau"] = tau
        d["retrieved_codes"] = [codes_at_tau(c, parse_label_map(m), tau)
                                for c, m in zip(d["pre_selection_codes"], d["label_map"])]
        frames.append(d)
    return evaluate(pd.concat(frames, ignore_index=True), k_list=None, group_cols=["tau"])


def main():
    ap = argparse.ArgumentParser(description="Stage 3: LLM selection over reranked candidates.")
    ap.add_argument("--predictions", default=str(RESULTS_DIR / "rerank" / "predictions.csv"),
                    help="stage 2 predictions.csv")
    ap.add_argument("--rerank_method", default=PIPELINE_METHOD)
    ap.add_argument("--save_dir", default=str(RESULTS_DIR / "selection" / "run_0"))
    ap.add_argument("--models", default="llama3,qwen3")
    ap.add_argument("--llama3_path", default=None, help="local model directory (optional)")
    ap.add_argument("--qwen3_path", default=None, help="local model directory (optional)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--greedy", action="store_true", help="disable sampling")
    ap.add_argument("--tau", type=int, default=TAU, help="threshold for the saved retrieved_codes")
    ap.add_argument("--tau_list", default=str(TAU), help="thresholds evaluated offline, e.g. 3,2")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max_rows", type=int, default=None, help="quick test")
    args = ap.parse_args()

    df = pd.read_csv(args.predictions)
    df = df[df["method"] == args.rerank_method].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No rows for '{args.rerank_method}' in {args.predictions}")
    if args.max_rows:
        df = df.head(args.max_rows)
    code2text = load_cpt_descriptions(strip=False)
    paths = {"llama3": args.llama3_path, "qwen3": args.qwen3_path}

    rows = []
    for alias in [m.strip() for m in args.models.split(",") if m.strip()]:
        llm = LocalLLM(resolve_model(alias, paths), device=args.device, do_sample=not args.greedy,
                       temperature=args.temperature, seed=args.seed)
        for i, (_, r) in enumerate(df.iterrows(), start=1):
            query = str(r["query"])
            candidates = ensure_code_list(r["retrieved_codes"], deduplicate=True)
            excl = has_exclusion_flag(r, query)
            labels = label_candidates(llm, query, candidates, code2text, excl) if candidates else {}
            rec = r.drop(labels=["retrieved_codes", "retrieved_scores"], errors="ignore").to_dict()
            rows.append({**rec,
                         "method": f"{args.rerank_method}__sel_{alias}",
                         "seed": args.seed,
                         "has_exclusion_flag": excl,
                         "pre_selection_codes": candidates,
                         "label_map": label_map_json(labels, candidates),
                         "retrieved_codes": codes_at_tau(candidates, labels, args.tau)})
            if i % 20 == 0:
                print(f"  [{alias}] {i}/{len(df)}")
        llm.unload()

    pred = pd.DataFrame(rows)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(save_dir / "predictions.csv", index=False)
    print(f"[SAVED] {save_dir / 'predictions.csv'}")

    tau_list = [int(t) for t in args.tau_list.split(",") if t.strip()]
    per_query, summary = evaluate_label_maps(pred, tau_list)
    save_outputs(per_query, summary, save_dir)


if __name__ == "__main__":
    main()
