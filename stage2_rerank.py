"""
Stage 2 — Reranking (paper, "Reranking").

Rescores the retrieved pool C(q) with a model that reads the query and each
code description together (Eq 4), and keeps the top K_rerank = 20 as R(q).

Rerankers (compared in Table 2):
  bge_m3  BAAI/bge-reranker-v2-m3      cross-encoder, sigmoid of logit
  medcpt  ncbi/MedCPT-Cross-Encoder     cross-encoder, raw logit
  qwen3   Qwen/Qwen3-Reranker-4B        generative, P("yes")      <- used in the pipeline

Usage:
  python stage2_rerank.py --predictions results/retrieval/predictions.csv
  python stage2_rerank.py --predictions ... --rerankers qwen3        # pipeline reranker only

Outputs (save_dir):
  predictions.csv       method = "<retriever>__<reranker name>", e.g. mxbai_dense__Qwen3-4B
  per_query_metrics.csv
  summary_metrics.csv   Recall / Precision / Leakage / MAP @ K   (Table 2, Reranker rows)

Changes from the original code
(SunnyBrook/Final/reranker.py + run_reranker_eval.py, merged into this file)
---------------------------------------------------------------------------
[changed] BGE-v2-m3 and MedCPT share one CrossEncoderReranker class; each keeps its
          original settings (BGE: sigmoid of logit, fp16 on GPU; MedCPT: raw logit, fp32).
[changed] excluded_codes is passed through to the output (needed for Eq 8).
[changed] The input retriever is set explicitly (--retrieval_method mxbai_dense) instead of
          being auto-selected from the retrieval summary. Paper: "The retriever with the
          highest recall was used in the full pipeline" -> mxbai.
[changed] Metrics are evaluated at K = 20 only (paper: "reranking at K = 20").
[changed] Output file all_reranked_predictions.csv -> predictions.csv.
[removed] PubMedBERT-ColBERT and BGE-v2-Gemma rerankers (not in the paper), together with
          the FlagEmbedding and pylate dependencies; reranker alias groups; semantic metrics.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import pandas as pd

from config import CPT_DATA, K_RERANK, RESULTS_DIR
from metrics import ensure_code_list, evaluate, normalize_cpt_code, save_outputs

PASS_THROUGH = ["id", "category", "complexity", "has_exclusion", "query", "answer_codes", "excluded_codes"]


# [changed] reranker.load_cpt_code2text() and run_selection_eval.load_cpt_code2text() merged
def load_cpt_descriptions(path=CPT_DATA, strip: bool = True) -> dict[str, str]:
    """
    CPT code -> lower-cased description.

    strip=True is what the reranker saw; the selection prompt used the
    unstripped text (85 descriptions carry leading/trailing whitespace), so
    stage 3 calls this with strip=False to keep prompts byte-identical.
    """
    df = pd.read_excel(path)
    df = df[df["system"] == "CPT4"].dropna(subset=["code", "display"])
    df = df.drop_duplicates(subset="code", keep="first")
    out = {}
    for code, text in zip(df["code"], df["display"].astype(str)):
        code = normalize_cpt_code(code)
        text = (text.strip() if strip else text).lower()
        if code and text:
            out[code] = text
    return out


# ---------------------------------------------------------------------
# Rerankers
# ---------------------------------------------------------------------

def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class BaseReranker:
    name = "base"

    def score(self, query: str, texts: list[str]) -> list[float]:
        raise NotImplementedError

    def rerank(self, query: str, codes: list[str], code2text: dict[str, str], top_k: int):
        codes = [c for c in codes if code2text.get(c)]          # candidates without a description are skipped
        if not codes:
            return []
        scores = self.score(query, [code2text[c] for c in codes])
        ranked = sorted(zip(codes, scores), key=lambda x: float(x[1]), reverse=True)[:top_k]
        return [(c, round(float(s), 6)) for c, s in ranked]


class CrossEncoderReranker(BaseReranker):
    """
    Eq 4 for the two cross-encoders: s = F(q, d(c)) from one joint forward pass.
    [changed] shared code for the original BGERerankerM3 and MedCPTReranker classes.
    """
    model_id = ""
    use_sigmoid = False
    half = False

    def __init__(self, device=None, batch_size: int = 16, max_length: int = 512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.device = torch, device or _device()
        self.batch_size, self.max_length = batch_size, max_length
        dtype = torch.float16 if (self.half and self.device.startswith("cuda")) else torch.float32
        print(f"[{self.name}] loading {self.model_id} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, torch_dtype=dtype).to(self.device).eval()

    def score(self, query, texts):
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = [[query, t] for t in texts[i:i + self.batch_size]]
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length,
                                    return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                logits = self.model(**inputs).logits.squeeze(-1)
            if self.use_sigmoid:
                logits = self.torch.sigmoid(logits)
            s = logits.cpu().tolist()
            out.extend(s if isinstance(s, list) else [s])
        return out


class BGERerankerM3(CrossEncoderReranker):
    name, model_id, use_sigmoid, half = "BGE-v2-m3", "BAAI/bge-reranker-v2-m3", True, True


class MedCPTReranker(CrossEncoderReranker):
    name, model_id, use_sigmoid, half = "MedCPT", "ncbi/MedCPT-Cross-Encoder", False, False


class Qwen3Reranker(BaseReranker):
    """
    Eq 4 for Qwen3-Reranker-4B: "prompted to judge whether a document meets the needs of
    the query and to answer only yes or no, with the probability assigned to yes used as
    the score." Unchanged from the original (prompt text, 1,024-character document cut).
    """
    name = "Qwen3-4B"
    model_id = "Qwen/Qwen3-Reranker-4B"
    PROMPT = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the needs of the User's Query. "
        "The answer can only be 'yes' or 'no'."
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Query: {query}\nDocument: {doc}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )

    def __init__(self, device=None, batch_size: int = 4, max_length: int = 4096, doc_max_chars: int = 1024):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.device = torch, device or _device()
        self.batch_size, self.max_length, self.doc_max_chars = batch_size, max_length, doc_max_chars
        cuda = self.device.startswith("cuda")
        print(f"[{self.name}] loading {self.model_id} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side="left")
        kwargs: dict[str, Any] = {"torch_dtype": torch.float16 if cuda else torch.float32}
        if cuda:
            kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs).eval()
        if not cuda:
            self.model = self.model.to(self.device)
        self.input_device = next(self.model.parameters()).device
        self.yes = self.tokenizer.convert_tokens_to_ids("yes")
        self.no = self.tokenizer.convert_tokens_to_ids("no")

    def score(self, query, texts):
        prompts = [self.PROMPT.format(query=query, doc=t[: self.doc_max_chars]) for t in texts]
        out = []
        for i in range(0, len(prompts), self.batch_size):
            inputs = self.tokenizer(prompts[i:i + self.batch_size], return_tensors="pt", padding=True,
                                    truncation=True, max_length=self.max_length).to(self.input_device)
            with self.torch.no_grad():
                logits = self.model(**inputs).logits[:, -1, :]
            yes_no = self.torch.stack([logits[:, self.yes], logits[:, self.no]], dim=1)
            out.extend(self.torch.softmax(yes_no, dim=1)[:, 0].cpu().tolist())
        return out


RERANKERS = {"bge_m3": BGERerankerM3, "medcpt": MedCPTReranker, "qwen3": Qwen3Reranker}


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Stage 2: rerank retrieved CPT candidates.")
    ap.add_argument("--predictions", default=str(RESULTS_DIR / "retrieval" / "predictions.csv"),
                    help="stage 1 predictions.csv")
    ap.add_argument("--save_dir", default=str(RESULTS_DIR / "rerank"))
    ap.add_argument("--retrieval_method", default="mxbai_dense", help="stage-1 method to rerank")
    ap.add_argument("--rerankers", default=",".join(RERANKERS))
    ap.add_argument("--k", type=int, default=K_RERANK, help="candidates kept per query (K_rerank)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max_rows", type=int, default=None, help="quick test")
    args = ap.parse_args()

    df = pd.read_csv(args.predictions)
    df = df[df["method"] == args.retrieval_method].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No rows for retrieval method '{args.retrieval_method}' in {args.predictions}")
    if args.max_rows:
        df = df.head(args.max_rows)
    code2text = load_cpt_descriptions(strip=True)

    rows = []
    for key in [r.strip() for r in args.rerankers.split(",") if r.strip()]:
        reranker = RERANKERS[key](device=args.device)
        print(f"[run] {args.retrieval_method} -> {reranker.name}")
        for _, r in df.iterrows():
            candidates = ensure_code_list(r["retrieved_codes"], deduplicate=True)
            ranked = reranker.rerank(r["query"], candidates, code2text, args.k)
            rows.append({"method": f"{args.retrieval_method}__{reranker.name}",
                         # [changed] PASS_THROUGH now includes excluded_codes
                         **{c: r[c] for c in PASS_THROUGH if c in df.columns},
                         "retrieved_codes": [c for c, _ in ranked],
                         "retrieved_scores": dict(ranked),
                         "base_retrieved_codes": candidates})
        del reranker
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    pred = pd.DataFrame(rows)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(save_dir / "predictions.csv", index=False)
    print(f"[SAVED] {save_dir / 'predictions.csv'}")

    per_query, summary = evaluate(pred, k_list=[args.k])
    save_outputs(per_query, summary, save_dir)


if __name__ == "__main__":
    main()
