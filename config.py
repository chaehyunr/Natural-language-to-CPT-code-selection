"""
Paths, model registry, and the fixed settings of the paper's experiments.

Changes from the original code (SunnyBrook/Final/config.py)
----------------------------------------------------------
[changed] All experiment settings now live here. They were previously spread
          across CLI defaults of run_retrieval_eval.py / run_selection_eval.py /
          run_llm_only.py and the arguments passed by run_all.py.
[changed] DENSE_MODEL_CONFIG + MODEL_ALIAS merged into DENSE_MODELS / SPARSE_MODELS,
          keyed by the alias that is also the Qdrant collection name.
[removed] SPLADE (not in the paper), COLLECTION_NAME (unused),
          DEFAULT_TOP_K = 100 and DEFAULT_PREFETCH_LIMIT = 200 (never used: the CLI
          defaults 30 and 60 always overrode them), TEST_DATA (old dataset).
[removed] Hard-coded cluster model paths; local copies are passed with --llama3_path / --qwen3_path.
"""

from pathlib import Path

BASE_DIR = Path(__file__).parent
CPT_DATA = BASE_DIR / "data" / "web_cpt_codes.xlsx"            # CPT corpus (not distributed, see data/README.md)
VAL_DATA = BASE_DIR / "data" / "cpt_query_validation_dataset.csv"
DB_DIR = BASE_DIR / "db"                                        # local Qdrant index
RESULTS_DIR = BASE_DIR / "results"

# Dataset columns
QUERY_COL = "Query"
INCLUDED_COL = "Included CPT Codes"
EXCLUDED_COL = "Excluded CPT Codes"
HAS_EXCLUSION_COL = "Has Exclusion"

# Paper, "Evaluation dataset": 18 of the 145 authored queries had more than 10
# ground-truth codes and were removed, leaving 127.
# [changed] was the CLI default --max_true_codes 10 in run_retrieval_eval.py / run_llm_only.py
MAX_TRUE_CODES = 10

# Paper, "Retrieval": three dense encoders and BM25; alias = Qdrant collection name.
DENSE_MODELS = {
    "bge":          {"name": "BAAI/bge-small-en-v1.5",             "backend": "fastembed"},
    "mxbai":        {"name": "mixedbread-ai/mxbai-embed-large-v1", "backend": "sentence_transformer"},
    "clinicalbert": {"name": "medicalai/ClinicalBERT",             "backend": "sentence_transformer"},
}
SPARSE_MODELS = {
    "bm25": {"name": "Qdrant/bm25"},
}

# Paper, "LLM selection": two open-weight LLMs.
# [removed] BioMistral-7B (excluded from the paper: did not follow the output format)
LLM_MODELS = {
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "qwen3":  "Qwen/Qwen3-8B",
}

# Paper, "Experimental setup"
K_RETRIEVE = 30               # retrieval keeps 30 candidates
K_RERANK = 20                 # reranking keeps 20
RRF_K = 60                    # RRF constant, Eq 3
PREFETCH_LIMIT = 60           # candidates taken from each source before RRF (not stated in the paper; value used)
TAU = 3                       # selection threshold, Eq 6
SEEDS = [0, 1, 2]             # selection run three times, different seed per run
TEMPERATURE = 0.7             # sampling temperature
MAX_PROMPT_TOKENS = 4096      # prompts truncated to 4,096 tokens
MAX_NEW_TOKENS = 512          # generation limited to 512 new tokens

BATCH_SIZE = 200              # Qdrant upsert batch size
