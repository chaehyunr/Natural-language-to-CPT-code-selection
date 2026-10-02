"""
Build the local Qdrant index of the CPT corpus (paper, "CPT code corpus").

"The description of each code was converted into an embedding vector with a text
encoder, and the vectors were stored in a vector database together with the code
identifiers and descriptions. The index was built once before the experiments."

One collection per encoder, named by its alias:
  bge, mxbai, clinicalbert  (dense, cosine)
  bm25                      (sparse)
Payload per point: cpt_code, text (lower-cased description).

Usage:
  python build_index.py                     # all encoders
  python build_index.py --encoders mxbai    # a subset

Changes from the original code (SunnyBrook/Final/build_db.py)
------------------------------------------------------------
[changed] DenseDBBuilder / SparseDBBuilder classes merged into one build_collection().
          Corpus loading, embedding, point ids, payload and distance are unchanged.
[changed] --encoders option to build a subset of collections.
[changed] fastembed / qdrant imports moved inside the functions, so modules that only
          need load_cpt_corpus() (e.g. the direct LLM baseline) do not require them.
[removed] SPLADE collection (not compared in the paper).
"""

import argparse
import atexit

import pandas as pd

from config import BATCH_SIZE, CPT_DATA, DB_DIR, DENSE_MODELS, SPARSE_MODELS


def load_cpt_corpus(path=CPT_DATA) -> pd.DataFrame:
    """
    CPT4 rows only, one row per code -> 8,745 codes.
    Paper, "Retrieval": "Descriptions are lower-cased and otherwise left unchanged."
    """
    cpt = pd.read_excel(path)
    cpt = cpt[cpt["system"] == "CPT4"].reset_index(drop=True)
    cpt = cpt.drop_duplicates(subset="code", keep="first").reset_index(drop=True)
    cpt["display"] = cpt["display"].str.lower()
    return cpt


def load_dense_encoder(alias: str):
    """Returns text -> vector. bge via fastembed; mxbai and ClinicalBERT via SentenceTransformer."""
    cfg = DENSE_MODELS[alias]
    if cfg["backend"] == "fastembed":
        from fastembed import TextEmbedding
        model = TextEmbedding(cfg["name"])
        return lambda text: list(model.embed([text]))[0].tolist()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(cfg["name"])
    return lambda text: model.encode(text).tolist()


def load_sparse_encoder(alias: str):
    """Returns text -> Qdrant SparseVector (BM25)."""
    from fastembed import SparseTextEmbedding
    from qdrant_client.models import SparseVector
    model = SparseTextEmbedding(SPARSE_MODELS[alias]["name"])

    def encode(text):
        r = list(model.embed([text]))[0]
        return SparseVector(indices=r.indices.tolist(), values=r.values.tolist())
    return encode


def build_collection(client, alias: str, cpt: pd.DataFrame) -> None:
    # [changed] replaces DenseDBBuilder.build_db() and SparseDBBuilder.build_db()
    from qdrant_client.models import Distance, PointStruct, SparseIndexParams, SparseVectorParams, VectorParams

    if alias in [c.name for c in client.get_collections().collections]:
        print(f"[skip] '{alias}' already exists")
        return

    dense = alias in DENSE_MODELS
    encode = load_dense_encoder(alias) if dense else load_sparse_encoder(alias)
    if dense:
        dim = len(encode("test"))
        client.create_collection(alias, vectors_config={"dense": VectorParams(size=dim, distance=Distance.COSINE)})
    else:
        client.create_collection(alias, vectors_config={},
                                 sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams())})

    vec_name = "dense" if dense else "sparse"
    points = []
    for i, row in cpt.iterrows():
        text = str(row["display"])
        if not text:
            continue
        points.append(PointStruct(id=i, vector={vec_name: encode(text)},
                                  payload={"cpt_code": str(row["code"]), "text": text}))
        if (i + 1) % 1000 == 0:
            print(f"  [{alias}] {i + 1}/{len(cpt)}")

    for start in range(0, len(points), BATCH_SIZE):
        client.upsert(collection_name=alias, points=points[start:start + BATCH_SIZE])
    print(f"[done] '{alias}': {len(points)} codes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", default=",".join([*DENSE_MODELS, *SPARSE_MODELS]))
    args = ap.parse_args()

    from qdrant_client import QdrantClient
    DB_DIR.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(DB_DIR))
    atexit.register(client.close)

    cpt = load_cpt_corpus()
    print(f"CPT corpus: {len(cpt)} codes")
    for alias in [a.strip() for a in args.encoders.split(",") if a.strip()]:
        build_collection(client, alias, cpt)


if __name__ == "__main__":
    main()
