"""Layer 2: check the stored embeddings and the HNSW index itself.

Checks: the count against the chunks file; the model recorded on the collection against
the one configured now; dimensions, finite components and unit norms of the stored vectors;
duplicate vectors; a re-embedding of a sample against what is stored; self-retrieval; and
recall@10 of the approximate index against a brute-force search over the same vectors,
at several ef_search settings, for the golden questions and for a sample of stored vectors.
"""

import json
import random
import sys
import time

import chromadb
import numpy as np

from common import CHROMA_DIR, CHUNKS, COLLECTION, EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, GOLDEN, client

K = 10
EF_SEARCH = [10, 32, 100]
random.seed(7)


def exact_top_k(matrix: np.ndarray, query: np.ndarray, k: int) -> list[int]:
    scores = matrix @ query  # unit vectors, so the dot product is the cosine
    return list(np.argsort(-scores)[:k])


def main() -> None:
    failures = []
    chunks = [json.loads(line) for line in CHUNKS.open(encoding="utf-8")]
    store = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = store.get_collection(COLLECTION)

    print("== Count and version")
    n = col.count()
    print(f"points: {n}; chunks in the file: {len(chunks)}")
    if n != len(chunks):
        failures.append("point count differs from the chunk count")
    recorded = col.metadata.get("embedding_model")
    print(f"model recorded on the collection: {recorded}; configured now: {EMBEDDING_MODEL}")
    if recorded != EMBEDDING_MODEL:
        failures.append("the query path would use a different embedding model than the index")

    print("\n== Stored vectors")
    got = col.get(include=["embeddings", "metadatas"])
    ids = got["ids"]
    matrix = np.asarray(got["embeddings"], dtype=np.float32)
    print(f"shape {matrix.shape}")
    if matrix.shape[1] != EMBEDDING_DIMENSIONS:
        failures.append(f"stored dimension {matrix.shape[1]} is not {EMBEDDING_DIMENSIONS}")
    if not np.all(np.isfinite(matrix)):
        failures.append("non-finite components in stored vectors")
    norms = np.linalg.norm(matrix, axis=1)
    print(f"norms: min {norms.min():.4f}, max {norms.max():.4f}")
    if np.any(np.abs(norms - 1) > 1e-3):
        failures.append("stored vectors are not unit length")
    rounded = {}
    dupes = 0
    for i, v in enumerate(matrix):
        key = np.round(v, 4).tobytes()
        if key in rounded:
            dupes += 1
            if dupes <= 3:
                a, b = rounded[key], i
                print(f"  duplicate vector: {got['metadatas'][a]['path']} / {got['metadatas'][b]['path']}")
        else:
            rounded[key] = i
    print(f"duplicate vectors (identical to 4 decimals): {dupes}")

    print("\n== Re-embed a sample with the current client")
    oai = client()
    sample = random.sample(range(len(chunks)), 10)
    by_id = {c["id"]: c for c in chunks}
    pos = {cid: i for i, cid in enumerate(ids)}
    texts = [chunks[i]["embed_text"] for i in sample]
    r = oai.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    sims = []
    for d, i in zip(sorted(r.data, key=lambda d: d.index), sample):
        v = np.asarray(d.embedding, dtype=np.float32)
        sims.append(float(v @ matrix[pos[chunks[i]["id"]]]))
    print(f"cosine between the fresh and the stored vector: min {min(sims):.4f}, mean {np.mean(sims):.4f} (1.0 means the same model and settings)")
    if min(sims) < 0.99:
        failures.append("a fresh embedding does not match the stored one: model, dimensions or prefix changed")

    print("\n== Self-retrieval (a stored vector must find its own point first)")
    probe = random.sample(range(len(ids)), 100)
    hits = 0
    for i in probe:
        res = col.query(query_embeddings=[matrix[i].tolist()], n_results=1, include=[])
        hits += res["ids"][0][0] == ids[i]
    print(f"{hits}/100 at ef_search {col.configuration_json['hnsw']['ef_search'] if hasattr(col, 'configuration_json') else '?'}")

    print(f"\n== Recall@{K} of the HNSW index against brute force over the same vectors")
    golden = [json.loads(line) for line in GOLDEN.open(encoding="utf-8")]
    questions = [g["question"] for g in golden]
    qr = oai.embeddings.create(model=EMBEDDING_MODEL, input=questions)
    qvecs = np.asarray([d.embedding for d in sorted(qr.data, key=lambda d: d.index)], dtype=np.float32)
    svecs = matrix[random.sample(range(len(ids)), 50)]
    for name, queries in (("golden questions", qvecs), ("stored vectors", svecs)):
        truth = [set(ids[j] for j in exact_top_k(matrix, q, K)) for q in queries]
        for ef in EF_SEARCH:
            col.modify(configuration={"hnsw": {"ef_search": ef}})
            t0 = time.perf_counter()
            res = col.query(query_embeddings=queries.tolist(), n_results=K, include=[])
            dt = (time.perf_counter() - t0) / len(queries) * 1000
            recall = np.mean([len(set(res["ids"][i]) & truth[i]) / K for i in range(len(queries))])
            print(f"  {name:>16}  ef_search {ef:>3}: recall@{K} {recall:.3f}  ({dt:.1f} ms per query)")
    col.modify(configuration={"hnsw": {"ef_search": 100}})

    print()
    if failures:
        for f in failures:
            print("FAIL:", f)
        sys.exit(1)
    print("no hard failures")


if __name__ == "__main__":
    main()
