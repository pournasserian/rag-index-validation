"""Embed the chunks and load them into a Chroma collection with an HNSW index.

Every embedding response is checked before anything is stored: the returned indices
cover each input once, every vector has the model's dimensions, every component is
finite, and the norm is 1 (OpenAI's embeddings are unit length). The embeddings are
also saved to data/embeddings.npy, the ground truth for the recall check.
"""

import json
import math

import chromadb
import numpy as np

from common import CHROMA_DIR, CHUNKS, COLLECTION, EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, EMBEDDINGS, client

BATCH = 100
HNSW = {"space": "cosine", "ef_construction": 100, "max_neighbors": 16, "ef_search": 100}


def embed(oai, texts: list[str]) -> np.ndarray:
    out = np.zeros((len(texts), EMBEDDING_DIMENSIONS), dtype=np.float32)
    for start in range(0, len(texts), BATCH):
        batch = texts[start:start + BATCH]
        r = oai.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        indices = sorted(d.index for d in r.data)
        if indices != list(range(len(batch))):
            raise SystemExit(f"embedding response indices do not cover the batch once: {indices[:5]}...")
        for d in r.data:
            v = np.asarray(d.embedding, dtype=np.float32)
            if v.shape != (EMBEDDING_DIMENSIONS,):
                raise SystemExit(f"vector of {v.shape[0]} dimensions, expected {EMBEDDING_DIMENSIONS}: a different model or dimensions setting")
            if not np.all(np.isfinite(v)):
                raise SystemExit("non-finite component in an embedding")
            norm = float(np.linalg.norm(v))
            if not math.isclose(norm, 1.0, abs_tol=1e-3):
                raise SystemExit(f"norm {norm:.4f} is not 1: this model does not return unit vectors")
            out[start + d.index] = v
        print(f"embedded {min(start + BATCH, len(texts))}/{len(texts)}  (model {r.model}, {r.usage.total_tokens} tokens in this batch)")
    return out


def main() -> None:
    chunks = [json.loads(line) for line in CHUNKS.open(encoding="utf-8")]
    oai = client()
    vectors = embed(oai, [c["embed_text"] for c in chunks])
    np.save(EMBEDDINGS, vectors)

    store = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        store.delete_collection(COLLECTION)
    except Exception:
        pass
    col = store.create_collection(
        COLLECTION,
        configuration={"hnsw": HNSW},
        metadata={"embedding_model": EMBEDDING_MODEL, "dimensions": EMBEDDING_DIMENSIONS, "chunks_file": str(CHUNKS.name)},
    )
    for start in range(0, len(chunks), BATCH):
        part = chunks[start:start + BATCH]
        col.add(
            ids=[c["id"] for c in part],
            embeddings=vectors[start:start + len(part)].tolist(),
            documents=[c["text"] for c in part],
            metadatas=[{"path": c["path"], "title": c["title"], "heading_path": c["heading_path"], "tokens": c["tokens"]} for c in part],
        )
    print(f"\n{col.count()} points in collection {COLLECTION!r} at {CHROMA_DIR} (HNSW {HNSW})")


if __name__ == "__main__":
    main()
