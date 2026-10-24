"""Layer 3: retrieval against the golden question set, scored with ranx.

Labels are on pages, not chunks, so they survive re-chunking. The top-k chunks are
reduced to pages in rank order before scoring. Reports hit rate, recall, MRR and nDCG
for the answerable questions, lists the misses, and compares the top-1 similarity of the
unanswerable questions with the answerable ones (the signal for "nothing good came back").

Usage: python eval_retrieval.py [--min-recall10 0.9] [--k 10]
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict

import chromadb
from ranx import Qrels, Run, evaluate

from common import CHROMA_DIR, COLLECTION, EMBEDDING_MODEL, GOLDEN, ROOT, client

RESULTS = ROOT / "results"


def pages_in_rank_order(res, i: int) -> list[tuple[str, float]]:
    seen, out = set(), []
    for meta, dist in zip(res["metadatas"][i], res["distances"][i]):
        page = meta["path"]
        if page not in seen:
            seen.add(page)
            out.append((page, 1.0 - dist))  # cosine similarity
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--min-recall10", type=float, default=None, help="exit 1 when recall@10 falls below this")
    args = ap.parse_args()

    golden = [json.loads(line) for line in GOLDEN.open(encoding="utf-8")]
    col = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION)
    oai = client()
    r = oai.embeddings.create(model=EMBEDDING_MODEL, input=[g["question"] for g in golden])
    qvecs = [d.embedding for d in sorted(r.data, key=lambda d: d.index)]
    # Fetch more chunks than k, since several chunks of one page collapse to one page.
    res = col.query(query_embeddings=qvecs, n_results=args.k * 3, include=["metadatas", "distances"])

    qrels, run, top1 = {}, {}, defaultdict(list)
    per_type = defaultdict(lambda: {"qrels": {}, "run": {}})
    misses = []
    for i, g in enumerate(golden):
        ranked = pages_in_rank_order(res, i)[: args.k]
        top1[g["answerable"]].append(ranked[0][1])
        if not g["answerable"]:
            continue
        qrels[g["id"]] = {p: 1 for p in g["relevant_pages"]}
        run[g["id"]] = {p: 1.0 / (rank + 1) for rank, (p, _) in enumerate(ranked)}
        per_type[g["type"]]["qrels"][g["id"]] = qrels[g["id"]]
        per_type[g["type"]]["run"][g["id"]] = run[g["id"]]
        if not any(p in g["relevant_pages"] for p, _ in ranked[:5]):
            misses.append((g, [p for p, _ in ranked[:3]]))

    metrics = ["hit_rate@5", "recall@5", f"recall@{args.k}", "mrr@10", "ndcg@10"]
    scores = evaluate(Qrels(qrels), Run(run), metrics)
    print(f"== {len(qrels)} answerable questions, labels on pages, model {EMBEDDING_MODEL}")
    for m in metrics:
        print(f"  {m:>10}: {scores[m]:.3f}")

    print("\n== By question type")
    for t, d in sorted(per_type.items()):
        s = evaluate(Qrels(d["qrels"]), Run(d["run"]), ["hit_rate@5", "mrr@10"])
        print(f"  {t:>12} (n={len(d['qrels']):>2}): hit_rate@5 {s['hit_rate@5']:.2f}  mrr@10 {s['mrr@10']:.2f}")

    print(f"\n== Misses in the top 5 ({len(misses)})")
    for g, got in misses:
        print(f"  {g['id']}: {g['question']}")
        print(f"      wanted {g['relevant_pages']}; got {got}")

    print("\n== Top-1 similarity, answerable against unanswerable questions")
    for label, key in (("answerable", True), ("unanswerable", False)):
        v = top1[key]
        if v:
            print(f"  {label:>13} (n={len(v):>2}): median {statistics.median(v):.3f}, min {min(v):.3f}, max {max(v):.3f}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "retrieval.json").write_text(json.dumps({"model": EMBEDDING_MODEL, "k": args.k, "n": len(qrels), "scores": scores, "misses": [g["id"] for g, _ in misses]}, indent=2))
    if args.min_recall10 is not None and scores[f"recall@{args.k}"] < args.min_recall10:
        print(f"\nFAIL: recall@{args.k} {scores[f'recall@{args.k}']:.3f} is below {args.min_recall10}")
        sys.exit(1)


if __name__ == "__main__":
    main()
