"""Layer 1: check the documents and their chunks before anything is embedded.

Reads data/manifest.json, data/corpus/ and data/chunks.jsonl and prints a report.
Exits non-zero on a hard failure: a document missing or empty, a chunk over the
embedding model's input limit, or a chunk without its metadata.
"""

import hashlib
import json
import re
import statistics
import sys
from collections import Counter, defaultdict

from datasketch import MinHash, MinHashLSH

from common import CHUNKS, CHUNK_TOKENS, CORPUS_DIR, MANIFEST

EMBEDDING_INPUT_LIMIT = 8192  # OpenAI's limit for every embedding model, as of October 2026
TINY = 20  # tokens; a chunk below this is a heading or a table fragment on its own
TERMINAL = re.compile(r"[.!?:;)\]`\"'|}]\s*$|```\s*$")
TAG = re.compile(r"<[^>]+>")


def shingles(text: str, n: int = 3) -> set[str]:
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))}


def main() -> None:
    failures = []
    manifest = json.loads(MANIFEST.read_text())
    chunks = [json.loads(line) for line in CHUNKS.open(encoding="utf-8")]
    by_doc = defaultdict(list)
    for c in chunks:
        by_doc[c["path"]].append(c)

    print("== Documents")
    expected = [p["path"] for p in manifest["pages"]]
    print(f"expected from the source: {len(expected)}; chunked: {len(by_doc)}")
    for rel in expected:
        f = CORPUS_DIR / rel
        if not f.exists():
            failures.append(f"missing file: {rel}")
            continue
        raw = f.read_text(encoding="utf-8", errors="replace")
        if not raw.strip():
            failures.append(f"empty file: {rel}")
        if rel not in by_doc:
            failures.append(f"no chunks: {rel}")
        bad = raw.count("�")
        if bad:
            print(f"  encoding: {bad} replacement characters in {rel}")
        markup = sum(len(m) for m in TAG.findall(raw)) / max(1, len(raw))
        if markup > 0.3:
            print(f"  markup: {markup:.0%} of {rel} is inside tags ({len(raw):,} bytes); check the extraction")

    print("\n== Chunks per document")
    counts = {d: len(v) for d, v in by_doc.items()}
    med = statistics.median(counts.values())
    print(f"chunks: {len(chunks)}; per document median {med:.0f}, min {min(counts.values())}, max {max(counts.values())}")
    for d, n in sorted(counts.items(), key=lambda x: -x[1])[:3]:
        flag = "  <- more than ten times the median" if n > 10 * med else ""
        print(f"  {n:>5}  {d}{flag}")

    print("\n== Chunk length (tokens of the embedded text)")
    toks = sorted(c["tokens"] for c in chunks)
    q = lambda p: toks[min(len(toks) - 1, int(p * len(toks)))]
    print(f"min {toks[0]}, median {q(0.5)}, p95 {q(0.95)}, max {toks[-1]}; cap {CHUNK_TOKENS}")
    over = [c for c in chunks if c["tokens"] > EMBEDDING_INPUT_LIMIT]
    if over:
        failures.append(f"{len(over)} chunks over the {EMBEDDING_INPUT_LIMIT}-token input limit")
    forced = sum(c["forced_split"] for c in chunks)
    tiny = [c for c in chunks if c["tokens"] < TINY]
    print(f"split by force at the cap: {forced}; under {TINY} tokens: {len(tiny)}")
    for c in tiny[:3]:
        print(f"  tiny: {c['path']} [{c['heading_path']}]: {c['text'][:60]!r}")
    dangling = [c for c in chunks if not TERMINAL.search(c["text"])]
    print(f"not ending in punctuation or a code fence: {len(dangling)} ({len(dangling) / len(chunks):.0%})")
    for c in dangling[:3]:
        print(f"  dangling: {c['path']}: ...{c['text'][-60:]!r}")

    print("\n== Metadata")
    for c in chunks:
        if not (c.get("path") and c.get("title") and c.get("heading_path") is not None and c.get("id")):
            failures.append(f"chunk without metadata: {c.get('id')}")
            break
    else:
        print("every chunk has an id, a path, a title and a heading path")
    no_context = sum(1 for c in chunks if c["heading_path"] == c["title"])
    print(f"chunks whose only context is the page title: {no_context}")

    print("\n== Duplicates")
    exact = Counter(hashlib.sha1(c["text"].encode()).hexdigest() for c in chunks)
    dupes = [h for h, n in exact.items() if n > 1]
    print(f"exact duplicate texts: {len(dupes)}")
    lsh = MinHashLSH(threshold=0.8, num_perm=128)
    sigs = {}
    for c in chunks:
        m = MinHash(num_perm=128)
        for s in shingles(c["text"]):
            m.update(s.encode())
        sigs[c["id"]] = m
        lsh.insert(c["id"], m)
    near = set()
    for cid, m in sigs.items():
        for other in lsh.query(m):
            if other != cid:
                near.add(tuple(sorted((cid, other))))
    print(f"near-duplicate pairs (estimated Jaccard >= 0.8 on word 3-grams): {len(near)}")
    by_id = {c["id"]: c for c in chunks}
    for a, b in sorted(near)[:5]:
        print(f"  {by_id[a]['path']} [{by_id[a]['heading_path']}]  ~  {by_id[b]['path']} [{by_id[b]['heading_path']}]")

    print()
    if failures:
        for f in failures:
            print("FAIL:", f)
        sys.exit(1)
    print("no hard failures")


if __name__ == "__main__":
    main()
