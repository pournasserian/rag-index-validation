"""Fetch the sample corpus: the Markdown pages of the MCP specification at a pinned commit.

The pages are Apache-2.0 and are not committed to this repository. They are downloaded
into data/corpus/ with a manifest that records the commit, so a run is reproducible.
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

from common import CORPUS_DIR, DATA_DIR, MANIFEST

REPO = "modelcontextprotocol/modelcontextprotocol"
COMMIT = "dae694f61a31f7a3ec2af3aff709e11f70f61e48"  # main on 2026-10-23
PREFIX = "docs/specification/2026-07-28/"
RAW = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/"
TREE = f"https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1"


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "rag-index-validation"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main() -> None:
    tree = json.loads(get(TREE))
    pages = sorted(
        e["path"]
        for e in tree["tree"]
        if e["type"] == "blob" and e["path"].startswith(PREFIX) and e["path"].endswith(".mdx")
    )
    if not pages:
        sys.exit("no pages found; is the commit or prefix wrong?")
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"repository": REPO, "commit": COMMIT, "prefix": PREFIX, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pages": []}
    for path in pages:
        rel = path[len(PREFIX):]
        target = CORPUS_DIR / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        body = get(RAW + path)
        target.write_bytes(body)
        manifest["pages"].append({"path": rel, "bytes": len(body)})
        print(f"{len(body):>8}  {rel}")
    DATA_DIR.mkdir(exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(pages)} pages written to {CORPUS_DIR}; manifest at {MANIFEST}")


if __name__ == "__main__":
    main()
