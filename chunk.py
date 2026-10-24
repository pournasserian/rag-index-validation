"""Split the corpus into chunks: one record per chunk in data/chunks.jsonl.

Structure first (title, then the ## and ### sections), then paragraphs packed up to a hard
cap in the embedding model's own tokens. Each chunk keeps its heading path, which is
prepended to the text that is embedded so a chunk can be placed without its page.
Chunk ids are deterministic for a given chunker, so re-running upserts the same ids.
"""

import hashlib
import html
import json
import re
from pathlib import Path

from common import CHUNKS, CHUNK_TOKENS, CORPUS_DIR, ENCODING, MANIFEST, count_tokens

TAG = re.compile(r"<[^>]+>")
FRONT = re.compile(r"\A---\n(.*?)\n---\n", re.S)
HEADING = re.compile(r"^(#{2,3})\s+(.*?)\s*$", re.M)
FENCE = "```"


def clean(raw: str) -> tuple[str, str]:
    """Return (title, body) with the MDX wrappers and HTML tags removed."""
    title = ""
    m = FRONT.match(raw)
    if m:
        t = re.search(r"^title:\s*(.+)$", m.group(1), re.M)
        title = t.group(1).strip().strip('"') if t else ""
        raw = raw[m.end():]
    lines = []
    for line in raw.splitlines():
        if line.startswith("import ") or line.startswith("export "):
            continue
        lines.append(line)
    body = "\n".join(lines)
    body = TAG.sub(" ", body)
    body = html.unescape(body)
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return title, body.strip()


def sections(title: str, body: str):
    """Yield (heading_path, text) for the title block and every ## / ### section."""
    pos = 0
    path = [title] if title else []
    h2 = None
    for m in HEADING.finditer(body):
        text = body[pos:m.start()].strip()
        if text:
            yield list(path), text
        level, heading = len(m.group(1)), m.group(2).strip()
        if level == 2:
            h2 = heading
            path = [p for p in path[:1]] + [heading]
        else:
            path = [p for p in path[:1]] + ([h2] if h2 else []) + [heading]
        pos = m.end()
    text = body[pos:].strip()
    if text:
        yield list(path), text


def paragraphs(text: str) -> list[str]:
    """Blank-line paragraphs, with a fenced code block kept whole."""
    out, buf, in_fence = [], [], False
    for line in text.split("\n"):
        if line.strip().startswith(FENCE):
            in_fence = not in_fence
            buf.append(line)
            if not in_fence:
                out.append("\n".join(buf).strip())
                buf = []
            continue
        if in_fence:
            buf.append(line)
        elif line.strip() == "":
            if buf:
                out.append("\n".join(buf).strip())
                buf = []
        else:
            buf.append(line)
    if buf:
        out.append("\n".join(buf).strip())
    return [p for p in out if p]


def pack(paras: list[str], cap: int):
    """Greedy packing of paragraphs up to cap tokens; an oversized paragraph is split by force."""
    buf, size = [], 0
    for p in paras:
        n = count_tokens(p)
        if n > cap:
            if buf:
                yield "\n\n".join(buf), False
                buf, size = [], 0
            toks = ENCODING.encode(p)
            for i in range(0, len(toks), cap):
                yield ENCODING.decode(toks[i:i + cap]), True
            continue
        if size + n + 1 > cap and buf:
            yield "\n\n".join(buf), False
            buf, size = [], 0
        buf.append(p)
        size += n + 1
    if buf:
        yield "\n\n".join(buf), False


def chunk_page(rel: str, raw: str) -> list[dict]:
    title, body = clean(raw)
    records = []
    for s_index, (heading_path, text) in enumerate(sections(title, body)):
        context = " > ".join(heading_path)
        cap = CHUNK_TOKENS - (count_tokens(context) + 2 if context else 0)
        for ordinal, (piece, forced) in enumerate(pack(paragraphs(text), cap)):
            cid = hashlib.sha1(f"{rel}#{s_index}#{ordinal}".encode()).hexdigest()[:16]
            embed_text = f"{context}\n\n{piece}" if context else piece
            records.append({
                "id": cid,
                "path": rel,
                "title": title,
                "heading_path": context,
                "section": s_index,
                "ordinal": ordinal,
                "text": piece,
                "embed_text": embed_text,
                "tokens": count_tokens(embed_text),
                "forced_split": forced,
            })
    return records


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    out = []
    for page in manifest["pages"]:
        rel = page["path"]
        raw = (CORPUS_DIR / rel).read_text(encoding="utf-8")
        out.extend(chunk_page(rel, raw))
    with CHUNKS.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(out)} chunks from {len(manifest['pages'])} pages, cap {CHUNK_TOKENS} tokens, written to {CHUNKS}")


if __name__ == "__main__":
    main()
