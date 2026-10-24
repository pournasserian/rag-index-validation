"""Shared settings: paths, the OpenAI-compatible client, the tokenizer.

Everything that names a model or an endpoint is read from the environment, so the same
code runs against OpenAI, OpenRouter or any other OpenAI-compatible server.
"""

import os
import sys
from pathlib import Path

import tiktoken
from openai import OpenAI

# The reports print the corpus's own characters; Windows consoles default to a legacy code page.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
MANIFEST = DATA_DIR / "manifest.json"
CHUNKS = DATA_DIR / "chunks.jsonl"
EMBEDDINGS = DATA_DIR / "embeddings.npy"
CHROMA_DIR = DATA_DIR / "chroma"
GOLDEN = ROOT / "golden.jsonl"
COLLECTION = "mcp-spec"

# The model ids as OpenAI names them; OpenRouter prefixes them with "openai/".
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = 1536
GENERATOR_MODEL = os.environ.get("GENERATOR_MODEL", "gpt-5-mini")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-5.4-mini")

# Chunking: a hard cap in the embedding model's own tokens, well under its 8192 limit.
CHUNK_TOKENS = 400
ENCODING = tiktoken.get_encoding("cl100k_base")


def client() -> OpenAI:
    """An OpenAI client; OPENAI_BASE_URL and OPENAI_API_KEY pick the endpoint."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("set OPENAI_API_KEY (and OPENAI_BASE_URL for a non-OpenAI endpoint)")
    return OpenAI()


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))
