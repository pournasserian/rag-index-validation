# rag-index-validation

Checks for a vector database built for retrieval-augmented generation (RAG), in four layers, cheapest first:

1. **The documents and their chunks** (`check_corpus.py`): counts against the source, extraction and markup, encoding, near-duplicates, the chunk length histogram, chunks cut mid-sentence, lost heading context, metadata, chunks per document.
2. **The embeddings and the HNSW index** (`build_index.py`, `check_index.py`): dimensions, finite components and unit norms; the model recorded on the collection against the one configured now; a fresh embedding against the stored one; self-retrieval; recall@10 of the approximate index against a brute-force search over the same vectors, at several `ef_search` settings.
3. **Retrieval against a golden question set** (`golden.jsonl`, `eval_retrieval.py`): hit rate, recall@k, MRR and nDCG with [ranx](https://github.com/AmenRa/ranx), by question type, the misses, and the top-1 similarity of out-of-scope questions against answerable ones. Labels are on pages, not chunks, so they survive re-chunking.
4. **The answer, with a judge** (`judge_faithfulness.py`): a binary faithfulness judge (split the answer into claims, check each against the retrieved context) with a planted false sentence as the control. The judge is a different model from the generator, and both are pinned by id.

The sample corpus is the Markdown of the [MCP specification](https://github.com/modelcontextprotocol/modelcontextprotocol) (revision 2026-07-28, 31 pages, Apache-2.0), fetched at a pinned commit by `fetch_corpus.py` and never committed here. The MIT licence covers the code and the hand-written questions in `golden.jsonl`.

This repository accompanies the article [Validating a vector database built for RAG](https://pournasserian.com/writing/validating-a-vector-database-for-rag).

## Run it

Python 3.12 or later. Any OpenAI-compatible endpoint works; the model ids are read from the environment (see `.env.example`).

```bash
uv venv && uv pip install openai chromadb tiktoken ranx datasketch numpy
export OPENAI_API_KEY=sk-...
# For OpenRouter instead of OpenAI:
# export OPENAI_BASE_URL=https://openrouter.ai/api/v1 EMBEDDING_MODEL=openai/text-embedding-3-small \
#        GENERATOR_MODEL=openai/gpt-5-mini JUDGE_MODEL=openai/gpt-5.4-mini

python fetch_corpus.py        # 31 pages into data/corpus/
python chunk.py               # data/chunks.jsonl, structure first, then a 400-token cap
python check_corpus.py        # layer 1; exits 1 on a hard failure
python build_index.py         # embeds the chunks, loads a Chroma collection with an HNSW index
python check_index.py         # layer 2
python eval_retrieval.py --min-recall10 0.9   # layer 3; exits 1 below the threshold
python judge_faithfulness.py --n 6            # layer 4; the only step that calls a chat model
```

`.github/workflows/ci.yml` runs layers 1 to 3 on a pull request. The retrieval metrics need no judge, so a CI run costs one embedding call per chunk and per question.

## What a run looked like

On 23 October 2026, with `text-embedding-3-small` through OpenRouter, 536 chunks from 31 pages:

- Layer 1 flagged the generated schema reference: 77 per cent of its bytes inside HTML tags, 162 of the 536 chunks (the median page has 9), 49 chunks cut by force at the cap, and all 11 near-duplicate pairs inside it.
- Layer 2 found nothing wrong, and recall@10 of the HNSW index against brute force was 0.998 for the golden questions even at `ef_search` 10. On 536 vectors the graph is nearly exact; the check earns its keep at the sizes where it is not.
- Layer 3, 50 answerable questions labelled by page: hit rate@5 0.98, recall@10 0.98, MRR@10 0.86, nDCG@10 0.88. Paraphrased questions were the weak type (hit rate@5 0.88). The five out-of-scope questions had a median top-1 similarity of 0.48 against 0.63 for the answerable ones, with overlapping ranges, so a similarity threshold alone would not separate them.
- Layer 4 (generator `gpt-5-mini`, judge `gpt-5.4-mini`) judged six answers: five scored 1.0 on every claim, and one came back as "I don't know" although its page was in the top 5, which page-level labels cannot see. The control answer with a planted false sentence scored 0.67 and the planted claim was the one flagged. The judge read 35,127 tokens and wrote 1,632 for the seven verdicts.

Your numbers will differ with the model, the chunker and the day the spec was fetched.

## Layout

| File | Layer | What it does |
|---|---|---|
| `common.py` | | Paths, the client, the tokenizer, the model ids from the environment |
| `fetch_corpus.py` | | Downloads the spec pages at a pinned commit, writes `data/manifest.json` |
| `chunk.py` | 1 | Splits pages by heading, then packs paragraphs up to the token cap; deterministic chunk ids |
| `check_corpus.py` | 1 | The corpus and chunk checks |
| `build_index.py` | 2 | Embeds with checks on every response, loads Chroma, saves `data/embeddings.npy` |
| `check_index.py` | 2 | The embedding and index checks, including ANN recall against brute force |
| `golden.jsonl` | 3 | 55 questions: 30 factoid, 6 multi-hop, 8 paraphrase, 6 keyword, 5 out of scope |
| `eval_retrieval.py` | 3 | Retrieval metrics with ranx, results in `results/retrieval.json` |
| `judge_faithfulness.py` | 4 | The binary faithfulness judge, results in `results/faithfulness.json` |

## Licence

MIT, for the code and the questions. The MCP specification pages keep their own licence (Apache-2.0) and are not part of this repository.
