"""Layer 4: a binary faithfulness judge for generated answers.

For a sample of golden questions: retrieve the top chunks, have the generator answer from
them only, then have a different model judge the answer in two steps (split it into claims,
check each claim against the retrieved context, supported or not). Faithfulness is the share
of supported claims. One control answer with a planted false sentence shows what the judge
catches. Model ids are pinned in common.py; the generator and the judge are different models.

Usage: python judge_faithfulness.py [--n 8]
"""

import argparse
import json
import random

import chromadb

from common import CHROMA_DIR, COLLECTION, EMBEDDING_MODEL, GENERATOR_MODEL, GOLDEN, JUDGE_MODEL, ROOT, client

RESULTS = ROOT / "results"
TOP_K = 5
random.seed(11)

GENERATE = (
    "Answer the question using only the context below. If the context does not contain the "
    "answer, say exactly: I don't know. Be brief.\n\nContext:\n{context}\n\nQuestion: {question}"
)
CLAIMS = (
    "Split the answer into its atomic factual claims, one per item, each a complete sentence "
    "that stands on its own. Return JSON: {{\"claims\": [\"...\", ...]}}. If the answer is only "
    "\"I don't know\", return an empty list.\n\nAnswer:\n{answer}"
)
VERIFY = (
    "You check whether a claim is supported by a context. A claim is supported only if it can "
    "be inferred from the context; general knowledge does not count. Return JSON: "
    "{{\"supported\": true or false, \"reason\": \"one sentence\"}}.\n\nContext:\n{context}\n\nClaim: {claim}"
)
PLANTED = " The protocol also requires every server to run on port 8080."


def chat(oai, model: str, prompt: str, json_mode: bool = False) -> tuple[str, int, int]:
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    r = oai.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], **kwargs)
    u = r.usage
    return r.choices[0].message.content or "", u.prompt_tokens, u.completion_tokens


def judge(oai, context: str, answer: str) -> dict:
    tokens = [0, 0]
    raw, p, c = chat(oai, JUDGE_MODEL, CLAIMS.format(answer=answer), json_mode=True)
    tokens[0] += p; tokens[1] += c
    claims = json.loads(raw).get("claims", [])
    verdicts = []
    for claim in claims:
        raw, p, c = chat(oai, JUDGE_MODEL, VERIFY.format(context=context, claim=claim), json_mode=True)
        tokens[0] += p; tokens[1] += c
        v = json.loads(raw)
        verdicts.append({"claim": claim, "supported": bool(v.get("supported")), "reason": v.get("reason", "")})
    supported = sum(v["supported"] for v in verdicts)
    score = supported / len(verdicts) if verdicts else None
    return {"claims": verdicts, "faithfulness": score, "judge_tokens": tokens}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    golden = [g for g in (json.loads(l) for l in GOLDEN.open(encoding="utf-8")) if g["answerable"]]
    sample = random.sample(golden, args.n)
    col = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION)
    oai = client()
    print(f"generator {GENERATOR_MODEL}, judge {JUDGE_MODEL}, top {TOP_K} chunks\n")

    out, gen_tokens = [], [0, 0]
    for i, g in enumerate(sample):
        q = oai.embeddings.create(model=EMBEDDING_MODEL, input=[g["question"]]).data[0].embedding
        res = col.query(query_embeddings=[q], n_results=TOP_K, include=["documents", "metadatas"])
        context = "\n\n---\n\n".join(f"[{m['heading_path']}]\n{d}" for d, m in zip(res["documents"][0], res["metadatas"][0]))
        answer, p, c = chat(oai, GENERATOR_MODEL, GENERATE.format(context=context, question=g["question"]))
        gen_tokens[0] += p; gen_tokens[1] += c
        verdict = judge(oai, context, answer)
        out.append({"id": g["id"], "question": g["question"], "answer": answer, **verdict})
        score = "n/a" if verdict["faithfulness"] is None else f"{verdict['faithfulness']:.2f}"
        print(f"{g['id']}: faithfulness {score} ({sum(v['supported'] for v in verdict['claims'])}/{len(verdict['claims'])} claims)")
        print(f"   Q: {g['question']}\n   A: {answer[:200]}{'...' if len(answer) > 200 else ''}")
        for v in verdict["claims"]:
            if not v["supported"]:
                print(f"   not supported: {v['claim']}  ({v['reason']})")
        if i == 0:
            planted = answer.rstrip() + PLANTED
            control = judge(oai, context, planted)
            out.append({"id": g["id"] + "-planted", "question": g["question"], "answer": planted, **control})
            caught = [v for v in control["claims"] if not v["supported"]]
            print(f"   control with a planted false sentence: faithfulness {control['faithfulness']:.2f}; flagged: {[v['claim'] for v in caught]}")
        print()

    judge_tokens = [sum(o["judge_tokens"][0] for o in out), sum(o["judge_tokens"][1] for o in out)]
    scores = [o["faithfulness"] for o in out if o["faithfulness"] is not None and not o["id"].endswith("-planted")]
    print(f"mean faithfulness over {len(scores)} answers: {sum(scores) / len(scores):.3f}")
    print(f"tokens: generator {gen_tokens[0]} in / {gen_tokens[1]} out; judge {judge_tokens[0]} in / {judge_tokens[1]} out")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "faithfulness.json").write_text(json.dumps({"generator": GENERATOR_MODEL, "judge": JUDGE_MODEL, "samples": out, "generator_tokens": gen_tokens, "judge_tokens": judge_tokens}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
