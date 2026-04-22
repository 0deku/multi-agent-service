import json
from pathlib import Path

from app.services.rag import RAGService

DATA_PATH = Path("data/rag_eval.json")


def main():
    rag = RAGService()
    tests = _load_tests()

    hybrid_hit = 0
    vector_hit = 0
    bm25_hit = 0

    for t in tests:
        result = rag.query_with_debug(t["query"], top_k=3, candidate_k=8)

        expect = t["expect_topic"]
        hybrid_ok = any(expect in d for d in result["docs"])
        vector_ok = any(expect in d for d in result["vector_docs"][:3])
        bm25_ok = any(expect in d for d in result["bm25_docs"][:3])

        hybrid_hit += int(hybrid_ok)
        vector_hit += int(vector_ok)
        bm25_hit += int(bm25_ok)

        print(
            f"{t['query']} -> hybrid={'PASS' if hybrid_ok else 'FAIL'} "
            f"vector={'PASS' if vector_ok else 'FAIL'} bm25={'PASS' if bm25_ok else 'FAIL'}"
        )

    total = len(tests)
    print(f"Hybrid Hit@3: {hybrid_hit}/{total} = {hybrid_hit / max(total,1):.4f}")
    print(f"Vector Hit@3: {vector_hit}/{total} = {vector_hit / max(total,1):.4f}")
    print(f"BM25 Hit@3: {bm25_hit}/{total} = {bm25_hit / max(total,1):.4f}")


def _load_tests():
    with DATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    main()
