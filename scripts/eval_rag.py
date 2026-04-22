import json
from app.services.rag import RAGService

TESTS = [
    {"query": "我想了解尺码怎么选", "expect_topic": "尺码"},
    {"query": "可以退货吗", "expect_topic": "售后"},
    {"query": "会员有什么权益", "expect_topic": "会员"},
]

if __name__ == "__main__":
    rag = RAGService()
    passed = 0
    for t in TESTS:
        docs = rag.query(t["query"], top_k=3)
        hit = any(t["expect_topic"] in d for d in docs)
        print(t["query"], "->", "PASS" if hit else "FAIL")
        if hit:
            passed += 1
    print(f"Passed {passed}/{len(TESTS)}")
