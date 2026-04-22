import json
from app.services.multi_agents import MultiAgentRouter
from app.services.rag import RAGService
from app.services.memory import MemoryStore

if __name__ == "__main__":
    router = MultiAgentRouter()
    rag = RAGService()
    memory = MemoryStore()

    with open("data/router_eval.json", "r", encoding="utf-8") as f:
        tests = json.load(f)

    passed = 0
    for t in tests:
        context = rag.query(t["query"], top_k=2)
        agent, _ = router.route(t["query"], context, memory.get_memory("eval"))
        ok = agent == t["expect"]
        print(t["query"], "->", agent, "PASS" if ok else "FAIL")
        passed += 1 if ok else 0

    print(f"Passed {passed}/{len(tests)}")
