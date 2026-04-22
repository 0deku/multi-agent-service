import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.planner import Planner
from app.services.multi_agents import MultiAgentRouter
from app.services.rag import RAGService
from scripts.eval_data_schema import load_eval_dataset

DATA_PATH = Path("data/agent_eval.json")
REPORT_DIR = Path("reports")


def main():
    REPORT_DIR.mkdir(exist_ok=True)
    planner = Planner()
    router = MultiAgentRouter()
    rag = RAGService()

    tests = load_eval_dataset(DATA_PATH, dataset="agent")

    correct = 0
    details = []
    for t in tests:
        plan = planner.plan(t["query"], "", [])
        context = rag.query(t["query"]) if plan.use_rag else []
        route, _ = router.route(t["query"], context, {"summary": ""})

        ok = plan.agent == t["expect"]
        correct += 1 if ok else 0
        details.append(
            {
                "query": t["query"],
                "expect": t["expect"],
                "plan_agent": plan.agent,
                "router_agent": route,
                "use_rag": plan.use_rag,
                "ok": ok,
            }
        )
        time.sleep(0.2)

    report = {
        "total": len(tests),
        "correct": correct,
        "accuracy": round(correct / max(len(tests), 1), 4),
        "details": details,
    }

    out = REPORT_DIR / "agent_eval_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
