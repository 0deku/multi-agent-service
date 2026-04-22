import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.agent_orchestrator import AgentOrchestrator
from app.services.multi_agents import MultiAgentRouter
from app.services.planner import Planner
from app.services.rag import RAGService
from scripts.eval_data_schema import load_eval_dataset

REPORT_DIR = Path("reports")
AGENT_PATH = Path("data/agent_eval.json")
ROUTER_PATH = Path("data/router_eval.json")
RAG_PATH = Path("data/rag_eval.json")
E2E_PATH = Path("data/e2e_eval.json")
TOOL_PATH = Path("data/tool_eval.json")


def eval_agent(planner: Planner):
    tests = load_eval_dataset(AGENT_PATH, dataset="agent")
    correct = 0
    details = []
    by_domain = defaultdict(lambda: {"total": 0, "correct": 0})

    for t in tests:
        plan = planner.plan(t["query"], "", [])
        ok = plan.agent == t["expect"]
        correct += int(ok)
        domain = t.get("domain", t["expect"])
        by_domain[domain]["total"] += 1
        by_domain[domain]["correct"] += int(ok)
        details.append(
            {
                "query": t["query"],
                "domain": domain,
                "expect": t["expect"],
                "got": plan.agent,
                "ok": ok,
            }
        )

    return {
        "total": len(tests),
        "correct": correct,
        "accuracy": round(correct / max(len(tests), 1), 4),
        "per_domain": _acc_by_domain(by_domain),
        "details": details,
    }


def eval_router(router: MultiAgentRouter, rag: RAGService):
    tests = load_eval_dataset(ROUTER_PATH, dataset="router")
    correct = 0
    details = []
    by_domain = defaultdict(lambda: {"total": 0, "correct": 0})

    for t in tests:
        context = rag.query(t["query"], top_k=2)
        label, _ = router.route(t["query"], context, {"summary": "", "turns": []})
        ok = label == t["expect"]
        correct += int(ok)
        domain = t.get("domain", t["expect"])
        by_domain[domain]["total"] += 1
        by_domain[domain]["correct"] += int(ok)
        details.append(
            {
                "query": t["query"],
                "domain": domain,
                "expect": t["expect"],
                "got": label,
                "ok": ok,
            }
        )

    return {
        "total": len(tests),
        "correct": correct,
        "accuracy": round(correct / max(len(tests), 1), 4),
        "per_domain": _acc_by_domain(by_domain),
        "details": details,
    }


def eval_rag(rag: RAGService):
    tests = load_eval_dataset(RAG_PATH, dataset="rag")
    hybrid_hits = 0
    vector_hits = 0
    bm25_hits = 0
    details = []
    by_domain = defaultdict(lambda: {"total": 0, "hit": 0})

    for t in tests:
        result = rag.query_with_debug(t["query"], top_k=3, candidate_k=8)
        docs = result["docs"]
        ok = any(t["expect_topic"] in d for d in docs)
        hybrid_hits += int(ok)

        vector_ok = any(t["expect_topic"] in d for d in result.get("vector_docs", [])[:3])
        bm25_ok = any(t["expect_topic"] in d for d in result.get("bm25_docs", [])[:3])
        vector_hits += int(vector_ok)
        bm25_hits += int(bm25_ok)

        domain = t.get("domain", "unknown")
        by_domain[domain]["total"] += 1
        by_domain[domain]["hit"] += int(ok)
        details.append(
            {
                "query": t["query"],
                "domain": domain,
                "expect_topic": t["expect_topic"],
                "hybrid_ok": ok,
                "vector_ok": vector_ok,
                "bm25_ok": bm25_ok,
                "fallback": result.get("fallback", ""),
            }
        )

    total = len(tests)
    return {
        "total": total,
        "hit": hybrid_hits,
        "hit_rate": round(hybrid_hits / max(total, 1), 4),
        "vector_hit_rate": round(vector_hits / max(total, 1), 4),
        "bm25_hit_rate": round(bm25_hits / max(total, 1), 4),
        "hybrid_hit_rate": round(hybrid_hits / max(total, 1), 4),
        "per_domain": _hit_by_domain(by_domain),
        "details": details,
    }


def eval_e2e(orchestrator: AgentOrchestrator):
    tests = load_eval_dataset(E2E_PATH, dataset="e2e")
    details = []

    success_count = 0
    factual_pass = 0
    hallucination_violations = 0
    policy_violations = 0
    failure_stage_breakdown = defaultdict(int)

    for t in tests:
        session_id = f"full-e2e-{t['id']}"
        last_result = None
        for turn in t.get("turns", []):
            last_result = orchestrator.handle_message(session_id=session_id, message=turn)

        if not last_result:
            continue

        response = str(last_result.get("response", ""))
        has_must = all(tok in response for tok in t.get("must_have_facts", []))
        has_forbidden = any(tok in response for tok in t.get("must_not_have_facts", []))
        policy_ok = not any(tok in response for tok in ["保证100%", "绝对", "必定"])

        success = bool(last_result.get("resolved", False)) and has_must and not has_forbidden
        success_count += int(success)
        factual_pass += int(has_must)
        hallucination_violations += int(has_forbidden)
        policy_violations += int(not policy_ok)

        if not success:
            failure_stage_breakdown[str(last_result.get("failure_stage", "none"))] += 1

        details.append(
            {
                "id": t["id"],
                "scenario": t.get("scenario", "unknown"),
                "resolved": bool(last_result.get("resolved", False)),
                "resolved_reason": last_result.get("resolved_reason", ""),
                "failure_stage": last_result.get("failure_stage", "none"),
                "success": success,
            }
        )

    total = len(tests)
    return {
        "total": total,
        "task_success_rate": _safe_rate(success_count, total),
        "factual_pass_rate": _safe_rate(factual_pass, total),
        "hallucination_rate": _safe_rate(hallucination_violations, total),
        "policy_violation_rate": _safe_rate(policy_violations, total),
        "failure_stage_breakdown": dict(sorted(failure_stage_breakdown.items())),
        "details": details,
    }


def eval_tool_quality(orchestrator: AgentOrchestrator):
    tests = load_eval_dataset(TOOL_PATH, dataset="tool")
    details = []

    expected_total = 0
    selected_expected = 0
    selected_total = 0
    selected_correct = 0
    forbidden_total = 0
    forbidden_hit = 0
    args_total = 0
    args_valid = 0

    for t in tests:
        result = orchestrator.handle_message(session_id=f"full-tool-{t['id']}", message=t["query"])
        executed_tools = list(result.get("executed_tools", []))
        tool_errors = dict(result.get("tool_errors", {}))

        expected_tools = list(t.get("expected_tools", []))
        forbidden_tools = list(t.get("forbidden_tools", []))
        arg_constraints = dict(t.get("arg_constraints", {}))

        expected_total += len(expected_tools)
        selected_expected += sum(1 for tool in expected_tools if tool in executed_tools)

        selected_total += len(executed_tools)
        selected_correct += sum(1 for tool in executed_tools if tool in expected_tools)

        forbidden_total += len(forbidden_tools)
        forbidden_hit += sum(1 for tool in forbidden_tools if tool in executed_tools)

        for tool in arg_constraints.keys():
            args_total += 1
            if tool in executed_tools and not str(tool_errors.get(tool) or ""):
                args_valid += 1

        details.append(
            {
                "id": t["id"],
                "expected_tools": expected_tools,
                "executed_tools": executed_tools,
                "tool_errors": tool_errors,
                "failure_stage": result.get("failure_stage", "none"),
            }
        )

    return {
        "total": len(tests),
        "tool_selection_precision": _safe_rate(selected_correct, selected_total),
        "tool_selection_recall": _safe_rate(selected_expected, expected_total),
        "tool_arg_valid_rate": _safe_rate(args_valid, args_total),
        "tool_forbidden_violation_rate": _safe_rate(forbidden_hit, forbidden_total),
        "details": details,
    }


def main():
    REPORT_DIR.mkdir(exist_ok=True)
    planner = Planner()
    router = MultiAgentRouter()
    rag = RAGService()
    orchestrator = AgentOrchestrator()

    report = {
        "dataset_sizes": {
            "agent": len(_load_json(AGENT_PATH)),
            "router": len(_load_json(ROUTER_PATH)),
            "rag": len(_load_json(RAG_PATH)),
            "e2e": len(_load_json(E2E_PATH)),
            "tool": len(_load_json(TOOL_PATH)),
            "total": len(_load_json(AGENT_PATH))
            + len(_load_json(ROUTER_PATH))
            + len(_load_json(RAG_PATH))
            + len(_load_json(E2E_PATH))
            + len(_load_json(TOOL_PATH)),
        },
        "agent": eval_agent(planner),
        "router": eval_router(router, rag),
        "rag": eval_rag(rag),
        "e2e": eval_e2e(orchestrator),
        "tool_quality": eval_tool_quality(orchestrator),
    }

    out = REPORT_DIR / "full_eval_report.json"
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    out.write_text(payload, encoding="utf-8")
    print(payload, flush=True)


def _acc_by_domain(stats):
    result = {}
    for d, v in stats.items():
        total = v["total"]
        correct = v["correct"]
        result[d] = {
            "total": total,
            "correct": correct,
            "accuracy": round(correct / max(total, 1), 4),
        }
    return result


def _hit_by_domain(stats):
    result = {}
    for d, v in stats.items():
        total = v["total"]
        hit = v["hit"]
        result[d] = {
            "total": total,
            "hit": hit,
            "hit_rate": round(hit / max(total, 1), 4),
        }
    return result


def _safe_rate(n: int, d: int) -> float:
    return round(n / max(d, 1), 4)


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"eval_full_error: {exc}", flush=True)
        raise
