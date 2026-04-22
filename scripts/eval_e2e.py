import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.agent_orchestrator import AgentOrchestrator
from scripts.eval_data_schema import load_eval_dataset

DATA_PATH = Path("data/e2e_eval.json")
REPORT_DIR = Path("reports")


def main():
    REPORT_DIR.mkdir(exist_ok=True)
    orchestrator = AgentOrchestrator()

    tests = load_eval_dataset(DATA_PATH, dataset="e2e")
    details = []

    success_count = 0
    factual_pass = 0
    hallucination_violations = 0
    policy_violations = 0

    by_domain = defaultdict(lambda: {"total": 0, "success": 0})
    failure_stage_breakdown = defaultdict(int)

    for t in tests:
        session_id = f"eval-e2e-{t['id']}"
        last_result = None
        for turn in t.get("turns", []):
            last_result = orchestrator.handle_message(session_id=session_id, message=turn)

        if not last_result:
            continue

        response = str(last_result.get("response", ""))
        used_tools = list(last_result.get("executed_tools", []))
        expected_tools = list(t.get("expected_tools", []))
        failure_stage = str(last_result.get("failure_stage", "none"))

        has_must = _contains_all(response, t.get("must_have_facts", []))
        has_forbidden = _contains_any(response, t.get("must_not_have_facts", []))
        success = bool(last_result.get("resolved", False)) and has_must and not has_forbidden

        factual_ok = has_must
        hallucination_ok = not has_forbidden
        policy_ok = _policy_ok(response)

        success_count += int(success)
        factual_pass += int(factual_ok)
        hallucination_violations += int(not hallucination_ok)
        policy_violations += int(not policy_ok)

        scenario = t.get("scenario", "unknown")
        by_domain[scenario]["total"] += 1
        by_domain[scenario]["success"] += int(success)

        if not success:
            failure_stage_breakdown[failure_stage] += 1

        details.append(
            {
                "id": t["id"],
                "scenario": scenario,
                "expected_tools": expected_tools,
                "executed_tools": used_tools,
                "resolved": bool(last_result.get("resolved", False)),
                "resolved_reason": last_result.get("resolved_reason", ""),
                "failure_stage": failure_stage,
                "factual_ok": factual_ok,
                "hallucination_ok": hallucination_ok,
                "policy_ok": policy_ok,
                "success": success,
            }
        )

    total = len(tests)
    report = {
        "total": total,
        "task_success_rate": _safe_rate(success_count, total),
        "factual_pass_rate": _safe_rate(factual_pass, total),
        "hallucination_rate": _safe_rate(hallucination_violations, total),
        "policy_violation_rate": _safe_rate(policy_violations, total),
        "failure_stage_breakdown": dict(sorted(failure_stage_breakdown.items())),
        "per_domain": {
            d: {
                "total": v["total"],
                "success": v["success"],
                "task_success_rate": _safe_rate(v["success"], v["total"]),
            }
            for d, v in by_domain.items()
        },
        "details": details,
    }

    out = REPORT_DIR / "e2e_eval_report.json"
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    out.write_text(payload, encoding="utf-8")
    print(payload, flush=True)


def _contains_all(text: str, tokens: list[str]) -> bool:
    return all(tok in text for tok in tokens)


def _contains_any(text: str, tokens: list[str]) -> bool:
    return any(tok in text for tok in tokens)


def _policy_ok(text: str) -> bool:
    blocked = ["保证100%", "绝对", "必定"]
    return not any(tok in text for tok in blocked)


def _safe_rate(n: int, d: int) -> float:
    return round(n / max(d, 1), 4)


if __name__ == "__main__":
    main()
