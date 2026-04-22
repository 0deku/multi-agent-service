import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.agent_orchestrator import AgentOrchestrator
from scripts.eval_data_schema import load_eval_dataset

DATA_PATH = Path("data/tool_eval.json")
REPORT_DIR = Path("reports")


def main():
    REPORT_DIR.mkdir(exist_ok=True)
    orchestrator = AgentOrchestrator()
    tests = load_eval_dataset(DATA_PATH, dataset="tool")

    details = []

    expected_total = 0
    selected_expected = 0
    selected_total = 0
    selected_correct = 0
    forbidden_total = 0
    forbidden_hit = 0
    args_total = 0
    args_valid = 0
    missing_args_expect_total = 0
    missing_args_asked = 0

    for t in tests:
        result = orchestrator.handle_message(session_id=f"eval-tool-{t['id']}", message=t["query"])

        executed_tools = list(result.get("executed_tools", []))
        tool_errors = dict(result.get("tool_errors", {}))
        expected_tools = list(t.get("expected_tools", []))
        forbidden_tools = list(t.get("forbidden_tools", []))
        arg_constraints = dict(t.get("arg_constraints", {}))
        expect_ask = bool(t.get("expect_ask_for_missing_args", False))

        expected_total += len(expected_tools)
        selected_expected += sum(1 for tool in expected_tools if tool in executed_tools)

        selected_total += len(executed_tools)
        selected_correct += sum(1 for tool in executed_tools if tool in expected_tools)

        forbidden_total += len(forbidden_tools)
        forbidden_hit += sum(1 for tool in forbidden_tools if tool in executed_tools)

        valid_count = 0
        total_count = 0
        for tool, required in arg_constraints.items():
            total_count += 1
            if tool in executed_tools:
                err = str(tool_errors.get(tool) or "")
                if not err:
                    valid_count += 1
            elif tool in expected_tools:
                # Expected but not executed counts as invalid.
                pass
            else:
                # Constraint on non-expected tools is ignored.
                valid_count += 1
        args_total += total_count
        args_valid += valid_count

        ask_for_more = "请提供" in str(result.get("response", "")) or any(
            err == "missing_required_args" for err in tool_errors.values()
        )
        if expect_ask:
            missing_args_expect_total += 1
            missing_args_asked += int(ask_for_more)

        details.append(
            {
                "id": t["id"],
                "query": t["query"],
                "expected_tools": expected_tools,
                "executed_tools": executed_tools,
                "forbidden_tools": forbidden_tools,
                "tool_errors": tool_errors,
                "ask_for_more_info": ask_for_more,
            }
        )

    precision = _safe_rate(selected_correct, selected_total)
    recall = _safe_rate(selected_expected, expected_total)

    report = {
        "total": len(tests),
        "tool_selection_precision": precision,
        "tool_selection_recall": recall,
        "tool_arg_valid_rate": _safe_rate(args_valid, args_total),
        "tool_forbidden_violation_rate": _safe_rate(forbidden_hit, forbidden_total),
        "missing_args_ask_rate": _safe_rate(missing_args_asked, missing_args_expect_total),
        "details": details,
    }

    out = REPORT_DIR / "tool_eval_report.json"
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    out.write_text(payload, encoding="utf-8")
    print(payload, flush=True)


def _safe_rate(n: int, d: int) -> float:
    return round(n / max(d, 1), 4)


if __name__ == "__main__":
    main()
