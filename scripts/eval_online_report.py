import json
from collections import defaultdict
from pathlib import Path

LOG_PATH = Path("reports/online_events.jsonl")
SLICES_PATH = Path("data/online_slices.json")
REPORT_DIR = Path("reports")


def main():
    REPORT_DIR.mkdir(exist_ok=True)
    events = _load_events(LOG_PATH)
    slices_cfg = _load_json(SLICES_PATH)

    overall = _compute_metrics(events)
    slice_reports = {}

    for s in slices_cfg.get("slices", []):
        name = s.get("name", "unknown")
        filters = s.get("filters", {})
        filtered = [e for e in events if _match_filters(e, filters)]
        slice_reports[name] = _compute_metrics(filtered)

    report = {
        "total_events": len(events),
        "overall": overall,
        "slices": slice_reports,
    }

    out = REPORT_DIR / "online_eval_report.json"
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    out.write_text(payload, encoding="utf-8")
    print(payload, flush=True)


def _load_events(path: Path):
    if not path.exists():
        return []

    events = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _match_filters(event: dict, filters: dict) -> bool:
    for k, v in filters.items():
        if k == "has_tool_error":
            has_err = bool(event.get("tool_errors"))
            if has_err != bool(v):
                return False
            continue

        if k == "min_latency_ms":
            latency = float(event.get("timings", {}).get("total_ms", 0.0))
            if latency < float(v):
                return False
            continue

        if k == "max_latency_ms":
            latency = float(event.get("timings", {}).get("total_ms", 0.0))
            if latency > float(v):
                return False
            continue

        if k == "failure_stage_in":
            failure_stage = str(event.get("failure_stage", "none"))
            allowed = set(v or [])
            if failure_stage not in allowed:
                return False
            continue

        if event.get(k) != v:
            return False
    return True


def _compute_metrics(events: list[dict]):
    total = len(events)
    resolved = sum(1 for e in events if bool(e.get("resolved", False)))
    handoff = sum(1 for e in events if bool(e.get("handoff_to_human", False)))
    tool_error = sum(1 for e in events if bool(e.get("tool_errors")))

    failure_stage_counts = defaultdict(int)
    for e in events:
        failure_stage_counts[str(e.get("failure_stage", "none"))] += 1

    latencies = sorted(float(e.get("timings", {}).get("total_ms", 0.0)) for e in events)
    p95_latency = _p95(latencies)

    costs = [float(e.get("cost_estimate", 0.0)) for e in events if e.get("cost_estimate") is not None]
    avg_cost = round(sum(costs) / max(len(costs), 1), 4) if costs else 0.0

    feedbacks = [e.get("user_feedback") for e in events if e.get("user_feedback") in ["up", "down", 1, 0]]
    if feedbacks:
        good = sum(1 for x in feedbacks if x in ["up", 1])
        csat = round(good / len(feedbacks), 4)
    else:
        csat = None

    return {
        "total": total,
        "resolution_rate": _safe_rate(resolved, total),
        "handoff_rate": _safe_rate(handoff, total),
        "tool_error_rate": _safe_rate(tool_error, total),
        "p95_latency_ms": p95_latency,
        "avg_cost_per_session": avg_cost,
        "csat": csat,
        "failure_stage_breakdown": dict(sorted(failure_stage_counts.items())),
    }


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    idx = min(len(values) - 1, int(round(len(values) * 0.95)) - 1)
    return round(values[max(idx, 0)], 2)


def _safe_rate(n: int, d: int) -> float:
    return round(n / max(d, 1), 4)


if __name__ == "__main__":
    main()
