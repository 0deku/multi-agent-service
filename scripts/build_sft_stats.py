import json
from collections import Counter
from pathlib import Path

TRAIN_PATH = Path("data/sft/planner_train_split.jsonl")
VALID_PATH = Path("data/sft/planner_valid_split.jsonl")
HARD_NEG_PATH = Path("data/sft/planner_hard_negative.jsonl")
REPORT_PATH = Path("reports/sft_planner_stats.json")


def main():
    train = _read_jsonl(TRAIN_PATH)
    valid = _read_jsonl(VALID_PATH)
    hard_neg = _read_jsonl(HARD_NEG_PATH)

    report = {
        "train": _summarize(train),
        "valid": _summarize(valid),
        "hard_negative": _summarize(hard_neg),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(REPORT_PATH)}, ensure_ascii=False), flush=True)


def _read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _summarize(rows: list[dict]):
    agent_counter = Counter()
    use_rag_counter = Counter()
    style_counter = Counter()
    tool_counter = Counter()
    weight_counter = Counter()

    for row in rows:
        meta = row.get("meta") or {}
        weight_counter[str(meta.get("sample_weight", 1.0))] += 1

        messages = row.get("messages") or []
        if len(messages) < 3:
            continue

        content = str(messages[2].get("content") or "")
        try:
            plan = json.loads(content)
        except Exception:
            continue

        agent_counter[str(plan.get("agent", "sales"))] += 1
        use_rag_counter[str(bool(plan.get("use_rag", True)))] += 1
        style_counter[str(plan.get("response_style", "简洁"))] += 1
        for tc in plan.get("tool_calls", []):
            tool_counter[str(tc.get("name", "unknown"))] += 1

    return {
        "count": len(rows),
        "agent_distribution": dict(agent_counter),
        "use_rag_distribution": dict(use_rag_counter),
        "response_style_distribution": dict(style_counter),
        "tool_call_distribution": dict(tool_counter),
        "sample_weight_distribution": dict(weight_counter),
        "distinct_agents": sorted(agent_counter.keys()),
    }


if __name__ == "__main__":
    main()
