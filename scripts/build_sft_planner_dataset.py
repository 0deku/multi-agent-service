import json
import random
from collections import Counter
from pathlib import Path

INPUT_FILES = [
    Path("data/agent_eval.json"),
    Path("data/tool_eval.json"),
    Path("data/e2e_eval.json"),
]
ONLINE_LOG_PATH = Path("reports/online_events.jsonl")
OUTPUT_PATH = Path("data/sft/planner_train.jsonl")
HARD_NEGATIVE_PATH = Path("data/sft/planner_hard_negative.jsonl")
TRAIN_SPLIT_PATH = Path("data/sft/planner_train_split.jsonl")
VALID_SPLIT_PATH = Path("data/sft/planner_valid_split.jsonl")
VALID_RATIO = 0.1
RANDOM_SEED = 42


def main():
    records = []
    hard_negatives = []

    records.extend(_from_agent_eval(Path("data/agent_eval.json")))
    records.extend(_from_tool_eval(Path("data/tool_eval.json")))
    records.extend(_from_e2e_eval(Path("data/e2e_eval.json")))

    online_pos, online_neg = _from_online_logs(ONLINE_LOG_PATH)
    records.extend(online_pos)
    hard_negatives.extend(online_neg)

    # de-duplicate by input+target
    dedup = {}
    for r in records:
        key = json.dumps({"messages": r["messages"]}, ensure_ascii=False)
        dedup[key] = r
    output_records = list(dedup.values())

    neg_dedup = {}
    for r in hard_negatives:
        key = json.dumps({"messages": r["messages"]}, ensure_ascii=False)
        neg_dedup[key] = r
    output_neg = list(neg_dedup.values())

    weighted_records = [_apply_sample_weight(r) for r in output_records]
    weighted_neg = [_apply_sample_weight(r) for r in output_neg]

    train_records, valid_records = _split_train_valid(weighted_records, VALID_RATIO, RANDOM_SEED)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(OUTPUT_PATH, weighted_records)
    _write_jsonl(HARD_NEGATIVE_PATH, weighted_neg)
    _write_jsonl(TRAIN_SPLIT_PATH, train_records)
    _write_jsonl(VALID_SPLIT_PATH, valid_records)

    summary = {
        "output": str(OUTPUT_PATH),
        "hard_negative_output": str(HARD_NEGATIVE_PATH),
        "train_split_output": str(TRAIN_SPLIT_PATH),
        "valid_split_output": str(VALID_SPLIT_PATH),
        "count": len(weighted_records),
        "hard_negative_count": len(weighted_neg),
        "train_count": len(train_records),
        "valid_count": len(valid_records),
        "valid_ratio": VALID_RATIO,
        "source_files": [str(p) for p in INPUT_FILES] + [str(ONLINE_LOG_PATH)],
        "distribution": _distribution(weighted_records),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def _from_agent_eval(path: Path):
    if not path.exists():
        return []
    data = _load_json(path)
    records = []
    for t in data:
        query = str(t.get("query", "")).strip()
        expect = str(t.get("expect", "sales")).strip() or "sales"
        target = {
            "agent": expect,
            "use_rag": True,
            "tool_calls": _default_tool_calls_for_agent(expect, query),
            "response_style": "简洁",
        }
        records.append(_build_record(query, target, memory_summary="", recent_turns=[]))
    return records


def _from_tool_eval(path: Path):
    if not path.exists():
        return []
    data = _load_json(path)
    records = []
    for t in data:
        query = str(t.get("query", "")).strip()
        expected_tools = list(t.get("expected_tools", []))
        agent = _agent_from_tools(expected_tools)
        target = {
            "agent": agent,
            "use_rag": True,
            "tool_calls": [{"name": name, "args": {"query": query}} for name in expected_tools],
            "response_style": "简洁",
        }
        records.append(_build_record(query, target, memory_summary="", recent_turns=[]))
    return records


def _from_e2e_eval(path: Path):
    if not path.exists():
        return []
    data = _load_json(path)
    records = []
    for t in data:
        turns = list(t.get("turns", []))
        scenario = str(t.get("scenario", "sales"))
        expected_tools = list(t.get("expected_tools", []))
        for turn in turns:
            query = str(turn).strip()
            target = {
                "agent": _normalize_agent(scenario),
                "use_rag": True,
                "tool_calls": [{"name": name, "args": {"query": query}} for name in expected_tools],
                "response_style": "简洁",
            }
            records.append(_build_record(query, target, memory_summary="", recent_turns=[]))
    return records


def _from_online_logs(path: Path):
    if not path.exists():
        return [], []

    pos_records = []
    neg_records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            query = str(event.get("query") or event.get("message") or "").strip()
            if not query:
                continue

            plan = event.get("plan") or {}
            agent = _normalize_agent(str(plan.get("agent") or event.get("agent") or "sales"))
            use_rag = bool(plan.get("use_rag", True))
            tool_calls = plan.get("tool_calls") or []
            if not tool_calls:
                tool_calls = [{"name": t, "args": {"query": query}} for t in event.get("executed_tools", [])]

            target = {
                "agent": agent,
                "use_rag": use_rag,
                "tool_calls": tool_calls,
                "response_style": str(plan.get("response_style") or "简洁"),
            }
            memory_summary = str(event.get("memory_summary") or "")
            recent_turns = event.get("recent_turns") or []
            record = _build_record(query, target, memory_summary=memory_summary, recent_turns=recent_turns)

            resolved = bool(event.get("resolved", False))
            has_tool_error = bool(event.get("tool_errors"))
            failure_stage = str(event.get("failure_stage", "none"))

            if resolved and not has_tool_error and failure_stage in {"none", ""}:
                pos_records.append(record)
            else:
                neg_record = {
                    **record,
                    "meta": {
                        "resolved": resolved,
                        "has_tool_error": has_tool_error,
                        "failure_stage": failure_stage,
                        "sample_weight": 1.5,
                    },
                }
                neg_records.append(neg_record)

    return pos_records, neg_records


def _build_record(query: str, target: dict, memory_summary: str, recent_turns: list):
    system = (
        "你是电商智能客服的任务规划器。"
        "请产出一个JSON对象，字段包括: agent(sales/after_sales/promo/inventory), "
        "use_rag(true/false), tool_calls(数组, 每项含name和args), response_style(简洁/正式/活泼)。"
        "只输出JSON，不要其他内容。"
    )
    user = (
        f"用户问题：{query}\n"
        f"历史总结：{memory_summary}\n"
        f"最近对话：{recent_turns}\n"
        "可用工具名称: inventory, order, promotion。"
    )

    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ]
    }


def _apply_sample_weight(record: dict) -> dict:
    record = dict(record)
    meta = dict(record.get("meta") or {})

    weight = float(meta.get("sample_weight", 1.0))

    try:
        plan = json.loads(record["messages"][2]["content"])
    except Exception:
        plan = {}

    agent = str(plan.get("agent") or "sales")
    tools = [str(tc.get("name", "")) for tc in plan.get("tool_calls", [])]

    if agent in {"after_sales", "inventory"}:
        weight = max(weight, 1.2)
    if len(tools) >= 2:
        weight = max(weight, 1.3)

    meta["sample_weight"] = round(weight, 2)
    record["meta"] = meta
    return record


def _split_train_valid(records: list[dict], valid_ratio: float, seed: int):
    if not records:
        return [], []

    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)

    valid_size = int(len(shuffled) * valid_ratio)
    if valid_size <= 0 and len(shuffled) > 1:
        valid_size = 1

    valid = shuffled[:valid_size]
    train = shuffled[valid_size:]
    return train, valid


def _write_jsonl(path: Path, records: list[dict]):
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _default_tool_calls_for_agent(agent: str, query: str):
    if agent == "after_sales":
        return [{"name": "order", "args": {"query": query}}]
    if agent == "promo":
        return [{"name": "promotion", "args": {"query": query}}]
    if agent == "inventory":
        return [{"name": "inventory", "args": {"query": query}}]
    return []


def _agent_from_tools(tools: list[str]) -> str:
    s = set(tools)
    if "order" in s:
        return "after_sales"
    if "promotion" in s and "inventory" not in s:
        return "promo"
    if "inventory" in s and "promotion" not in s:
        return "inventory"
    return "sales"


def _normalize_agent(agent: str) -> str:
    if agent in {"sales", "after_sales", "promo", "inventory"}:
        return agent
    return "sales"


def _distribution(records: list[dict]) -> dict:
    agent_counter = Counter()
    use_rag_counter = Counter()
    tool_counter = Counter()
    weight_counter = Counter()

    for r in records:
        meta = r.get("meta") or {}
        weight_counter[str(meta.get("sample_weight", 1.0))] += 1

        try:
            plan = json.loads(r["messages"][2]["content"])
        except Exception:
            continue
        agent_counter[str(plan.get("agent", "sales"))] += 1
        use_rag_counter[str(bool(plan.get("use_rag", True)))] += 1
        for tc in plan.get("tool_calls", []):
            tool_counter[str(tc.get("name", "unknown"))] += 1

    return {
        "agent": dict(agent_counter),
        "use_rag": dict(use_rag_counter),
        "tool_calls": dict(tool_counter),
        "sample_weight": dict(weight_counter),
    }


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    main()
