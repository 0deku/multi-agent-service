from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

AGENT_SET = {"sales", "after_sales", "promo", "inventory"}
TOOL_BY_DOMAIN = {
    "after_sales": "order",
    "inventory": "inventory",
    "promo": "promotion",
    "sales": "inventory",
}


def normalize_example(raw: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    query = str(raw.get("query") or raw.get("user_query") or "").strip()

    primary_intent = str(raw.get("primary_intent") or raw.get("expect") or "sales")
    if primary_intent not in AGENT_SET:
        primary_intent = "sales"

    domain = str(raw.get("domain") or primary_intent)
    if domain not in AGENT_SET:
        domain = primary_intent

    secondary_intents = raw.get("secondary_intents") or []
    if not isinstance(secondary_intents, list):
        secondary_intents = []

    expected_tools = list(raw.get("expected_tools") or _tools_from_tool_type(raw.get("tool_type")) or [])
    if not expected_tools and dataset in {"tool", "e2e"}:
        expected_tools = [TOOL_BY_DOMAIN.get(domain, "inventory")]

    need_tool = bool(raw.get("need_tool", bool(expected_tools)))
    tool_type = list(raw.get("tool_type") or expected_tools)

    required_slots = raw.get("required_slots") or {}
    if not isinstance(required_slots, dict):
        required_slots = {}

    policy_tags = raw.get("policy_tags") or []
    if not isinstance(policy_tags, list):
        policy_tags = []

    quality_flags = raw.get("quality_flags") or []
    if not isinstance(quality_flags, list):
        quality_flags = []

    turns = raw.get("turns") or ([query] if query else [])
    if not isinstance(turns, list):
        turns = [query] if query else []

    must_have = raw.get("must_have_facts") or raw.get("gold_response_keypoints") or []
    must_not = raw.get("must_not_have_facts") or []

    normalized = {
        "id": raw.get("id") or f"{dataset}-{abs(hash(query)) % 10_000_000}",
        "query": query,
        "turns": turns,
        "primary_intent": primary_intent,
        "secondary_intents": secondary_intents,
        "domain": domain,
        "need_tool": need_tool,
        "tool_type": tool_type,
        "required_slots": required_slots,
        "risk_level": str(raw.get("risk_level") or "low"),
        "policy_tags": policy_tags,
        "gold_action_plan": raw.get("gold_action_plan")
        or {
            "agent": primary_intent,
            "use_rag": True,
            "tool_calls": [{"name": t, "args": {"query": query}} for t in expected_tools],
            "response_style": "简洁",
        },
        "gold_response_keypoints": must_have,
        "quality_flags": quality_flags,
        # backward-compatible fields
        "expect": raw.get("expect") or primary_intent,
        "expected_tools": expected_tools,
        "forbidden_tools": list(raw.get("forbidden_tools") or []),
        "arg_constraints": dict(raw.get("arg_constraints") or {}),
        "expect_ask_for_missing_args": bool(raw.get("expect_ask_for_missing_args", False)),
        "must_have_facts": must_have,
        "must_not_have_facts": must_not,
        "expect_topic": raw.get("expect_topic") or _default_expect_topic(domain),
        "scenario": raw.get("scenario") or domain,
        "success_criteria": raw.get("success_criteria") or "给出可执行建议并避免编造",
    }
    return normalized


def load_eval_dataset(path: Path, dataset: str) -> list[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return [normalize_example(r, dataset=dataset) for r in raw]


def _tools_from_tool_type(tool_type: Any) -> list[str]:
    if not tool_type:
        return []
    if isinstance(tool_type, str):
        return [tool_type]
    if isinstance(tool_type, list):
        return [str(x) for x in tool_type]
    return []


def _default_expect_topic(domain: str) -> str:
    if domain == "after_sales":
        return "售后"
    if domain == "promo":
        return "会员"
    if domain == "inventory":
        return "尺码"
    return "鞋"
