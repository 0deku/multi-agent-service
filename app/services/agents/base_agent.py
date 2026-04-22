from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

from app.services.llm_clients import LLMClient
from app.services.schemas import AgentPlan, ToolCall
from app.services.tool_executor import ToolExecutor


@dataclass
class AgentExecutionResult:
    response: str
    tools: Dict[str, Any]
    executed_tools: List[str]
    tool_errors: Dict[str, str] = field(default_factory=dict)
    ask_for_more_info: bool = False


class BaseAgent(ABC):
    name: str = "sales"
    allowed_tools: set[str] = set()

    def __init__(self, llm: LLMClient, tool_executor: ToolExecutor):
        self.llm = llm
        self.tool_executor = tool_executor

    @property
    @abstractmethod
    def role_prompt(self) -> str:
        raise NotImplementedError

    def _filter_tool_calls(self, calls: List[ToolCall], message: str) -> List[ToolCall]:
        filtered: List[ToolCall] = []
        seen = set()
        for call in calls:
            if call.name in self.allowed_tools and call.name not in seen:
                args = dict(call.args or {"query": message})
                if "query" not in args:
                    args["query"] = message
                filtered.append(ToolCall(name=call.name, args=args))
                seen.add(call.name)
        return filtered

    def _build_system_prompt(self, response_style: str) -> str:
        return (
            "你是耐克卖衣服的智能客服，注意礼貌、简洁、给出明确建议。"
            "你要基于给定的知识内容回答，遇到不确定的问题要说明需要进一步信息。"
            f"当前角色：{self.name}。"
            f"回答风格：{response_style}。"
            f"角色职责：{self.role_prompt}。"
            "如果工具返回ok=false，请向用户索要必要信息，而非编造。"
        )

    def _build_user_prompt(
        self,
        message: str,
        context: List[Any],
        tool_context: Dict[str, Any],
        summary: str,
        recent_turns: List[Dict[str, Any]],
    ) -> str:
        return (
            f"用户问题：{message}\n\n"
            f"可用知识：{context}\n\n"
            f"业务工具结果：{tool_context}\n\n"
            f"历史总结：{summary}\n\n"
            f"最近对话：{recent_turns}"
        )

    def handle(
        self,
        message: str,
        plan: AgentPlan,
        context: List[Any],
        summary: str,
        recent_turns: List[Dict[str, Any]],
    ) -> AgentExecutionResult:
        tool_calls = self._filter_tool_calls(plan.tool_calls, message)
        tool_context = self.tool_executor.run_many_with_whitelist(
            message=message,
            calls=tool_calls,
            allowed_tools=self.allowed_tools,
        )

        tool_errors = {
            tool: str(payload.get("error") or "")
            for tool, payload in tool_context.items()
            if not payload.get("ok", False)
        }
        ask_for_more_info = any(err == "missing_required_args" for err in tool_errors.values())

        try:
            response = self.llm.chat(
                self._build_system_prompt(plan.response_style),
                self._build_user_prompt(message, context, tool_context, summary, recent_turns),
            )
        except Exception:
            response = "抱歉，当前系统繁忙，请稍后再试。"

        return AgentExecutionResult(
            response=response,
            tools=tool_context,
            executed_tools=list(tool_context.keys()),
            tool_errors=tool_errors,
            ask_for_more_info=ask_for_more_info,
        )
