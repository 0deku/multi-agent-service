import logging
import time
from typing import Any, Dict

from app.services.agents.registry import build_agent_registry
from app.services.llm_clients import LLMClient
from app.services.memory import MemoryStore
from app.services.planner import Planner
from app.services.rag import RAGService
from app.services.tool_executor import ToolExecutor
from app.utils.trace_utils import new_trace_id

logger = logging.getLogger("orchestrator")


class AgentOrchestrator:
    def __init__(self):
        self.rag = RAGService()
        self.memory = MemoryStore()
        self.planner = Planner()
        self.llm = LLMClient()
        self.tool_executor = ToolExecutor()
        self.agent_registry = build_agent_registry(self.llm, self.tool_executor)

    def handle_message(self, session_id: str, message: str):
        trace_id = new_trace_id()
        timings: Dict[str, float] = {}
        total_start = time.perf_counter()

        failure_stage = "none"
        fallback_reason = ""

        memory = self.memory.get_memory(session_id)
        recent = self.memory.recent_turns(session_id, limit=6)
        summary = memory.get("summary", "")

        start = time.perf_counter()
        try:
            plan = self.planner.plan(message, summary, recent)
        except Exception as exc:
            failure_stage = "plan"
            logger.exception("planner_error session=%s trace=%s", session_id, trace_id)
            return self._build_error_response(
                session_id=session_id,
                trace_id=trace_id,
                message=message,
                summary=summary,
                timings=timings,
                total_start=total_start,
                failure_stage=failure_stage,
                fallback_reason=f"planner_error:{exc}",
            )
        timings["plan_ms"] = _elapsed_ms(start)

        start = time.perf_counter()
        rag_debug = {"docs": []}
        try:
            if plan.use_rag:
                rag_debug = self.rag.query_with_debug(message)
                context = rag_debug.get("docs", [])
            else:
                context = []
        except Exception as exc:
            logger.exception("rag_error session=%s trace=%s", session_id, trace_id)
            failure_stage = "rag"
            fallback_reason = f"rag_error:{exc}"
            context = []
        timings["rag_ms"] = _elapsed_ms(start)

        agent_name = plan.agent
        selected_agent = self.agent_registry.get(agent_name) or self.agent_registry["sales"]

        start = time.perf_counter()
        try:
            execution = selected_agent.handle(
                message=message,
                plan=plan,
                context=context,
                summary=summary,
                recent_turns=recent,
            )
        except Exception as exc:
            logger.exception("agent_execution_error session=%s trace=%s agent=%s", session_id, trace_id, agent_name)
            fallback_reason = f"agent_error:{exc}"
            failure_stage = "response"
            selected_agent = self.agent_registry["sales"]
            execution = selected_agent.handle(
                message=message,
                plan=plan,
                context=context,
                summary=summary,
                recent_turns=recent,
            )
            agent_name = "sales"
        timings["agent_exec_ms"] = _elapsed_ms(start)

        tool_context = execution.tools
        agent_response = execution.response
        executed_tools = execution.executed_tools
        tool_errors = execution.tool_errors

        if tool_errors and failure_stage == "none":
            failure_stage = "tool"

        if tool_context:
            logger.info(
                "tools_used=%s session=%s trace=%s tool_errors=%s",
                executed_tools,
                session_id,
                trace_id,
                tool_errors,
            )

        start = time.perf_counter()
        self.memory.update_memory(session_id, message, agent_response)
        timings["memory_write_ms"] = _elapsed_ms(start)

        summary_prompt = (
            "你是对话总结助手，请用中文简洁总结用户偏好、尺码、预算和关键需求。"
            "只输出总结内容，不要加多余解释。"
        )
        summary_input = f"已有总结：{summary}\n\n新增对话：用户={message}；客服={agent_response}"

        start = time.perf_counter()
        try:
            new_summary = self.llm.chat(summary_prompt, summary_input)
        except Exception:
            new_summary = summary
        self.memory.update_summary(session_id, new_summary)
        timings["summary_ms"] = _elapsed_ms(start)

        timings["total_ms"] = _elapsed_ms(total_start)
        resolved, resolved_reason = _estimate_resolved(
            agent_response,
            execution.ask_for_more_info,
            tool_errors,
            failure_stage,
        )

        if not resolved and failure_stage == "none":
            failure_stage = "response"

        logger.info(
            "request_done session=%s trace=%s agent=%s executed_tools=%s fallback_reason=%s "
            "failure_stage=%s resolved=%s resolved_reason=%s timings=%s",
            session_id,
            trace_id,
            agent_name,
            executed_tools,
            fallback_reason,
            failure_stage,
            resolved,
            resolved_reason,
            timings,
        )

        return {
            "session_id": session_id,
            "trace_id": trace_id,
            "agent": agent_name,
            "response": agent_response,
            "context": context,
            "rag_debug": rag_debug,
            "tools": tool_context,
            "executed_tools": executed_tools,
            "tool_errors": tool_errors,
            "fallback_reason": fallback_reason,
            "failure_stage": failure_stage,
            "resolved": resolved,
            "resolved_reason": resolved_reason,
            "plan": plan.model_dump(),
            "timings": timings,
        }

    def _build_error_response(
        self,
        session_id: str,
        trace_id: str,
        message: str,
        summary: str,
        timings: Dict[str, float],
        total_start: float,
        failure_stage: str,
        fallback_reason: str,
    ) -> Dict[str, Any]:
        agent_response = "抱歉，当前系统繁忙，请稍后再试。"
        start = time.perf_counter()
        self.memory.update_memory(session_id, message, agent_response)
        timings["memory_write_ms"] = _elapsed_ms(start)

        self.memory.update_summary(session_id, summary)
        timings["summary_ms"] = 0.0
        timings["total_ms"] = _elapsed_ms(total_start)

        return {
            "session_id": session_id,
            "trace_id": trace_id,
            "agent": "sales",
            "response": agent_response,
            "context": [],
            "rag_debug": {"docs": []},
            "tools": {},
            "executed_tools": [],
            "tool_errors": {},
            "fallback_reason": fallback_reason,
            "failure_stage": failure_stage,
            "resolved": False,
            "resolved_reason": "system_error",
            "plan": {},
            "timings": timings,
        }


def _elapsed_ms(start_perf: float) -> float:
    return round((time.perf_counter() - start_perf) * 1000, 2)


def _estimate_resolved(
    response: str,
    ask_for_more_info: bool,
    tool_errors: Dict[str, str],
    failure_stage: str,
) -> tuple[bool, str]:
    if failure_stage in {"plan", "rag", "tool", "policy"}:
        return False, failure_stage
    if ask_for_more_info:
        return False, "need_more_info"
    if any(err and err != "" for err in tool_errors.values()):
        return False, "tool_error"
    unresolved_tokens = ["需要您提供", "请提供", "无法判断", "稍后再试"]
    if any(tok in response for tok in unresolved_tokens):
        return False, "unresolved_text"
    return True, "ok"
