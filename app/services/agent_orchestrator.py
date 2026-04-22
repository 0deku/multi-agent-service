import logging

from app.services.rag import RAGService
from app.services.memory import MemoryStore
from app.services.multi_agents import MultiAgentRouter
from app.services.llm_clients import LLMClient
from app.services.tools import InventoryTool, OrderTool, PromoTool
from app.utils.logging_utils import timed

logger = logging.getLogger("orchestrator")


class AgentOrchestrator:
    def __init__(self):
        self.rag = RAGService()
        self.memory = MemoryStore()
        self.router = MultiAgentRouter()
        self.llm = LLMClient()
        self.inventory_tool = InventoryTool()
        self.order_tool = OrderTool()
        self.promo_tool = PromoTool()

    @timed(logger, "handle_message")
    def handle_message(self, session_id: str, message: str):
        memory = self.memory.get_memory(session_id)
        context = self.rag.query(message)

        agent_name, agent_prompt = self.router.route(message, context, memory)
        logger.info("route=%s session=%s", agent_name, session_id)

        recent = self.memory.recent_turns(session_id, limit=6)
        summary = memory.get("summary", "")

        tool_context = self._maybe_call_tools(message)
        if tool_context:
            logger.info("tools_used=%s session=%s", list(tool_context.keys()), session_id)

        system_prompt = (
            "你是耐克卖衣服的智能客服，注意礼貌、简洁、给出明确建议。"
            "你要基于给定的知识内容回答，遇到不确定的问题要说明需要进一步信息。"
            f"当前角色：{agent_name}。"
        )
        user_prompt = (
            f"用户问题：{message}\n\n"
            f"可用知识：{context}\n\n"
            f"业务工具结果：{tool_context}\n\n"
            f"历史总结：{summary}\n\n"
            f"最近对话：{recent}"
        )

        try:
            agent_response = self.llm.chat(system_prompt, user_prompt)
        except Exception as exc:
            logger.exception("llm_error session=%s", session_id)
            agent_response = "抱歉，当前系统繁忙，请稍后再试。"

        self.memory.update_memory(session_id, message, agent_response)

        summary_prompt = (
            "你是对话总结助手，请用中文简洁总结用户偏好、尺码、预算和关键需求。"
            "只输出总结内容，不要加多余解释。"
        )
        summary_input = f"已有总结：{summary}\n\n新增对话：用户={message}；客服={agent_response}"
        try:
            new_summary = self.llm.chat(summary_prompt, summary_input)
        except Exception:
            new_summary = summary
        self.memory.update_summary(session_id, new_summary)

        return {
            "session_id": session_id,
            "agent": agent_name,
            "response": agent_response,
            "context": context,
            "tools": tool_context,
        }

    def _maybe_call_tools(self, message: str):
        msg = message.lower()
        results = {}
        if any(k in msg for k in ["库存", "有货", "尺码", "颜色"]):
            results["inventory"] = self.inventory_tool.lookup(message)
        if any(k in msg for k in ["订单", "物流", "发货", "运单"]):
            results["order"] = self.order_tool.lookup(message)
        if any(k in msg for k in ["优惠", "活动", "折扣", "会员"]):
            results["promotion"] = self.promo_tool.lookup(message)
        return results
