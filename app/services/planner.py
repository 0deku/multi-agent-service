import json
import re
from typing import List, Dict, Any

from app.services.llm_clients import LLMClient
from app.services.schemas import AgentPlan, ToolCall
from app.utils.logging_utils import get_logger

logger = get_logger("planner")


INVENTORY_KEYWORDS = [
    "库存", "尺码", "颜色", "有货", "现货", "补货", "断货", "下架", "预售", "在库", "缺货",
    "码", "s码", "m码", "l码", "xl", "xxl",
]

AFTER_SALES_KEYWORDS = [
    "退货", "换货", "换码", "退款", "物流", "订单", "发货", "运单", "售后", "签收",
    "拒收", "丢件", "少件", "漏发", "瑕疵", "破损", "发票", "补开", "催单", "拦截",
]
PROMO_KEYWORDS = [
    "优惠", "活动", "会员", "折扣", "满减", "优惠券", "积分", "促销",
    "限时", "福利", "发券", "首单", "折扣码", "团购", "黑五", "会员日", "生日月",
]


class Planner:
    def __init__(self):
        self.llm = LLMClient()

    def plan(self, message: str, memory_summary: str, recent_turns: List[Dict[str, Any]]) -> AgentPlan:
        system_prompt = (
            "你是电商智能客服的任务规划器。"
            "请产出一个JSON对象，字段包括: "
            "agent(sales/after_sales/promo/inventory), use_rag(true/false), "
            "tool_calls(数组, 每项含name和args), response_style(简洁/正式/活泼)。"
            "当用户提到订单/物流时优先调用order工具；提到库存/尺码/颜色/有货/补货/码数时调用inventory；"
            "提到优惠/会员/活动时调用promotion。"
            "只输出JSON，不要其他内容。"
        )
        user_prompt = (
            f"用户问题：{message}\n"
            f"历史总结：{memory_summary}\n"
            f"最近对话：{recent_turns}\n"
            "可用工具名称: inventory, order, promotion。"
        )

        try:
            raw = self.llm.chat(system_prompt, user_prompt)
            data = json.loads(_extract_json(raw))
            plan = AgentPlan(**data)
            return self._normalize_plan(message, plan)
        except Exception as exc:
            logger.warning("planner_fallback error=%s", exc)
            return self._heuristic_plan(message)

    def _normalize_plan(self, message: str, plan: AgentPlan) -> AgentPlan:
        """对 planner 输出做兜底，保证稳定。"""
        if not plan.tool_calls:
            heuristic = self._heuristic_plan(message)
            if heuristic.tool_calls:
                plan.tool_calls = heuristic.tool_calls

        m = message.lower()
        if plan.agent == "sales":
            if _is_after_sales_intent(m):
                plan.agent = "after_sales"
            elif _is_promo_intent(m):
                plan.agent = "promo"
            elif _is_inventory_intent(m):
                plan.agent = "inventory"

        if plan.agent == "inventory" and not any(c.name == "inventory" for c in plan.tool_calls):
            plan.tool_calls.append(ToolCall(name="inventory", args={"query": message}))

        return plan

    def _heuristic_plan(self, message: str) -> AgentPlan:
        m = message.lower()
        tool_calls = []
        agent = "sales"

        if _is_after_sales_intent(m):
            agent = "after_sales"
            tool_calls.append(ToolCall(name="order", args={"query": message}))

        if _is_inventory_intent(m):
            agent = "inventory"
            tool_calls.append(ToolCall(name="inventory", args={"query": message}))

        if _is_promo_intent(m):
            agent = "promo"
            tool_calls.append(ToolCall(name="promotion", args={"query": message}))

        return AgentPlan(
            agent=agent,
            use_rag=True,
            tool_calls=tool_calls,
            response_style="简洁",
        )


def _is_inventory_intent(m: str) -> bool:
    if any(k in m for k in INVENTORY_KEYWORDS):
        return True
    if re.search(r"\b([smlx]{1,2}|xxl)\b", m):
        return True
    if re.search(r"\b\d{2}(\.5)?\b", m):
        return True
    return False


def _is_after_sales_intent(m: str) -> bool:
    if any(k in m for k in AFTER_SALES_KEYWORDS):
        return True
    patterns = [
        r"退款.*(多久到账|何时)",
        r"(物流|快递).*(异常|卡住|没更新)",
        r"(订单).*(取消|撤销|改地址|改收货)",
    ]
    return any(re.search(p, m) for p in patterns)


def _is_promo_intent(m: str) -> bool:
    if any(k in m for k in PROMO_KEYWORDS):
        return True
    promo_patterns = [
        r"满\d+减\d+",
        r"满\d+打\d+折",
        r"第[一二三123]单",
        r"\d+折",
    ]
    return any(re.search(p, m) for p in promo_patterns)


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= 0 and end > start:
        return text[start : end + 1]
    return "{}"
