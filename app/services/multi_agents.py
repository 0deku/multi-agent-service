import re
from app.services.llm_clients import LLMClient


INVENTORY_KEYWORDS = [
    "库存", "尺码", "颜色", "有货", "现货", "补货", "断货", "下架", "预售", "在库", "缺货",
    "码", "s码", "m码", "l码", "xl", "xxl",
]

AFTER_SALES_KEYWORDS = [
    "退货", "换货", "换码", "退款", "物流", "售后", "订单", "发货", "运单", "签收",
    "拒收", "丢件", "少件", "漏发", "瑕疵", "破损", "发票", "补开", "催单", "拦截",
]


class MultiAgentRouter:
    def __init__(self):
        self.agents = {
            "sales": "导购推荐：根据用户需求推荐鞋服搭配与尺码建议",
            "after_sales": "售后客服：退换货、物流、订单问题处理",
            "promo": "促销客服：活动、优惠、会员权益介绍",
            "inventory": "库存客服：尺码、颜色、库存状态咨询",
        }
        self.llm = LLMClient()

    def route(self, message: str, context, memory):
        msg = message.lower()
        if _is_after_sales_intent(msg):
            return "after_sales", "您可以提供订单号，我帮您查询退换货或物流进度。"
        if _is_promo_intent(msg):
            return "promo", "当前可用的促销包含会员折扣与季末清仓，我可以帮您筛选适合的活动。"
        if _is_inventory_intent(msg):
            return "inventory", f"我帮您查看库存和尺码信息。相关资料：{context}"

        system_prompt = "你是电商客服路由器，只输出一个标签。"
        user_prompt = (
            "根据用户问题选择最合适的客服类型，仅输出标签本身：\n"
            "- sales\n- after_sales\n- promo\n- inventory\n\n"
            f"用户问题：{message}\n"
        )
        try:
            label = self.llm.chat(system_prompt, user_prompt).strip().lower()
        except Exception:
            label = "sales"
        if label not in self.agents:
            label = "sales"
        return label, self.agents[label]


def _is_inventory_intent(msg: str) -> bool:
    if any(k in msg for k in INVENTORY_KEYWORDS):
        return True
    if re.search(r"\b([smlx]{1,2}|xxl)\b", msg):
        return True
    if re.search(r"\b\d{2}(\.5)?\b", msg):
        return True
    return False


def _is_after_sales_intent(msg: str) -> bool:
    if any(k in msg for k in AFTER_SALES_KEYWORDS):
        return True
    patterns = [
        r"退款.*(多久到账|何时)",
        r"(物流|快递).*(异常|卡住|没更新)",
        r"(订单).*(取消|撤销|改地址|改收货)",
    ]
    return any(re.search(p, msg) for p in patterns)


def _is_promo_intent(msg: str) -> bool:
    promo_keywords = [
        "优惠", "活动", "打折", "会员", "满减", "优惠券", "积分", "促销",
        "限时", "福利", "发券", "首单", "折扣码", "团购", "黑五", "会员日", "生日月",
    ]
    if any(k in msg for k in promo_keywords):
        return True
    promo_patterns = [r"满\d+减\d+", r"满\d+打\d+折", r"第[一二三123]单", r"\d+折"]
    return any(re.search(p, msg) for p in promo_patterns)
