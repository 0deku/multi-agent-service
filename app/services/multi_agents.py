from app.services.llm_clients import LLMClient

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
        if any(k in msg for k in ["退货", "换货", "退款", "物流", "售后"]):
            return "after_sales", "您可以提供订单号，我帮您查询退换货或物流进度。"
        if any(k in msg for k in ["优惠", "活动", "打折", "会员"]):
            return "promo", "当前可用的促销包含会员折扣与季末清仓，我可以帮您筛选适合的活动。"
        if any(k in msg for k in ["库存", "尺码", "颜色", "有货"]):
            return "inventory", f"我帮您查看库存和尺码信息。相关资料：{context}"

        system_prompt = "你是电商客服路由器，只输出一个标签。"
        user_prompt = (
            "根据用户问题选择最合适的客服类型，仅输出标签本身：\n"
            "- sales\n- after_sales\n- promo\n- inventory\n\n"
            f"用户问题：{message}\n"
        )
        label = self.llm.chat(system_prompt, user_prompt).strip().lower()
        if label not in self.agents:
            label = "sales"
        return label, self.agents[label]
