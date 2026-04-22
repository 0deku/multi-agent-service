from app.services.agents.base_agent import BaseAgent


class AfterSalesAgent(BaseAgent):
    name = "after_sales"
    allowed_tools = {"order"}

    @property
    def role_prompt(self) -> str:
        return "售后客服：处理退换货、物流、订单进度与异常，优先索取订单号等关键信息。"
