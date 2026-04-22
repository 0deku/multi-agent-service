from app.services.agents.base_agent import BaseAgent


class PromoAgent(BaseAgent):
    name = "promo"
    allowed_tools = {"promotion"}

    @property
    def role_prompt(self) -> str:
        return "促销客服：介绍活动规则、会员权益、优惠券门槛与使用条件。"
