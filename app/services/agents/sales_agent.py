from app.services.agents.base_agent import BaseAgent


class SalesAgent(BaseAgent):
    name = "sales"
    allowed_tools = {"inventory", "promotion"}

    @property
    def role_prompt(self) -> str:
        return "导购推荐：根据用户需求推荐鞋服搭配与尺码建议，必要时结合库存与活动信息。"
