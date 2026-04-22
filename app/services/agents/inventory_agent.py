from app.services.agents.base_agent import BaseAgent


class InventoryAgent(BaseAgent):
    name = "inventory"
    allowed_tools = {"inventory"}

    @property
    def role_prompt(self) -> str:
        return "库存客服：专注尺码、颜色、在库/缺货状态与补货信息查询。"
