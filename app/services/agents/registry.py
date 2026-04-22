from app.services.agents.after_sales_agent import AfterSalesAgent
from app.services.agents.base_agent import BaseAgent
from app.services.agents.inventory_agent import InventoryAgent
from app.services.agents.promo_agent import PromoAgent
from app.services.agents.sales_agent import SalesAgent
from app.services.llm_clients import LLMClient
from app.services.schemas import AgentName
from app.services.tool_executor import ToolExecutor


def build_agent_registry(llm: LLMClient, tool_executor: ToolExecutor) -> dict[AgentName, BaseAgent]:
    return {
        "sales": SalesAgent(llm=llm, tool_executor=tool_executor),
        "after_sales": AfterSalesAgent(llm=llm, tool_executor=tool_executor),
        "promo": PromoAgent(llm=llm, tool_executor=tool_executor),
        "inventory": InventoryAgent(llm=llm, tool_executor=tool_executor),
    }
