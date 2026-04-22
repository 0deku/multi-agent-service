from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


ToolName = Literal["inventory", "order", "promotion"]
AgentName = Literal["sales", "after_sales", "promo", "inventory"]


class ToolCall(BaseModel):
    name: ToolName
    args: Dict[str, Any] = Field(default_factory=dict)


class AgentPlan(BaseModel):
    agent: AgentName
    use_rag: bool = True
    tool_calls: List[ToolCall] = Field(default_factory=list)
    response_style: Literal["简洁", "正式", "活泼"] = "简洁"


class ToolExecutionResult(BaseModel):
    name: ToolName
    ok: bool
    output: Any = None
    error: Optional[str] = None
