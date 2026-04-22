from typing import Any, Dict, Iterable, List

from app.services.schemas import ToolCall, ToolExecutionResult
from app.services.tools import InventoryTool, OrderTool, PromoTool


class ToolExecutor:
    def __init__(self):
        self.inventory = InventoryTool()
        self.order = OrderTool()
        self.promo = PromoTool()

    def validate_call(self, call: ToolCall) -> str | None:
        args = call.args or {}
        query = str(args.get("query") or "").strip()

        if call.name in {"inventory", "promotion"}:
            return None if query else "missing_required_args"

        if call.name == "order":
            has_query = bool(query)
            has_order_id = bool(str(args.get("order_id") or "").strip())
            return None if (has_query or has_order_id) else "missing_required_args"

        return "unsupported_tool"

    def run(self, message: str, call: ToolCall) -> ToolExecutionResult:
        validation_error = self.validate_call(call)
        if validation_error:
            return ToolExecutionResult(name=call.name, ok=False, error=validation_error)

        try:
            if call.name == "inventory":
                query = str(call.args.get("query") or message)
                return ToolExecutionResult(name="inventory", ok=True, output=self.inventory.lookup(query))

            if call.name == "order":
                query = str(call.args.get("query") or message)
                return ToolExecutionResult(name="order", ok=True, output=self.order.lookup(query))

            if call.name == "promotion":
                query = str(call.args.get("query") or message)
                return ToolExecutionResult(name="promotion", ok=True, output=self.promo.lookup(query))

            return ToolExecutionResult(name=call.name, ok=False, error="unsupported_tool")
        except Exception as exc:
            return ToolExecutionResult(name=call.name, ok=False, error=str(exc))

    def run_many(self, message: str, calls: list[ToolCall]) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for call in calls:
            result = self.run(message, call)
            results[call.name] = {
                "ok": result.ok,
                "output": result.output,
                "error": result.error,
            }
        return results

    def run_many_with_whitelist(
        self,
        message: str,
        calls: List[ToolCall],
        allowed_tools: Iterable[str],
    ) -> Dict[str, Any]:
        allowed = set(allowed_tools)
        filtered_calls = [call for call in calls if call.name in allowed]
        return self.run_many(message, filtered_calls)
