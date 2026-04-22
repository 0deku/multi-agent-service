import json
import os
import re

from app.config import INVENTORY_DB, ORDER_DB, PROMO_DB


class InventoryTool:
    def __init__(self):
        self.data = _load_json(INVENTORY_DB)

    def lookup(self, message: str, args: dict | None = None):
        query = message.lower()
        args = args or {}
        focus = " ".join([str(args.get("name", "")), str(args.get("color", "")), str(args.get("size", ""))]).strip().lower()
        if focus:
            query = f"{query} {focus}"
        hits = []
        for item in self.data:
            text = f"{item.get('name','')} {item.get('color','')} {item.get('size','')}".lower()
            if any(k in query for k in [item.get("name", "").lower(), item.get("color", "").lower(), item.get("size", "").lower()]):
                hits.append(item)
            elif any(k in text for k in _keywords(query)):
                hits.append(item)
        return hits[:5]


class OrderTool:
    def __init__(self):
        self.data = _load_json(ORDER_DB)

    def lookup(self, message: str, args: dict | None = None):
        args = args or {}
        order_id = args.get("order_id") or extract_order_id(message)
        if not order_id:
            return {"hint": "请提供订单号以便查询", "example": "NKE2024-0001"}
        for order in self.data:
            if order.get("order_id") == order_id:
                return order
        return {"order_id": order_id, "status": "未找到订单"}


class PromoTool:
    def __init__(self):
        self.data = _load_json(PROMO_DB)

    def lookup(self, message: str, args: dict | None = None):
        args = args or {}
        query = message.lower()
        keyword = args.get("keyword", "")
        if keyword:
            query = f"{query} {keyword}".lower()
        hits = []
        for promo in self.data:
            text = f"{promo.get('title','')} {promo.get('description','')}".lower()
            if any(k in text for k in _keywords(query)):
                hits.append(promo)
        return hits[:3]


def _load_json(path: str):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_order_id(message: str):
    match = re.search(r"(NKE\d{4}-\d{4})", message)
    return match.group(1) if match else ""


def _keywords(text: str):
    return [w for w in re.split(r"\W+", text) if w]
