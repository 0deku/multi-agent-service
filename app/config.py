import os

APP_ENV = os.getenv("APP_ENV", "dev")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

RAG_REBUILD = os.getenv("RAG_REBUILD", "0") == "1"

INVENTORY_DB = os.getenv("INVENTORY_DB", "data/inventory.json")
ORDER_DB = os.getenv("ORDER_DB", "data/orders.json")
PROMO_DB = os.getenv("PROMO_DB", "data/promotions.json")
