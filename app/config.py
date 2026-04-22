import os

APP_ENV = os.getenv("APP_ENV", "dev")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_EMBED_MODEL = os.getenv("QWEN_EMBED_MODEL", "text-embedding-v3")
QWEN_RERANK_MODEL = os.getenv("QWEN_RERANK_MODEL", "qwen-rank")

RAG_REBUILD = os.getenv("RAG_REBUILD", "0") == "1"
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
RAG_CANDIDATE_K = int(os.getenv("RAG_CANDIDATE_K", "8"))
RAG_BM25_CANDIDATE_K = int(os.getenv("RAG_BM25_CANDIDATE_K", "8"))

INVENTORY_DB = os.getenv("INVENTORY_DB", "data/inventory.json")
ORDER_DB = os.getenv("ORDER_DB", "data/orders.json")
PROMO_DB = os.getenv("PROMO_DB", "data/promotions.json")
