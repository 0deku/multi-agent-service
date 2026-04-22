import os
import json
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from app.services.llm_clients import LLMClient

class RAGService:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.client = PersistentClient(path="data/chroma")
        self.llm = LLMClient()
        self.seed_docs = []

        if os.getenv("RAG_REBUILD", "0") == "1":
            try:
                self.client.delete_collection("nike")
            except Exception:
                pass

        self.collection = self.client.get_or_create_collection("nike")

        if self.collection.count() == 0:
            self._load_seed_data()

    def _load_seed_data(self):
        data_path = os.path.join("data", "nike_seed.json")
        if not os.path.exists(data_path):
            return
        with open(data_path, "r", encoding="utf-8") as f:
            docs = json.load(f)
        self.seed_docs = docs

        ids = [d["id"] for d in docs]
        texts = [self._compose_text(d) for d in docs]
        metadatas = [
            {
                "category": d.get("category", ""),
                "topic": d.get("topic", ""),
                "title": d.get("title", ""),
                "tags": ",".join(d.get("tags", [])),
            }
            for d in docs
        ]
        embeddings = self.embedder.encode(texts).tolist()

        self.collection.add(documents=texts, ids=ids, embeddings=embeddings, metadatas=metadatas)

    def _compose_text(self, doc: dict) -> str:
        parts = [
            f"标题：{doc.get('title', '')}",
            f"类别：{doc.get('category', '')}",
            f"主题：{doc.get('topic', '')}",
            f"内容：{doc.get('text', '')}",
            f"标签：{', '.join(doc.get('tags', []))}",
        ]
        return "\n".join([p for p in parts if p and not p.endswith(": ")])

    def query(self, query: str, top_k: int = 3, candidate_k: int = 8):
        rewritten = self._rewrite_query(query)
        candidates = self._vector_recall(rewritten, candidate_k)
        candidates = self._keyword_recall(rewritten, candidates)

        if not candidates:
            return []

        pairs = [[rewritten, d] for d in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked[:top_k]]

    def _rewrite_query(self, query: str) -> str:
        prompt = (
            "你是检索改写助手，请将用户问题改写为更适合检索的短句。"
            "只输出改写后的句子，不要解释。"
        )
        try:
            rewritten = self.llm.chat(prompt, query).strip()
            return rewritten if rewritten else query
        except Exception:
            return query

    def _vector_recall(self, query: str, candidate_k: int):
        emb = self.embedder.encode([query]).tolist()
        results = self.collection.query(query_embeddings=emb, n_results=candidate_k)
        return results["documents"][0] if results.get("documents") else []

    def _keyword_recall(self, query: str, existing_docs: list):
        if not self.seed_docs:
            return existing_docs
        query_lower = query.lower()
        scored = []
        for doc in self.seed_docs:
            text = self._compose_text(doc)
            hit = sum(1 for t in doc.get("tags", []) if t.lower() in query_lower)
            if hit > 0 or doc.get("title", "").lower() in query_lower:
                scored.append(text)
        merged = list(dict.fromkeys(existing_docs + scored))
        return merged
