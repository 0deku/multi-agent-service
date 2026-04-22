import json
import os
import re
from typing import Dict, List

from chromadb import PersistentClient
from rank_bm25 import BM25Okapi

from app.config import (
    QWEN_EMBED_MODEL,
    QWEN_RERANK_MODEL,
    RAG_BM25_CANDIDATE_K,
    RAG_CANDIDATE_K,
    RAG_TOP_K,
)
from app.services.llm_clients import LLMClient


class RAGService:
    """Hybrid RAG only: Vector(Qwen Embedding) + BM25 + RRF + Qwen rerank."""

    def __init__(self):
        self.llm = LLMClient()
        self.seed_docs = self._load_seed_docs()

        self.client = PersistentClient(path="data/chroma")
        self.collection = self.client.get_or_create_collection("nike")

        self._doc_by_id: Dict[str, str] = {}
        self._doc_order: List[str] = []
        self._init_doc_cache()

        self.bm25 = BM25Okapi([self._tokenize(self._doc_by_id[d]) for d in self._doc_order])
        self._bm25_doc_ids = list(self._doc_order)

        self._rewrite_cache: Dict[str, str] = {}
        self._embed_cache: Dict[str, List[float]] = {}

        if os.getenv("RAG_REBUILD", "0") == "1":
            try:
                self.client.delete_collection("nike")
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection("nike")

        if self.collection.count() == 0 and self.seed_docs:
            self._build_collection_from_seed()

    def _load_seed_docs(self) -> List[dict]:
        data_path = os.path.join("data", "nike_seed.json")
        if not os.path.exists(data_path):
            return []
        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _init_doc_cache(self):
        for idx, d in enumerate(self.seed_docs):
            doc_id = str(d.get("id") or f"seed-{idx}")
            self._doc_order.append(doc_id)
            self._doc_by_id[doc_id] = self._compose_text(d)

    def _build_collection_from_seed(self):
        ids = [str(d.get("id") or f"seed-{idx}") for idx, d in enumerate(self.seed_docs)]
        texts = [self._doc_by_id[_id] for _id in ids]
        metadatas = [
            {
                "category": d.get("category", ""),
                "topic": d.get("topic", ""),
                "title": d.get("title", ""),
                "tags": ",".join(d.get("tags", [])),
            }
            for d in self.seed_docs
        ]
        embeddings = self._embed_texts(texts)
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

    def query(self, query: str, top_k: int | None = None, candidate_k: int | None = None) -> List[str]:
        return self.query_with_debug(query, top_k=top_k, candidate_k=candidate_k)["docs"]

    def query_with_debug(self, query: str, top_k: int | None = None, candidate_k: int | None = None) -> Dict[str, List[str] | str | bool]:
        top_k = top_k or RAG_TOP_K
        candidate_k = candidate_k or RAG_CANDIDATE_K
        bm25_k = max(RAG_BM25_CANDIDATE_K, candidate_k)

        fallback_reason = ""

        rewritten = self._rewrite_query(query)
        vector_doc_ids = self._vector_recall_ids(rewritten, candidate_k)
        bm25_doc_ids = self._bm25_recall_ids(rewritten, bm25_k)

        merged_doc_ids = self._hybrid_merge_rrf(vector_doc_ids, bm25_doc_ids)
        if not merged_doc_ids and self._doc_order:
            merged_doc_ids = self._doc_order[:top_k]
            fallback_reason = "empty_rrf_fallback"

        candidates = [self._doc_by_id[d] for d in merged_doc_ids if d in self._doc_by_id]
        docs, rerank_fallback = self._rerank_with_qwen_rank(rewritten, candidates, top_k)
        if rerank_fallback and not fallback_reason:
            fallback_reason = rerank_fallback

        return {
            "docs": docs,
            "vector_docs": [self._doc_by_id[d] for d in vector_doc_ids if d in self._doc_by_id],
            "bm25_docs": [self._doc_by_id[d] for d in bm25_doc_ids if d in self._doc_by_id],
            "merged_docs": candidates,
            "fallback": fallback_reason,
            "rewrite_cache_hit": query in self._rewrite_cache,
        }

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not self.llm.enabled or not self.llm.qwen_client:
            raise RuntimeError("QWEN_API_KEY is required for qwen embedding")

        uncached = [t for t in texts if t not in self._embed_cache]
        if uncached:
            resp = self.llm.qwen_client.embeddings.create(
                model=QWEN_EMBED_MODEL,
                input=uncached,
                timeout=self.llm.request_timeout,
            )
            for text, item in zip(uncached, resp.data):
                self._embed_cache[text] = item.embedding

        return [self._embed_cache[t] for t in texts]

    def _rewrite_query(self, query: str) -> str:
        if query in self._rewrite_cache:
            return self._rewrite_cache[query]

        prompt = (
            "你是检索改写助手，请将用户问题改写为更适合检索的短句。"
            "只输出改写后的句子，不要解释。"
        )
        try:
            rewritten = self.llm.chat(prompt, query).strip()
            out = rewritten if rewritten else query
        except Exception:
            out = query

        self._rewrite_cache[query] = out
        return out

    def _vector_recall_ids(self, query: str, candidate_k: int) -> List[str]:
        try:
            emb = self._embed_texts([query])
            results = self.collection.query(query_embeddings=emb, n_results=candidate_k)
            ids = results.get("ids") or []
            if ids and ids[0]:
                return [str(x) for x in ids[0]]
            return []
        except Exception:
            return []

    def _bm25_recall_ids(self, query: str, candidate_k: int) -> List[str]:
        tokens = self._tokenize(query)
        if not tokens or not self._bm25_doc_ids:
            return self._bm25_doc_ids[:candidate_k]

        scores = self.bm25.get_scores(tokens)
        ranked_pairs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        selected = [self._bm25_doc_ids[i] for i, s in ranked_pairs if s > 0][:candidate_k]
        return selected if selected else self._bm25_doc_ids[:candidate_k]

    def _hybrid_merge_rrf(self, vector_ids: List[str], bm25_ids: List[str], k: int = 60) -> List[str]:
        scores: Dict[str, float] = {}

        for rank, doc_id in enumerate(vector_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

        for rank, doc_id in enumerate(bm25_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in ranked]

    def _rerank_with_qwen_rank(self, query: str, candidates: List[str], top_k: int) -> tuple[List[str], str]:
        if not candidates:
            return [], "empty_candidates"
        if len(candidates) <= top_k:
            return candidates, ""
        if not self.llm.enabled or not self.llm.qwen_client:
            return candidates[:top_k], "rerank_client_unavailable"

        numbered_docs = "\n\n".join([f"[{i}] {doc}" for i, doc in enumerate(candidates)])
        system_prompt = (
            "你是检索重排助手。给定query和候选文档，请按相关性从高到低排序。"
            "仅输出JSON数组，内容是文档下标，例如 [2,0,1]。不要输出其他文字。"
        )
        user_prompt = f"query: {query}\n\n候选文档:\n{numbered_docs}"

        try:
            resp = self.llm.qwen_client.chat.completions.create(
                model=QWEN_RERANK_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                timeout=self.llm.request_timeout,
            )
            content = (resp.choices[0].message.content or "").strip()
            start = content.find("[")
            end = content.rfind("]")
            if start < 0 or end <= start:
                return candidates[:top_k], "rerank_parse_failed"
            indices = json.loads(content[start : end + 1])
            ranked = []
            used = set()
            for idx in indices:
                if isinstance(idx, int) and 0 <= idx < len(candidates) and idx not in used:
                    ranked.append(candidates[idx])
                    used.add(idx)
            for idx, doc in enumerate(candidates):
                if idx not in used:
                    ranked.append(doc)
            return ranked[:top_k], ""
        except Exception:
            return candidates[:top_k], "rerank_exception"

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        ascii_tokens = re.findall(r"[a-z0-9]+", text)
        zh_tokens = re.findall(r"[\u4e00-\u9fff]", text)
        return ascii_tokens + zh_tokens
