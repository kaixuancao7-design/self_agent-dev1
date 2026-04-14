"""
混合检索器：多阶段过滤架构
实现 Query Processing、Hybrid Search Execution、Filtering & Reranking
"""

import re
import math
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from rag.bm25_index import bm25_index
from utils.logger_handler import logger


class HybridRetriever:
    def __init__(self, vector_service: "VectorStoreService", k1: float = 1.5, b: float = 0.75):
        self.vector_service = vector_service
        self.k1 = k1
        self.b = b
        # 简单同义词字典，可扩展
        self.synonyms = {
            "机器学习": ["ML", "machine learning"],
            "人工智能": ["AI", "artificial intelligence"],
            "深度学习": ["DL", "deep learning"],
            # 添加更多...
        }

    def _extract_keywords(self, query: str) -> List[str]:
        """关键词提取：去除停用词，提取实体和动词"""
        # 简单实现：分词，去除常见停用词
        stop_words = {"的", "了", "和", "是", "在", "有", "为", "对", "与", "从", "到", "这", "那", "我", "你", "他", "她", "它", "我们", "你们", "他们"}
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 1]
        return keywords

    def _expand_query(self, query: str) -> Tuple[str, List[str]]:
        """查询扩展：同义词扩展"""
        keywords = self._extract_keywords(query)
        expanded_keywords = set(keywords)
        for kw in keywords:
            if kw in self.synonyms:
                expanded_keywords.update(self.synonyms[kw])
        # Sparse: OR 连接所有关键词
        sparse_query = " ".join(expanded_keywords)
        # Dense: 使用原始query
        dense_query = query
        return sparse_query, dense_query

    def _rrf_fusion(self, dense_results: List[Tuple[str, float]], sparse_results: List[Tuple[str, float]], k: int = 60) -> List[Tuple[str, float]]:
        """RRF融合算法"""
        scores = defaultdict(float)
        # Dense结果
        for rank, (doc_id, _) in enumerate(dense_results, start=1):
            scores[doc_id] += 1 / (k + rank)
        # Sparse结果
        for rank, (doc_id, _) in enumerate(sparse_results, start=1):
            scores[doc_id] += 1 / (k + rank)
        # 排序
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores

    def _apply_filters(self, candidates: List[Tuple[str, float]], filters: Dict[str, Any]) -> List[Tuple[str, float]]:
        """应用元数据过滤"""
        # 简单实现：根据doc_type过滤（可扩展）
        if not filters:
            return candidates
        filtered = []
        for doc_id, score in candidates:
            # 这里需要根据实际元数据过滤，暂时跳过
            # 例如：if doc_id.startswith(filters.get('doc_type', '')):
            filtered.append((doc_id, score))
        return filtered

    def _rerank(self, query: str, candidates: List[Tuple[str, float]], rerank_method: str = "none") -> List[Tuple[str, float]]:
        """重排：可选Cross-Encoder或LLM"""
        if rerank_method == "none":
            return candidates
        # 实现Cross-Encoder或LLM重排
        # 暂时返回原顺序
        logger.warning("Rerank not implemented, using original order")
        return candidates

    def retrieve(self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None, rerank_method: str = "none") -> List[Tuple[str, Any]]:
        """多阶段过滤检索"""
        # 1. Query Processing
        sparse_query, dense_query = self._expand_query(query)
        logger.info(f"Sparse query: {sparse_query}, Dense query: {dense_query}")

        # 2. Hybrid Search Execution
        # Dense Route
        dense_retriever = self.vector_service.get_retriever()
        dense_docs = dense_retriever.invoke(dense_query)
        dense_results = [(doc.metadata.get('source', str(i)), 1.0) for i, doc in enumerate(dense_docs)]  # 简化score

        # Sparse Route
        sparse_results = bm25_index.search(sparse_query, top_k=top_k)

        # Fusion
        fused_results = self._rrf_fusion(dense_results, sparse_results)

        # 3. Filtering & Reranking
        # Filtering
        filtered_results = self._apply_filters(fused_results, filters or {})

        # Reranking
        reranked_results = self._rerank(query, filtered_results, rerank_method)

        # 返回Top-K
        final_results = reranked_results[:top_k]

        # 获取实际文档内容（简化）
        results = []
        for doc_id, score in final_results:
            # 这里需要根据doc_id获取文档内容
            # 暂时返回doc_id和score
            results.append((doc_id, {"score": score, "content": f"Document {doc_id}"}))

        return results


# 全局混合检索器实例（可选）
hybrid_retriever = None

def get_hybrid_retriever(vector_service: "VectorStoreService"):
    global hybrid_retriever
    if hybrid_retriever is None:
        hybrid_retriever = HybridRetriever(vector_service)
    return hybrid_retriever
