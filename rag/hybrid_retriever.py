"""
混合检索器：多阶段过滤架构
实现 Query Processing、Hybrid Search Execution、Filtering & Reranking
"""

import re
import math
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from rag.bm25_index import bm25_index
from rag.reranker import RerankerFactory, BaseReranker
from utils.logger_handler import logger


class HybridRetriever:
    def __init__(self, vector_service: "VectorStoreService", k1: float = 1.5, b: float = 0.75, 
                 rerank_method: str = "linear", **rerank_kwargs):
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
        # 初始化重排器
        self.reranker: BaseReranker = RerankerFactory.create(rerank_method, **rerank_kwargs)

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

    def _rerank(self, query: str, candidates: List[Tuple[str, float]], rerank_method: str = None) -> List[Tuple[str, float]]:
        """重排：使用配置的重排器进行精排"""
        # 如果没有指定方法，使用初始化时配置的重排器
        if rerank_method is None or rerank_method == "default":
            # 将格式转换为重排器期望的格式
            rerank_candidates = [(doc_id, {'score': score, 'content': f"Document {doc_id}"}) 
                                for doc_id, score in candidates]
            reranked = self.reranker.rerank(query, rerank_candidates, top_k=len(candidates))
            # 转换回原格式
            return [(doc_id, result.get('rerank_score', result.get('score', score))) 
                    for (doc_id, result), (_, score) in zip(reranked, candidates)]
        
        # 如果指定了其他方法，动态创建重排器
        temp_reranker = RerankerFactory.create(rerank_method)
        rerank_candidates = [(doc_id, {'score': score, 'content': f"Document {doc_id}"}) 
                            for doc_id, score in candidates]
        reranked = temp_reranker.rerank(query, rerank_candidates, top_k=len(candidates))
        return [(doc_id, result.get('rerank_score', result.get('score', score))) 
                for (doc_id, result), (_, score) in zip(reranked, candidates)]

    def retrieve(self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None, rerank_method: str = "default") -> List[Tuple[str, Any]]:
        """多阶段过滤检索"""
        # 1. Query Processing
        sparse_query, dense_query = self._expand_query(query)
        logger.info(f"Sparse query: {sparse_query}, Dense query: {dense_query}")

        # 2. Hybrid Search Execution
        # Dense Route
        dense_retriever = self.vector_service.get_retriever()
        dense_docs = dense_retriever.invoke(dense_query)
        dense_results = [(doc.metadata.get('source', str(i)), doc.metadata.get('score', 1.0)) 
                         for i, doc in enumerate(dense_docs)]

        # Sparse Route
        sparse_results = bm25_index.search(sparse_query, top_k=top_k)

        # Fusion
        fused_results = self._rrf_fusion(dense_results, sparse_results)

        # 3. Filtering & Reranking
        # Filtering
        filtered_results = self._apply_filters(fused_results, filters or {})

        # Reranking
        reranked_results = self._rerank(query, filtered_results, rerank_method)
        logger.info(f"[HybridRetriever] 重排完成，使用方法: {rerank_method}")

        # 返回Top-K
        final_results = reranked_results[:top_k]

        # 获取实际文档内容
        results = []
        for doc_id, score in final_results:
            # 尝试从向量服务获取文档内容
            doc_content = f"Document {doc_id}"
            try:
                # 这里可以调用向量服务获取实际文档内容
                docs = self.vector_service.retrieve_docs(query, top_k=1, filters={"source": doc_id})
                if docs:
                    doc_content = docs[0].get('content', doc_content)
            except Exception:
                pass
            results.append((doc_id, {"score": score, "content": doc_content}))

        return results


# 全局混合检索器实例（可选）
hybrid_retriever = None

def get_hybrid_retriever(vector_service: "VectorStoreService"):
    global hybrid_retriever
    if hybrid_retriever is None:
        hybrid_retriever = HybridRetriever(vector_service)
    return hybrid_retriever
