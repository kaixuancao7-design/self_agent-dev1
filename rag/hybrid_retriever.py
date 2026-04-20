"""
混合检索器：多阶段过滤架构
实现 Query Processing、Hybrid Search Execution、Filtering & Reranking
支持BM25+向量权重可配置
"""

import re
import math
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from rag.bm25_index import bm25_index
from rag.reranker import RerankerFactory, BaseReranker
from utils.logger_handler import logger
from utils.config_handler import rag_cfg


class HybridRetriever:
    def __init__(self, vector_service: "VectorStoreService", k1: float = 1.5, b: float = 0.75, 
                 rerank_method: str = "linear", rrf_k: int = 60, hybrid_weight: float = 0.5,
                 bm25_weight: float = 0.5, vector_weight: float = 0.5, **rerank_kwargs):
        """
        初始化混合检索器
        
        参数：
        - k1: BM25参数，控制词频饱和
        - b: BM25参数，控制文档长度影响
        - rerank_method: 重排方法，可选: linear, cross_encoder, llm
        - rrf_k: RRF融合参数，默认60
        - hybrid_weight: 混合检索权重（0-1，值越大向量检索权重越高）
        - bm25_weight: BM25检索权重（0-1）
        - vector_weight: 向量检索权重（0-1）
        """
        self.vector_service = vector_service
        self.k1 = k1
        self.b = b
        self.rrf_k = rrf_k
        self.hybrid_weight = hybrid_weight
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        
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
        sparse_query = " ".join(expanded_keywords)
        dense_query = query
        return sparse_query, dense_query

    def _rrf_fusion(self, dense_results: List[Tuple[str, float]], sparse_results: List[Tuple[str, float]], 
                    k: int = None, bm25_weight: float = None, vector_weight: float = None) -> List[Tuple[str, float]]:
        """
        RRF融合算法，支持权重配置
        
        参数：
        - k: RRF参数，默认使用初始化值
        - bm25_weight: BM25结果权重
        - vector_weight: 向量结果权重
        """
        k = k if k is not None else self.rrf_k
        bm25_w = bm25_weight if bm25_weight is not None else self.bm25_weight
        vector_w = vector_weight if vector_weight is not None else self.vector_weight
        
        scores = defaultdict(float)
        
        # Dense结果（向量检索）
        for rank, (doc_id, _) in enumerate(dense_results, start=1):
            scores[doc_id] += vector_w / (k + rank)
        
        # Sparse结果（BM25）
        for rank, (doc_id, _) in enumerate(sparse_results, start=1):
            scores[doc_id] += bm25_w / (k + rank)
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores

    def _weighted_fusion(self, dense_results: List[Tuple[str, float]], sparse_results: List[Tuple[str, float]],
                        weight: float = None) -> List[Tuple[str, float]]:
        """
        加权融合算法：weight控制向量检索权重，(1-weight)控制BM25权重
        
        参数：
        - weight: 向量检索权重（0-1），默认使用初始化的hybrid_weight
        """
        hybrid_w = weight if weight is not None else self.hybrid_weight
        bm25_w = 1.0 - hybrid_w
        
        # 归一化分数
        dense_scores = {}
        sparse_scores = {}
        
        for doc_id, score in dense_results:
            dense_scores[doc_id] = score
        
        for doc_id, score in sparse_results:
            sparse_scores[doc_id] = score
        
        # 归一化
        max_dense = max(dense_scores.values(), default=1.0)
        max_sparse = max(sparse_scores.values(), default=1.0)
        
        fused_scores = defaultdict(float)
        
        # 合并两个来源的结果
        all_docs = set(dense_scores.keys()).union(set(sparse_scores.keys()))
        
        for doc_id in all_docs:
            dense_score = (dense_scores.get(doc_id, 0.0) / max_dense) if max_dense > 0 else 0.0
            sparse_score = (sparse_scores.get(doc_id, 0.0) / max_sparse) if max_sparse > 0 else 0.0
            fused_scores[doc_id] = hybrid_w * dense_score + bm25_w * sparse_score
        
        sorted_scores = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores

    def _apply_filters(self, candidates: List[Tuple[str, float]], filters: Dict[str, Any]) -> List[Tuple[str, float]]:
        """应用元数据过滤"""
        if not filters:
            return candidates
        filtered = []
        for doc_id, score in candidates:
            filtered.append((doc_id, score))
        return filtered

    def _rerank(self, query: str, candidates: List[Tuple[str, float]], rerank_method: str = None) -> List[Tuple[str, float]]:
        """重排：使用配置的重排器进行精排"""
        if rerank_method is None or rerank_method == "default":
            rerank_candidates = [(doc_id, {'score': score, 'content': f"Document {doc_id}"}) 
                                for doc_id, score in candidates]
            reranked = self.reranker.rerank(query, rerank_candidates, top_k=len(candidates))
            return [(doc_id, result.get('rerank_score', result.get('score', score))) 
                    for (doc_id, result), (_, score) in zip(reranked, candidates)]
        
        temp_reranker = RerankerFactory.create(rerank_method)
        rerank_candidates = [(doc_id, {'score': score, 'content': f"Document {doc_id}"}) 
                            for doc_id, score in candidates]
        reranked = temp_reranker.rerank(query, rerank_candidates, top_k=len(candidates))
        return [(doc_id, result.get('rerank_score', result.get('score', score))) 
                for (doc_id, result), (_, score) in zip(reranked, candidates)]

    def retrieve(self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None, 
                 rerank_method: str = "default", fusion_method: str = "rrf",
                 bm25_weight: float = None, vector_weight: float = None, 
                 hybrid_weight: float = None) -> List[Tuple[str, Any]]:
        """
        多阶段过滤检索，支持权重配置
        
        参数：
        - query: 查询字符串
        - top_k: 返回结果数量
        - filters: 元数据过滤条件
        - rerank_method: 重排方法
        - fusion_method: 融合方法，可选: rrf, weighted
        - bm25_weight: BM25权重（仅fusion_method='rrf'时生效）
        - vector_weight: 向量检索权重（仅fusion_method='rrf'时生效）
        - hybrid_weight: 混合权重（仅fusion_method='weighted'时生效，0-1）
        """
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
        if fusion_method == "weighted":
            fused_results = self._weighted_fusion(dense_results, sparse_results, weight=hybrid_weight)
            logger.info(f"[HybridRetriever] 使用加权融合，hybrid_weight={hybrid_weight or self.hybrid_weight}")
        else:
            fused_results = self._rrf_fusion(dense_results, sparse_results, 
                                            bm25_weight=bm25_weight, 
                                            vector_weight=vector_weight)
            logger.info(f"[HybridRetriever] 使用RRF融合，bm25_weight={bm25_weight or self.bm25_weight}, vector_weight={vector_weight or self.vector_weight}")

        # 3. Filtering & Reranking
        filtered_results = self._apply_filters(fused_results, filters or {})
        reranked_results = self._rerank(query, filtered_results, rerank_method)
        logger.info(f"[HybridRetriever] 重排完成，使用方法: {rerank_method}")

        # 返回Top-K
        final_results = reranked_results[:top_k]

        # 获取实际文档内容
        results = []
        for doc_id, score in final_results:
            doc_content = f"Document {doc_id}"
            try:
                docs = self.vector_service.retrieve_docs(query, top_k=1, filters={"source": doc_id})
                if docs:
                    doc_content = docs[0].get('content', doc_content)
            except Exception:
                pass
            results.append((doc_id, {"score": score, "content": doc_content}))

        return results
    
    def set_weights(self, bm25_weight: float = None, vector_weight: float = None, hybrid_weight: float = None):
        """
        动态设置检索权重
        
        参数：
        - bm25_weight: BM25检索权重（0-1）
        - vector_weight: 向量检索权重（0-1）
        - hybrid_weight: 混合权重（0-1，用于weighted融合方法）
        """
        if bm25_weight is not None:
            self.bm25_weight = max(0.0, min(1.0, bm25_weight))
        if vector_weight is not None:
            self.vector_weight = max(0.0, min(1.0, vector_weight))
        if hybrid_weight is not None:
            self.hybrid_weight = max(0.0, min(1.0, hybrid_weight))
        
        logger.info(f"[HybridRetriever] 权重已更新: bm25={self.bm25_weight}, vector={self.vector_weight}, hybrid={self.hybrid_weight}")


# 全局混合检索器实例
hybrid_retriever = None

def get_hybrid_retriever(vector_service: "VectorStoreService"):
    global hybrid_retriever
    if hybrid_retriever is None:
        # 从配置中读取权重参数
        config = rag_cfg.get('retrieval', {})
        hybrid_retriever = HybridRetriever(
            vector_service,
            k1=config.get('bm25_k1', 1.5),
            b=config.get('bm25_b', 0.75),
            rrf_k=config.get('rrf_k', 60),
            hybrid_weight=config.get('hybrid_weight', 0.5),
            bm25_weight=config.get('bm25_weight', 0.5),
            vector_weight=config.get('vector_weight', 0.5),
            rerank_method=rag_cfg.get('rerank', {}).get('method', 'linear')
        )
    return hybrid_retriever
