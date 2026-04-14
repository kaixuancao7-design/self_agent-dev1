"""
BM25 索引模块
用于构建和查询 BM25 倒排索引，支持 IDF 统计信息
当前使用 pickle 存储，可迁移至 SQLite
"""

import os
import pickle
import math
from typing import Dict, List, Tuple
from collections import defaultdict
from utils.path_tool import get_abs_path
from utils.logger_handler import logger


class BM25Index:
    def __init__(self, index_path: str = None):
        self.index_path = index_path or get_abs_path("bm25_index.pkl")
        self.inverted_index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)  # term -> [(doc_id, freq), ...]
        self.doc_lengths: Dict[str, int] = {}  # doc_id -> length
        self.doc_count = 0
        self.total_terms = 0
        self.avg_doc_length = 0.0
        self.idf_cache: Dict[str, float] = {}
        self.k1 = 1.5  # BM25 参数
        self.b = 0.75

    def add_document(self, doc_id: str, text: str):
        """添加文档到索引"""
        terms = self._tokenize(text)
        term_freq = defaultdict(int)
        for term in terms:
            term_freq[term] += 1

        doc_length = len(terms)
        self.doc_lengths[doc_id] = doc_length
        self.doc_count += 1
        self.total_terms += doc_length

        for term, freq in term_freq.items():
            self.inverted_index[term].append((doc_id, freq))

        self.avg_doc_length = self.total_terms / self.doc_count if self.doc_count > 0 else 0
        self.idf_cache.clear()  # 重置 IDF 缓存

    def _tokenize(self, text: str) -> List[str]:
        """简单分词，按空格分割，可扩展为更复杂的分词"""
        return text.lower().split()

    def _idf(self, term: str) -> float:
        """计算 IDF"""
        if term in self.idf_cache:
            return self.idf_cache[term]
        df = len(self.inverted_index.get(term, []))
        if df == 0:
            idf = 0.0
        else:
            idf = math.log((self.doc_count - df + 0.5) / (df + 0.5))
        self.idf_cache[term] = idf
        return idf

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """搜索查询，返回 (doc_id, score) 列表"""
        query_terms = self._tokenize(query)
        scores = defaultdict(float)

        for term in query_terms:
            if term not in self.inverted_index:
                continue
            idf = self._idf(term)
            postings = self.inverted_index[term]
            for doc_id, freq in postings:
                tf = freq / (freq + self.k1 * (1 - self.b + self.b * (self.doc_lengths[doc_id] / self.avg_doc_length)))
                scores[doc_id] += idf * tf

        # 排序并返回 top_k
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_k]

    def save(self):
        """保存索引到 pickle 文件"""
        data = {
            'inverted_index': dict(self.inverted_index),
            'doc_lengths': self.doc_lengths,
            'doc_count': self.doc_count,
            'total_terms': self.total_terms,
            'avg_doc_length': self.avg_doc_length,
            'k1': self.k1,
            'b': self.b
        }
        with open(self.index_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"BM25 索引已保存到 {self.index_path}")

    def load(self):
        """从 pickle 文件加载索引"""
        if not os.path.exists(self.index_path):
            logger.warning(f"BM25 索引文件不存在: {self.index_path}")
            return
        with open(self.index_path, 'rb') as f:
            data = pickle.load(f)
        self.inverted_index = defaultdict(list, data['inverted_index'])
        self.doc_lengths = data['doc_lengths']
        self.doc_count = data['doc_count']
        self.total_terms = data['total_terms']
        self.avg_doc_length = data['avg_doc_length']
        self.k1 = data.get('k1', 1.5)
        self.b = data.get('b', 0.75)
        logger.info(f"BM25 索引已从 {self.index_path} 加载")

    def get_metadata(self) -> Dict:
        """获取索引元数据"""
        return {
            'doc_count': self.doc_count,
            'total_terms': self.total_terms,
            'avg_doc_length': self.avg_doc_length,
            'unique_terms': len(self.inverted_index),
            'k1': self.k1,
            'b': self.b
        }


# 全局 BM25 索引实例
bm25_index = BM25Index()


if __name__ == "__main__":
    # 示例使用
    bm25_index.add_document("doc1", "hello world")
    bm25_index.add_document("doc2", "hello python world")
    bm25_index.save()
    bm25_index.load()
    results = bm25_index.search("hello")
    print(results)