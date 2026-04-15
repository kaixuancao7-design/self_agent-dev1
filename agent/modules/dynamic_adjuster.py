from typing import List, Dict, Any, Optional, Callable
from utils.logger_handler import logger


class DynamicAdjuster:
    """
    动态调整策略模块：处理步骤失败后的备选方案，动态调整执行策略
    """
    
    MAX_RETRY_ATTEMPTS = 3  # 最大重试次数
    RETRY_DELAY_SECONDS = 2  # 重试延迟
    
    def __init__(self):
        self.strategies = {}
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """注册默认策略"""
        self.strategies["rag_summarize_failure"] = self._handle_rag_failure
        self.strategies["tool_timeout"] = self._handle_timeout
        self.strategies["empty_result"] = self._handle_empty_result
        self.strategies["low_quality"] = self._handle_low_quality
        self.strategies["fact_conflict"] = self._handle_fact_conflict
    
    def adjust(self, 
               task: Dict[str, Any], 
               error: Optional[Exception] = None, 
               result: Optional[Any] = None,
               quality_score: Optional[float] = None,
               confidence_score: Optional[float] = None) -> Dict[str, Any]:
        """
        根据执行结果动态调整策略
        
        :param task: 当前任务
        :param error: 错误信息（如果有）
        :param result: 执行结果
        :param quality_score: 质量分数（如果有）
        :param confidence_score: 置信度分数（如果有）
        :return: 调整策略，包含 action 和 params
        """
        # 确定问题类型
        problem_type = self._identify_problem(task, error, result, quality_score, confidence_score)
        
        # 获取对应的处理策略
        strategy = self.strategies.get(problem_type, self._handle_unknown)
        
        logger.info(f"[DynamicAdjuster] 检测到问题类型: {problem_type}，应用策略: {strategy.__name__}")
        
        return strategy(task, error, result)
    
    def _identify_problem(self, 
                         task: Dict[str, Any], 
                         error: Optional[Exception], 
                         result: Optional[Any],
                         quality_score: Optional[float],
                         confidence_score: Optional[float]) -> str:
        """识别问题类型"""
        if error:
            if "timeout" in str(error).lower():
                return "tool_timeout"
            return "unknown_error"
        
        if result is None or (isinstance(result, str) and not result.strip()):
            return "empty_result"
        
        if quality_score is not None and quality_score < 0.5:
            return "low_quality"
        
        if confidence_score is not None and confidence_score < 0.5:
            return "fact_conflict"
        
        tool = task.get("tool")
        if tool and result and (isinstance(result, str) and "无法找到" in result or "未找到" in result):
            return f"{tool}_failure"
        
        return "success"
    
    def _handle_rag_failure(self, task: Dict[str, Any], error: Optional[Exception], result: Optional[Any]) -> Dict[str, Any]:
        """处理RAG检索失败"""
        current_query = task.get("params", {}).get("query", "")
        
        # 生成备选检索词
        alternative_queries = self._generate_alternative_queries(current_query)
        
        return {
            "action": "retry_with_params",
            "params": {
                "query": alternative_queries[0] if alternative_queries else current_query
            },
            "message": f"检索失败，尝试使用替代检索词: {alternative_queries[0] if alternative_queries else current_query}"
        }
    
    def _handle_timeout(self, task: Dict[str, Any], error: Optional[Exception], result: Optional[Any]) -> Dict[str, Any]:
        """处理超时错误"""
        return {
            "action": "retry",
            "params": {},
            "message": "操作超时，正在重试..."
        }
    
    def _handle_empty_result(self, task: Dict[str, Any], error: Optional[Exception], result: Optional[Any]) -> Dict[str, Any]:
        """处理空结果"""
        tool = task.get("tool")
        
        if tool == "rag_summarize":
            return self._handle_rag_failure(task, error, result)
        
        return {
            "action": "skip",
            "params": {},
            "message": "结果为空，跳过此步骤"
        }
    
    def _handle_low_quality(self, task: Dict[str, Any], error: Optional[Exception], result: Optional[Any]) -> Dict[str, Any]:
        """处理低质量结果"""
        return {
            "action": "retry",
            "params": {},
            "message": "结果质量较低，重新执行..."
        }
    
    def _handle_fact_conflict(self, task: Dict[str, Any], error: Optional[Exception], result: Optional[Any]) -> Dict[str, Any]:
        """处理事实冲突"""
        return {
            "action": "retry",
            "params": {},
            "message": "发现事实冲突，重新生成答案..."
        }
    
    def _handle_unknown(self, task: Dict[str, Any], error: Optional[Exception], result: Optional[Any]) -> Dict[str, Any]:
        """处理未知问题"""
        return {
            "action": "skip",
            "params": {},
            "message": "遇到未知问题，跳过此步骤"
        }
    
    def _generate_alternative_queries(self, original_query: str) -> List[str]:
        """生成备选检索词"""
        alternatives = []
        
        # 简化查询
        if len(original_query) > 10:
            alternatives.append(" ".join(original_query.split()[:3]))
        
        # 添加同义词
        synonyms = {
            "最新": ["最近", "近期", "最新的"],
            "趋势": ["动态", "发展", "走向"],
            "分析": ["解析", "解读", "研究"],
            "方案": ["策略", "方法", "办法"]
        }
        
        for word, syns in synonyms.items():
            if word in original_query:
                for syn in syns:
                    alternatives.append(original_query.replace(word, syn))
        
        # 添加相关术语
        related_terms = {
            "AI": ["人工智能", "大模型", "机器学习"],
            "行业": ["领域", "产业", "市场"],
            "技术": ["方法", "手段", "技术方案"]
        }
        
        for term, related in related_terms.items():
            if term in original_query:
                alternatives.append(f"{original_query} {' '.join(related)}")
        
        return list(set(alternatives))[:3]
    
    def register_strategy(self, problem_type: str, handler: Callable):
        """注册自定义策略"""
        self.strategies[problem_type] = handler
        logger.info(f"[DynamicAdjuster] 注册新策略: {problem_type}")


# 全局实例
dynamic_adjuster = DynamicAdjuster()
