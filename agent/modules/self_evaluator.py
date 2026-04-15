from typing import Dict, Any, Optional, Tuple
from model.factory import chat_model
from utils.logger_handler import logger
from utils.prompt_loader import load_prompt


class SelfEvaluator:
    """
    自我评估模块：评估每步结果质量，判断是否需要调整策略或重新执行
    """
    
    QUALITY_THRESHOLD = 0.7  # 质量阈值，低于此值需要重新执行
    
    def __init__(self):
        self.model = chat_model
        self.evaluation_prompt = load_prompt("self_evaluation")
    
    def evaluate(self, task: Dict[str, Any], result: Any, context: Optional[str] = None) -> Tuple[float, str, bool]:
        """
        评估任务执行结果质量
        
        :param task: 当前任务
        :param result: 任务执行结果
        :param context: 上下文信息
        :return: (质量分数, 评估意见, 是否需要重新执行)
        """
        prompt = self._build_evaluation_prompt(task, result, context)
        
        try:
            response = self.model.invoke(prompt)
            quality, feedback, need_retry = self._parse_evaluation(response)
            
            logger.info(f"[SelfEvaluator] 任务 {task.get('id')} 评估完成，质量分数: {quality:.2f}，需要重试: {need_retry}")
            return quality, feedback, need_retry
        except Exception as e:
            logger.error(f"[SelfEvaluator] 评估失败: {e}")
            return self._default_evaluation(result)
    
    def evaluate_final_answer(self, question: str, answer: str, sources: Optional[str] = None) -> Tuple[float, str]:
        """
        评估最终答案质量
        
        :param question: 用户问题
        :param answer: 生成的答案
        :param sources: 参考资料
        :return: (质量分数, 改进建议)
        """
        prompt = self._build_final_evaluation_prompt(question, answer, sources)
        
        try:
            response = self.model.invoke(prompt)
            return self._parse_final_evaluation(response)
        except Exception as e:
            logger.error(f"[SelfEvaluator] 最终答案评估失败: {e}")
            return 0.5, "评估失败，使用默认分数"
    
    def _build_evaluation_prompt(self, task: Dict[str, Any], result: Any, context: Optional[str]) -> str:
        """构建任务评估提示词"""
        base_prompt = self.evaluation_prompt if self.evaluation_prompt else self._default_evaluation_prompt()
        
        result_str = str(result)[:500] if len(str(result)) > 500 else str(result)
        
        return f"""{base_prompt}

任务信息：
- 任务ID: {task.get('id')}
- 任务标题: {task.get('title')}
- 任务描述: {task.get('description')}

任务执行结果：
{result_str}

上下文信息：
{context or '无'}

请评估此结果的质量。
"""
    
    def _default_evaluation_prompt(self) -> str:
        """默认评估提示词"""
        return """你是一位专业的评估专家，负责评估任务执行结果的质量。

评估标准：
1. 相关性：结果是否与任务目标相关
2. 完整性：结果是否完整覆盖任务需求
3. 准确性：结果是否准确无误
4. 有用性：结果是否能帮助完成后续任务

输出格式（JSON）：
{
    "quality_score": 0.85,
    "feedback": "结果质量良好，能满足任务需求",
    "need_retry": false
}

quality_score 范围：0-1，0表示完全失败，1表示完美
need_retry：是否需要重新执行任务
"""
    
    def _parse_evaluation(self, response: Any) -> Tuple[float, str, bool]:
        """解析评估结果"""
        import json
        
        content = getattr(response, 'content', str(response))
        
        try:
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                result = json.loads(json_str)
                
                quality = float(result.get("quality_score", 0.5))
                feedback = result.get("feedback", "评估完成")
                need_retry = bool(result.get("need_retry", False))
                
                return quality, feedback, need_retry
        except json.JSONDecodeError:
            logger.warning("[SelfEvaluator] JSON解析失败")
        
        return self._default_evaluation(content)
    
    def _default_evaluation(self, result: Any) -> Tuple[float, str, bool]:
        """默认评估"""
        result_str = str(result)
        
        if not result_str or result_str.strip() == "":
            return 0.0, "结果为空", True
        
        if len(result_str) < 20:
            return 0.3, "结果过于简短", True
        
        return 0.7, "使用默认评估分数", False
    
    def _build_final_evaluation_prompt(self, question: str, answer: str, sources: Optional[str]) -> str:
        """构建最终答案评估提示词"""
        sources_str = sources[:1000] if sources else "无"
        
        return f"""你是一位专业的答案评估专家。请评估以下问答的质量：

用户问题：{question}

生成答案：{answer}

参考资料：{sources_str}

评估维度：
1. 准确性：答案是否准确反映参考资料
2. 完整性：是否完整回答了用户问题
3. 相关性：答案是否与问题相关
4. 逻辑性：回答是否逻辑清晰

请输出质量分数（0-1）和改进建议。

输出格式（JSON）：
{{
    "quality_score": 0.85,
    "suggestion": "建议增加更多具体案例"
}}
"""
    
    def _parse_final_evaluation(self, response: Any) -> Tuple[float, str]:
        """解析最终答案评估结果"""
        import json
        
        content = getattr(response, 'content', str(response))
        
        try:
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                result = json.loads(json_str)
                
                quality = float(result.get("quality_score", 0.5))
                suggestion = result.get("suggestion", "无建议")
                
                return quality, suggestion
        except json.JSONDecodeError:
            logger.warning("[SelfEvaluator] 最终答案评估JSON解析失败")
        
        return 0.5, "解析失败，使用默认分数"


# 全局实例
self_evaluator = SelfEvaluator()
