from typing import Dict, Any, Optional, Tuple, List
from model.factory import chat_model
from utils.logger_handler import logger
from utils.prompt_loader import load_prompt


class FactChecker:
    """
    事实核查模块：校验答案与检索到的原文是否一致，避免"幻觉"问题
    """
    
    CONFIDENCE_THRESHOLD = 0.7  # 置信度阈值
    
    def __init__(self):
        self.model = chat_model
        self.check_prompt = load_prompt("fact_check")
    
    def check(self, answer: str, sources: List[Dict[str, Any]]) -> Tuple[float, List[Dict[str, Any]], str]:
        """
        核查答案与参考资料的一致性
        
        :param answer: 生成的答案
        :param sources: 参考资料列表
        :return: (置信度分数, 不一致列表, 修正建议)
        """
        if not sources:
            logger.warning("[FactChecker] 没有参考资料可供核查")
            return 0.0, [], "没有参考资料"
        
        prompt = self._build_check_prompt(answer, sources)
        
        try:
            response = self.model.invoke(prompt)
            confidence, inconsistencies, suggestion = self._parse_check_result(response)
            
            logger.info(f"[FactChecker] 事实核查完成，置信度: {confidence:.2f}，不一致项: {len(inconsistencies)}")
            return confidence, inconsistencies, suggestion
        except Exception as e:
            logger.error(f"[FactChecker] 核查失败: {e}")
            return self._default_check_result(answer, sources)
    
    def highlight_conflicts(self, answer: str, sources: List[Dict[str, Any]]) -> str:
        """
        高亮显示答案中与参考资料冲突的部分
        
        :param answer: 生成的答案
        :param sources: 参考资料列表
        :return: 带有高亮标记的答案
        """
        confidence, inconsistencies, _ = self.check(answer, sources)
        
        if confidence >= self.CONFIDENCE_THRESHOLD:
            return answer
        
        # 简单实现：标记不一致的部分
        result = answer
        for inconsistency in inconsistencies:
            conflicting_text = inconsistency.get("answer_text", "")
            if conflicting_text and conflicting_text in result:
                result = result.replace(conflicting_text, f"⚠️{conflicting_text}⚠️")
        
        return result
    
    def _build_check_prompt(self, answer: str, sources: List[Dict[str, Any]]) -> str:
        """构建事实核查提示词"""
        base_prompt = self.check_prompt if self.check_prompt else self._default_check_prompt()
        
        # 格式化参考资料
        sources_str = ""
        for i, source in enumerate(sources, 1):
            content = source.get("page_content", "")[:300]
            metadata = source.get("metadata", {})
            source_name = metadata.get("source", f"参考资料{i}")
            sources_str += f"参考资料{i} [{source_name}]:\n{content}\n\n"
        
        return f"""{base_prompt}

生成的答案：
{answer}

参考资料：
{sources_str}

请进行事实核查。
"""
    
    def _default_check_prompt(self) -> str:
        """默认事实核查提示词"""
        return """你是一位专业的事实核查专家。请仔细比对生成的答案与参考资料，找出所有不一致的地方。

核查标准：
1. 事实准确性：答案中的陈述是否与参考资料一致
2. 数据准确性：数字、日期、名称等是否准确
3. 引用完整性：是否正确引用了参考资料中的信息
4. 无中生有：答案中是否有参考资料中没有的内容

输出格式（JSON）：
{
    "confidence_score": 0.85,
    "inconsistencies": [
        {
            "answer_text": "答案中的错误陈述",
            "source_text": "参考资料中的正确内容",
            "conflict_type": "事实错误"
        }
    ],
    "suggestion": "修正建议"
}

confidence_score 范围：0-1，表示答案与参考资料的一致性程度
"""
    
    def _parse_check_result(self, response: Any) -> Tuple[float, List[Dict[str, Any]], str]:
        """解析核查结果"""
        import json
        
        content = getattr(response, 'content', str(response))
        
        try:
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                result = json.loads(json_str)
                
                confidence = float(result.get("confidence_score", 0.5))
                inconsistencies = result.get("inconsistencies", [])
                suggestion = result.get("suggestion", "无建议")
                
                return confidence, inconsistencies, suggestion
        except json.JSONDecodeError:
            logger.warning("[FactChecker] JSON解析失败")
        
        return self._default_check_result(content, [])
    
    def _default_check_result(self, answer: str, sources: List[Dict[str, Any]]) -> Tuple[float, List[Dict[str, Any]], str]:
        """默认核查结果"""
        if not sources:
            return 0.0, [], "没有参考资料"
        
        # 简单的文本匹配检查
        answer_lower = answer.lower()
        source_text = "\n".join(str(s.get("page_content", "")).lower() for s in sources)
        
        # 统计答案中有多少内容可以在参考资料中找到
        answer_chunks = answer_lower.split()
        found_chunks = sum(1 for chunk in answer_chunks if chunk in source_text)
        
        if len(answer_chunks) == 0:
            return 0.0, [], "答案为空"
        
        confidence = min(found_chunks / len(answer_chunks), 1.0)
        
        if confidence < 0.5:
            return confidence, [{"answer_text": answer, "source_text": "参考资料不匹配", "conflict_type": "内容不匹配"}], "答案与参考资料匹配度较低"
        
        return confidence, [], "检查通过"


# 全局实例
fact_checker = FactChecker()
