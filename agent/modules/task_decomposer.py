from typing import List, Dict, Any, Optional
from model.factory import chat_model
from utils.logger_handler import logger
from utils.prompt_loader import load_prompt


class TaskDecomposer:
    """
    任务拆解模块：将复杂目标拆解为可执行的子任务序列
    """
    
    def __init__(self):
        self.model = chat_model
        self.decompose_prompt = load_prompt("task_decompose")
    
    def decompose(self, goal: str, context: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        将复杂目标拆解为子任务
        
        :param goal: 用户的复杂目标
        :param context: 可选的上下文信息
        :return: 子任务列表，每个任务包含 id, title, description, dependencies, priority
        """
        prompt = self._build_decompose_prompt(goal, context)
        
        try:
            response = self.model.invoke(prompt)
            result = self._parse_decompose_result(response)
            
            logger.info(f"[TaskDecomposer] 任务拆解完成，目标: {goal}，子任务数: {len(result)}")
            return result
        except Exception as e:
            logger.error(f"[TaskDecomposer] 任务拆解失败: {e}")
            return self._fallback_decompose(goal)
    
    def _build_decompose_prompt(self, goal: str, context: Optional[str]) -> str:
        """构建任务拆解提示词"""
        base_prompt = self.decompose_prompt if self.decompose_prompt else self._default_decompose_prompt()
        
        context_section = f"\n上下文信息：\n{context}" if context else ""
        
        return f"""{base_prompt}
        
用户目标：{goal}{context_section}

请输出拆解后的子任务列表。
"""
    
    def _default_decompose_prompt(self) -> str:
        """默认任务拆解提示词"""
        return """你是一位专业的任务拆解专家。请将用户的复杂目标拆解为一系列可执行的子任务。

拆解规则：
1. 子任务应具有明确的目标和可验证的结果
2. 子任务之间应有合理的依赖关系
3. 子任务数量应适中（3-8个）
4. 每个子任务应标注优先级（高、中、低）

输出格式（JSON）：
[
    {
        "id": "task_1",
        "title": "子任务标题",
        "description": "子任务详细描述",
        "dependencies": ["task_2"],
        "priority": "high",
        "tool": "工具名称（如适用）"
    }
]
"""
    
    def _parse_decompose_result(self, response: Any) -> List[Dict[str, Any]]:
        """解析模型返回的任务拆解结果"""
        import json
        
        content = getattr(response, 'content', str(response))
        
        try:
            # 尝试提取 JSON
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                result = json.loads(json_str)
                
                if isinstance(result, list):
                    return result
        except json.JSONDecodeError:
            logger.warning("[TaskDecomposer] JSON解析失败，使用默认拆解")
        
        return self._fallback_decompose(content)
    
    def _fallback_decompose(self, goal: str) -> List[Dict[str, Any]]:
        """默认的简单任务拆解"""
        return [
            {
                "id": "task_1",
                "title": "分析需求",
                "description": f"分析用户目标：{goal}",
                "dependencies": [],
                "priority": "high",
                "tool": None
            },
            {
                "id": "task_2",
                "title": "检索相关信息",
                "description": "从知识库中检索与目标相关的信息",
                "dependencies": ["task_1"],
                "priority": "high",
                "tool": "rag_summarize"
            },
            {
                "id": "task_3",
                "title": "生成解决方案",
                "description": "基于检索结果生成解决方案",
                "dependencies": ["task_2"],
                "priority": "high",
                "tool": None
            }
        ]


# 全局实例
task_decomposer = TaskDecomposer()
