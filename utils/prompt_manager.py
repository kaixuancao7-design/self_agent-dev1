"""
提示词管理器 - 实现动态提示词生成、版本管理、反馈收集和自动优化
"""
import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from utils.path_tool import get_abs_path
from utils.config_handler import prompts_cfg
from utils.logger_handler import logger

class DynamicPromptGenerator:
    """
    动态提示词生成器
    根据用户问题类型和上下文生成定制化提示词
    """
    
    def __init__(self):
        self.base_prompts = self._load_base_prompts()
        self.intent_rules = {
            "knowledge_query": self._get_knowledge_query_rules(),
            "task_request": self._get_task_request_rules(),
            "data_retrieval": self._get_data_retrieval_rules(),
            "conversation": self._get_conversation_rules(),
            "report": self._get_report_rules()
        }
    
    def _load_base_prompts(self) -> Dict[str, str]:
        """加载基础提示词模板"""
        prompt_dir = get_abs_path(prompts_cfg.get("prompt_dir", "prompts"))
        prompts = {}
        if not os.path.exists(prompt_dir):
            logger.warning(f"提示词目录不存在: {prompt_dir}")
            return prompts
        
        for filename in os.listdir(prompt_dir):
            if filename.endswith(".txt"):
                name = filename[:-4]
                try:
                    with open(os.path.join(prompt_dir, filename), 'r', encoding='utf-8') as f:
                        prompts[name] = f.read()
                except Exception as e:
                    logger.error(f"加载提示词文件{filename}失败: {e}")
        return prompts
    
    def _get_knowledge_query_rules(self) -> str:
        """知识查询专用规则"""
        return """
## 知识查询专用规则
1. 优先从知识库检索信息
2. 标注信息来源（如：【文档1】）
3. 保持回答简洁（100-150字）
4. 确保事实准确，禁止编造
"""
    
    def _get_task_request_rules(self) -> str:
        """任务请求专用规则"""
        return """
## 任务请求专用规则
1. 先拆解任务（3-5个子任务）
2. 评估每个子任务的可行性
3. 提供执行计划和预期交付物
4. 识别需要调用的工具
"""
    
    def _get_data_retrieval_rules(self) -> str:
        """数据检索专用规则"""
        return """
## 数据检索专用规则
1. 获取用户相关数据
2. 数据格式清晰易读
3. 保护用户隐私
4. 数据脱敏处理
"""
    
    def _get_conversation_rules(self) -> str:
        """闲聊对话专用规则"""
        return """
## 闲聊对话专用规则
1. 保持友好自然的语气
2. 适当展开话题
3. 避免过于正式
4. 必要时调用工具补充信息
"""
    
    def _get_report_rules(self) -> str:
        """报告生成专用规则"""
        return """
## 报告生成专用规则
1. 严格按照流程：获取用户ID → 获取月份 → 填充上下文 → 获取外部数据
2. 报告内容结构化
3. 包含数据图表和分析结论
4. 格式美观专业
"""
    
    def generate_prompt(self, prompt_type: str = "main", 
                       intent_type: str = None, 
                       context: Dict[str, Any] = None) -> str:
        """
        生成定制化提示词
        
        :param prompt_type: 提示词类型（main/rag_summarize/task_decompose等）
        :param intent_type: 用户意图类型（knowledge_query/task_request等）
        :param context: 上下文信息（用于模板替换）
        :return: 生成的提示词
        """
        # 获取基础提示词
        base_prompt = self.base_prompts.get(prompt_type, "")
        if not base_prompt:
            logger.warning(f"未找到基础提示词: {prompt_type}")
            return ""
        
        # 根据意图类型添加专用规则
        if intent_type and intent_type in self.intent_rules:
            base_prompt += self.intent_rules[intent_type]
        
        # 替换模板变量
        if context:
            base_prompt = self._replace_template_variables(base_prompt, context)
        
        return base_prompt
    
    def _replace_template_variables(self, prompt: str, context: Dict[str, Any]) -> str:
        """替换提示词中的模板变量（调用统一函数）"""
        from utils.prompt_loader import _replace_template_variables as replace_vars
        return replace_vars(prompt, context)


class PromptVersionManager:
    """
    提示词版本管理器
    支持提示词版本管理、A/B测试、回滚等功能
    """
    
    def __init__(self):
        self.version_dir = get_abs_path("prompts/versions")
        os.makedirs(self.version_dir, exist_ok=True)
        self.current_version = self._get_current_version()
    
    def _get_current_version(self) -> str:
        """获取当前使用的版本"""
        version_file = os.path.join(self.version_dir, "current_version.txt")
        if os.path.exists(version_file):
            with open(version_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        return "v1.0"
    
    def _set_current_version(self, version: str):
        """设置当前使用的版本"""
        version_file = os.path.join(self.version_dir, "current_version.txt")
        with open(version_file, 'w', encoding='utf-8') as f:
            f.write(version)
        self.current_version = version
    
    def create_version(self, version_name: str, prompt_type: str, content: str):
        """
        创建提示词版本
        
        :param version_name: 版本名称（如 v1.1）
        :param prompt_type: 提示词类型（main/rag_summarize等）
        :param content: 提示词内容
        """
        version_path = os.path.join(self.version_dir, version_name)
        os.makedirs(version_path, exist_ok=True)
        
        prompt_path = os.path.join(version_path, f"{prompt_type}.txt")
        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # 记录版本信息
        info_path = os.path.join(version_path, "version_info.json")
        info = {
            "version": version_name,
            "prompt_type": prompt_type,
            "created_at": datetime.now().isoformat(),
            "content_hash": hashlib.md5(content.encode()).hexdigest()
        }
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"创建提示词版本: {version_name}/{prompt_type}")
    
    def get_version(self, version_name: str, prompt_type: str) -> Optional[str]:
        """
        获取指定版本的提示词
        
        :param version_name: 版本名称
        :param prompt_type: 提示词类型
        :return: 提示词内容，如果不存在返回None
        """
        prompt_path = os.path.join(self.version_dir, version_name, f"{prompt_type}.txt")
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        return None
    
    def list_versions(self, prompt_type: str = None) -> List[str]:
        """
        列出所有版本
        
        :param prompt_type: 提示词类型（可选）
        :return: 版本名称列表
        """
        versions = []
        for item in os.listdir(self.version_dir):
            item_path = os.path.join(self.version_dir, item)
            if os.path.isdir(item_path):
                info_path = os.path.join(item_path, "version_info.json")
                if os.path.exists(info_path):
                    with open(info_path, 'r', encoding='utf-8') as f:
                        info = json.load(f)
                        if prompt_type is None or info.get("prompt_type") == prompt_type:
                            versions.append(item)
        return sorted(versions)
    
    def activate_version(self, version_name: str):
        """激活指定版本"""
        self._set_current_version(version_name)
        logger.info(f"激活提示词版本: {version_name}")
    
    def rollback(self, version_name: str):
        """回滚到指定版本"""
        if version_name in self.list_versions():
            self.activate_version(version_name)
            logger.info(f"回滚到版本: {version_name}")
        else:
            logger.error(f"版本不存在: {version_name}")
    
    def ab_test(self, versions: List[str], test_cases: List[Dict[str, str]],
               evaluator_func) -> Dict[str, float]:
        """
        A/B测试不同版本的提示词效果
        
        :param versions: 要测试的版本列表
        :param test_cases: 测试用例列表，每个用例包含query和expected
        :param evaluator_func: 评估函数，返回分数（0-1）
        :return: 各版本的平均分数
        """
        results = {v: [] for v in versions}
        
        for test_case in test_cases:
            query = test_case.get("query", "")
            expected = test_case.get("expected", "")
            
            for version in versions:
                prompt = self.get_version(version, "main")
                if prompt:
                    score = evaluator_func(prompt, query, expected)
                    results[version].append(score)
        
        # 计算平均分
        averages = {}
        for version, scores in results.items():
            if scores:
                averages[version] = sum(scores) / len(scores)
            else:
                averages[version] = 0.0
        
        return averages


class FeedbackCollector:
    """
    反馈收集系统
    收集用户反馈并存储，支持后续分析
    """
    
    def __init__(self):
        self.feedback_dir = get_abs_path("prompts/feedback")
        os.makedirs(self.feedback_dir, exist_ok=True)
        self.feedback_store = self._load_feedback()
    
    def _load_feedback(self) -> List[Dict[str, Any]]:
        """加载已存储的反馈"""
        feedback_file = os.path.join(self.feedback_dir, "feedback.json")
        if os.path.exists(feedback_file):
            with open(feedback_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def _save_feedback(self):
        """保存反馈到文件"""
        feedback_file = os.path.join(self.feedback_dir, "feedback.json")
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(self.feedback_store, f, ensure_ascii=False, indent=2)
    
    def record_feedback(self, prompt_version: str, query: str, 
                       response: str, rating: int, comment: str = "",
                       issue_type: str = None):
        """
        记录用户反馈
        
        :param prompt_version: 使用的提示词版本
        :param query: 用户查询
        :param response: AI响应
        :param rating: 评分（1-5分）
        :param comment: 用户评论
        :param issue_type: 问题类型（hallucination/irrelevant/too_long/too_short等）
        :raises ValueError: 当参数验证失败时
        """
        # 验证必填字段
        if not prompt_version:
            raise ValueError("prompt_version不能为空")
        if not query:
            raise ValueError("query不能为空")
        if not response:
            raise ValueError("response不能为空")
        
        # 验证评分范围
        if not isinstance(rating, int) or not (1 <= rating <= 5):
            raise ValueError("评分必须在1-5之间")
        
        # 验证issue_type（如果提供）
        valid_issue_types = [None, "hallucination", "irrelevant", "too_long", 
                            "too_short", "fact_error"]
        if issue_type not in valid_issue_types:
            raise ValueError(f"无效的issue_type: {issue_type}，有效值: {valid_issue_types}")
        
        feedback = {
            "id": hashlib.md5(f"{query}{response}{datetime.now()}".encode()).hexdigest(),
            "prompt_version": prompt_version,
            "query": query,
            "response": response,
            "rating": rating,
            "comment": comment,
            "issue_type": issue_type,
            "created_at": datetime.now().isoformat()
        }
        
        self.feedback_store.append(feedback)
        self._save_feedback()
        logger.info(f"记录反馈: {prompt_version}, 评分: {rating}")
    
    def get_feedback_by_version(self, version: str) -> List[Dict[str, Any]]:
        """获取指定版本的反馈"""
        return [f for f in self.feedback_store if f.get("prompt_version") == version]
    
    def get_low_rating_feedback(self, threshold: int = 3) -> List[Dict[str, Any]]:
        """获取低评分反馈"""
        return [f for f in self.feedback_store if f.get("rating", 5) < threshold]
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """获取反馈统计信息"""
        if not self.feedback_store:
            return {
                "total_count": 0,
                "average_rating": 0.0,
                "version_distribution": {},
                "issue_distribution": {}
            }
        
        total_count = len(self.feedback_store)
        avg_rating = sum(f.get("rating", 0) for f in self.feedback_store) / total_count
        
        version_dist = {}
        issue_dist = {}
        
        for f in self.feedback_store:
            version = f.get("prompt_version", "unknown")
            version_dist[version] = version_dist.get(version, 0) + 1
            
            issue = f.get("issue_type", "none")
            issue_dist[issue] = issue_dist.get(issue, 0) + 1
        
        return {
            "total_count": total_count,
            "average_rating": round(avg_rating, 2),
            "version_distribution": version_dist,
            "issue_distribution": issue_dist
        }


class FeedbackOptimizer:
    """
    基于反馈的自动优化器
    分析反馈数据，生成优化建议
    """
    
    def __init__(self):
        self.feedback_collector = FeedbackCollector()
        self.version_manager = PromptVersionManager()
        self.optimization_rules = {
            "hallucination": self._optimize_hallucination,
            "irrelevant": self._optimize_irrelevant,
            "too_long": self._optimize_too_long,
            "too_short": self._optimize_too_short,
            "fact_error": self._optimize_fact_error
        }
    
    def analyze_feedback(self) -> Dict[str, Any]:
        """分析反馈数据"""
        stats = self.feedback_collector.get_feedback_stats()
        low_rating = self.feedback_collector.get_low_rating_feedback()
        
        # 识别主要问题类型
        issue_counts = stats.get("issue_distribution", {})
        top_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            "stats": stats,
            "top_issues": top_issues,
            "low_rating_examples": low_rating[:5]  # 取前5个低评分示例
        }
    
    def _optimize_hallucination(self, examples: List[Dict[str, Any]]) -> str:
        """针对幻觉问题生成优化建议"""
        return """
## 幻觉问题优化建议
1. 增强事实核查规则，增加引用来源要求
2. 在提示词中添加："回答必须完全基于参考资料，禁止编造"
3. 增加验证步骤，确保每个结论都有文档支持
4. 对不确定的内容明确说明，不猜测
"""
    
    def _optimize_irrelevant(self, examples: List[Dict[str, Any]]) -> str:
        """针对不相关问题生成优化建议"""
        return """
## 不相关问题优化建议
1. 优化工具选择逻辑，增加意图识别精度
2. 在提示词中强调："严格围绕用户问题回答"
3. 添加问题分类步骤，确保正确理解用户意图
4. 增加反问确认机制，当意图不明确时询问用户
"""
    
    def _optimize_too_long(self, examples: List[Dict[str, Any]]) -> str:
        """针对回答过长问题生成优化建议"""
        return """
## 回答过长优化建议
1. 在提示词中添加明确的字数限制（如：50-150字）
2. 强制使用分点回答，每个要点简洁明了
3. 增加"总结"要求，提取核心要点
4. 移除冗余信息，只保留关键内容
"""
    
    def _optimize_too_short(self, examples: List[Dict[str, Any]]) -> str:
        """针对回答过短问题生成优化建议"""
        return """
## 回答过短优化建议
1. 在提示词中要求提供详细解释
2. 增加"为什么"和"如何"的追问引导
3. 要求提供具体示例或步骤
4. 设置最小回答长度要求
"""
    
    def _optimize_fact_error(self, examples: List[Dict[str, Any]]) -> str:
        """针对事实错误问题生成优化建议"""
        return """
## 事实错误优化建议
1. 加强事实核查机制，增加多来源验证
2. 在提示词中强调引用权威来源
3. 增加数据验证步骤
4. 对关键数据要求提供来源链接或文档引用
"""
    
    def generate_optimization_suggestions(self) -> List[str]:
        """生成优化建议列表"""
        analysis = self.analyze_feedback()
        suggestions = []
        
        for issue_type, count in analysis.get("top_issues", []):
            if issue_type != "none" and issue_type in self.optimization_rules:
                examples = [f for f in self.feedback_collector.feedback_store 
                          if f.get("issue_type") == issue_type][:3]
                suggestion = self.optimization_rules[issue_type](examples)
                suggestions.append(suggestion)
        
        return suggestions
    
    def apply_optimizations(self, prompt_type: str = "main") -> str:
        """
        自动应用优化到提示词
        
        :param prompt_type: 提示词类型
        :return: 优化后的提示词版本名称
        """
        suggestions = self.generate_optimization_suggestions()
        
        if not suggestions:
            logger.info("没有需要应用的优化建议")
            return self.version_manager.current_version
        
        # 获取当前提示词
        current_prompt = self.version_manager.get_version(
            self.version_manager.current_version, prompt_type
        )
        
        if not current_prompt:
            logger.error("无法获取当前提示词")
            return self.version_manager.current_version
        
        # 应用优化建议
        for suggestion in suggestions:
            # 提取建议中的规则并添加到提示词
            current_prompt += "\n" + suggestion
        
        # 创建新版本
        version_parts = self.version_manager.current_version.split('.')
        new_version = f"v{int(version_parts[1]) + 1}"
        
        self.version_manager.create_version(new_version, prompt_type, current_prompt)
        self.version_manager.activate_version(new_version)
        
        logger.info(f"自动优化完成，创建新版本: {new_version}")
        return new_version


# 全局实例
prompt_generator = DynamicPromptGenerator()
version_manager = PromptVersionManager()
feedback_collector = FeedbackCollector()
feedback_optimizer = FeedbackOptimizer()


# 便捷函数
def generate_dynamic_prompt(prompt_type: str = "main", 
                           intent_type: str = None, 
                           context: Dict[str, Any] = None) -> str:
    """生成动态提示词"""
    return prompt_generator.generate_prompt(prompt_type, intent_type, context)


def record_prompt_feedback(prompt_version: str, query: str, 
                          response: str, rating: int, 
                          comment: str = "", issue_type: str = None):
    """记录提示词反馈"""
    feedback_collector.record_feedback(prompt_version, query, response, 
                                       rating, comment, issue_type)


def analyze_and_optimize_prompts() -> str:
    """分析反馈并自动优化提示词"""
    return feedback_optimizer.apply_optimizations()


if __name__ == "__main__":
    # 测试动态提示词生成
    prompt = prompt_generator.generate_prompt("main", "knowledge_query")
    print("动态生成的提示词预览:")
    print(prompt[:500])
    
    # 测试版本管理
    print("\n可用版本:", version_manager.list_versions())
    
    # 测试反馈收集
    feedback_collector.record_feedback(
        prompt_version="v1.0",
        query="什么是AI?",
        response="AI是人工智能",
        rating=4,
        comment="回答简洁明了",
        issue_type=None
    )
    print("\n反馈统计:", feedback_collector.get_feedback_stats())
    
    # 测试优化建议
    suggestions = feedback_optimizer.generate_optimization_suggestions()
    print("\n优化建议:", suggestions)
