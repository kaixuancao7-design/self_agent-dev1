"""
提示词相关函数 - 集成动态生成、版本管理和反馈优化
"""
from utils.config_handler import prompts_cfg
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from utils.template_utils import replace_template_variables

def _replace_template_variables(content: str, context: dict) -> str:
    """
    替换提示词中的模板变量（封装公共函数以保持向后兼容）
    
    :param content: 提示词内容
    :param context: 上下文变量字典
    :return: 替换后的内容
    """
    return replace_template_variables(content, context)

def load_system_prompt(intent_type: str = None, context: dict = None) -> str:
    """
    加载系统主提示词（支持动态生成）
    
    :param intent_type: 用户意图类型（可选）
    :param context: 上下文信息（可选）
    :return: 提示词内容
    """
    try:
        # 延迟导入，避免循环依赖
        from utils.prompt_manager import generate_dynamic_prompt
        
        # 尝试使用动态生成
        prompt = generate_dynamic_prompt("main", intent_type, context)
        if prompt:
            return prompt
        
        # 降级到文件加载
        system_prompt_path = get_abs_path(prompts_cfg["main_prompt_path"])
        with open(system_prompt_path, 'r', encoding='utf-8') as f:
            content = f.read()
            return _replace_template_variables(content, context)
    except Exception as e:
        logger.error(f"[加载主提示词]加载失败：{str(e)}")
        raise e

def load_rag_prompt(input: str = None, context: str = None) -> str:
    """
    加载RAG总结提示词（支持动态生成）
    
    :param input: 用户提问（可选）
    :param context: 参考资料（可选）
    :return: 提示词内容
    """
    try:
        # 延迟导入，避免循环依赖
        from utils.prompt_manager import generate_dynamic_prompt
        
        # 尝试使用动态生成
        prompt_context = {"input": input, "context": context} if input or context else None
        prompt = generate_dynamic_prompt("rag_summarize", None, prompt_context)
        if prompt:
            return prompt
        
        # 降级到文件加载
        rag_prompt_path = get_abs_path(prompts_cfg["rag_summary_prompt_path"])
        with open(rag_prompt_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 使用统一的模板替换函数
            return _replace_template_variables(content, {"input": input, "context": context})
    except Exception as e:
        logger.error(f"[加载RAG提示词]加载失败：{str(e)}")
        raise e

def load_report_prompt(context: dict = None) -> str:
    """
    加载报告生成提示词（支持动态生成）
    
    :param context: 上下文信息（可选）
    :return: 提示词内容
    """
    try:
        # 延迟导入，避免循环依赖
        from utils.prompt_manager import generate_dynamic_prompt
        
        prompt = generate_dynamic_prompt("report", None, context)
        if prompt:
            return prompt
        
        report_prompt_path = get_abs_path(prompts_cfg["report_prompt_path"])
        with open(report_prompt_path, 'r', encoding='utf-8') as f:
            content = f.read()
            return _replace_template_variables(content, context)
    except Exception as e:
        logger.error(f"[加载报告提示词]加载失败：{str(e)}")
        raise e

def load_prompt(prompt_name: str, intent_type: str = None, context: dict = None) -> str:
    """
    通用提示词加载函数（支持动态生成）
    
    :param prompt_name: 提示词名称（不带扩展名）
    :param intent_type: 用户意图类型（可选）
    :param context: 上下文信息（可选）
    :return: 提示词内容
    """
    import os
    
    try:
        # 延迟导入，避免循环依赖
        from utils.prompt_manager import generate_dynamic_prompt
        
        # 先尝试动态生成
        prompt = generate_dynamic_prompt(prompt_name, intent_type, context)
        if prompt:
            return prompt
        
        # 降级到文件加载
        prompt_dir = get_abs_path(prompts_cfg.get("prompt_dir", "prompts"))
        prompt_path = os.path.join(prompt_dir, f"{prompt_name}.txt")
        
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # 使用统一的模板替换函数
                return _replace_template_variables(content, context)
        else:
            logger.warning(f"[加载提示词]提示词文件不存在：{prompt_path}")
            return ""
    except Exception as e:
        logger.error(f"[加载提示词]加载失败：{str(e)}")
        return ""

# 以下为版本管理相关函数
def create_prompt_version(version_name: str, prompt_type: str, content: str):
    """创建提示词版本"""
    from utils.prompt_manager import version_manager
    version_manager.create_version(version_name, prompt_type, content)

def get_prompt_version(version_name: str, prompt_type: str) -> str:
    """获取指定版本的提示词"""
    from utils.prompt_manager import version_manager
    return version_manager.get_version(version_name, prompt_type)

def list_prompt_versions(prompt_type: str = None) -> list:
    """列出所有版本"""
    from utils.prompt_manager import version_manager
    return version_manager.list_versions(prompt_type)

def activate_prompt_version(version_name: str):
    """激活指定版本"""
    from utils.prompt_manager import version_manager
    version_manager.activate_version(version_name)

def rollback_prompt_version(version_name: str):
    """回滚到指定版本"""
    from utils.prompt_manager import version_manager
    version_manager.rollback(version_name)

# 以下为反馈收集相关函数
def collect_feedback(prompt_version: str, query: str, response: str, 
                    rating: int, comment: str = "", issue_type: str = None):
    """收集用户反馈"""
    from utils.prompt_manager import feedback_collector
    feedback_collector.record_feedback(prompt_version, query, response, 
                                       rating, comment, issue_type)

def get_feedback_stats() -> dict:
    """获取反馈统计信息"""
    from utils.prompt_manager import feedback_collector
    return feedback_collector.get_feedback_stats()

# 以下为优化相关函数
def analyze_feedback() -> dict:
    """分析反馈数据"""
    from utils.prompt_manager import feedback_optimizer
    return feedback_optimizer.analyze_feedback()

def generate_optimization_suggestions() -> list:
    """生成优化建议"""
    from utils.prompt_manager import feedback_optimizer
    return feedback_optimizer.generate_optimization_suggestions()

def apply_optimizations(prompt_type: str = "main") -> str:
    """应用自动优化"""
    from utils.prompt_manager import feedback_optimizer
    return feedback_optimizer.apply_optimizations(prompt_type)

if __name__ == "__main__":
    print("=== 测试动态提示词生成 ===")
    prompt = load_system_prompt("knowledge_query")
    print(prompt[:300])
    
    print("\n=== 测试版本管理 ===")
    from utils.prompt_manager import version_manager
    print("当前版本:", version_manager.current_version)
    print("可用版本:", list_prompt_versions())
    
    print("\n=== 测试反馈收集 ===")
    collect_feedback(
        prompt_version="v1.0",
        query="测试问题",
        response="测试回答",
        rating=5,
        comment="测试反馈"
    )
    print("反馈统计:", get_feedback_stats())
    
    print("\n=== 测试优化建议 ===")
    suggestions = generate_optimization_suggestions()
    print(f"生成{len(suggestions)}条优化建议")
