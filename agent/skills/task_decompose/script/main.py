"""
Task Decompose Skill - 将复杂目标拆解为子任务序列
"""
import json
from langchain_core.tools import tool
from utils.logger_handler import logger


@tool(description="将复杂目标自动拆解为可执行的子任务序列")
def task_decompose(goal: str) -> str:
    """
    将复杂目标自动拆解为可执行的子任务序列
    
    :param goal: 复杂目标描述
    :return: JSON格式的子任务列表
    """
    logger.info(f"[task_decompose] 拆解任务: {goal}")
    
    try:
        from agent.modules.task_decomposer import TaskDecomposer
        
        decomposer = TaskDecomposer()
        result = decomposer.decompose(goal)
        logger.info(f"[task_decompose] 拆解完成")
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"[task_decompose] 拆解失败: {e}", exc_info=True)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


if __name__ == "__main__":
    # 测试
    result = task_decompose.invoke({"goal": "帮我优化招聘流程"})
    print(result)
