"""
高级能力工具
包含任务拆解、自我评估、事实核查等高级Agent能力
"""
from langchain_core.tools import tool
from agent.modules import task_decomposer, self_evaluator, fact_checker
from utils.logger_handler import logger
import json


@tool(description="将复杂任务拆解为可执行的子任务序列")
def task_decompose(goal: str) -> str:
    """
    将复杂目标拆解为可执行的子任务序列
    
    :param goal: 用户的复杂目标（如："帮我制定一个AI行业研究方案"）
    :return: JSON格式的子任务列表
    """
    logger.info(f"[task_decompose] 开始任务拆解: {goal}")
    try:
        tasks = task_decomposer.decompose(goal)
        result = json.dumps(tasks, ensure_ascii=False, indent=2)
        logger.info(f"[task_decompose] 拆解完成，生成 {len(tasks)} 个子任务")
        return result
    except Exception as e:
        logger.error(f"[task_decompose] 任务拆解失败: {e}", exc_info=True)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool(description="评估任务执行结果的质量")
def evaluate_result(task_description: str, result: str) -> str:
    """
    评估任务执行结果的质量
    
    :param task_description: 任务描述
    :param result: 执行结果
    :return: JSON格式的评估结果（包含质量分数、评估意见、是否需要重试）
    """
    logger.info(f"[evaluate_result] 开始评估任务: {task_description}")
    try:
        task = {
            "id": "eval_task",
            "title": task_description,
            "description": task_description
        }
        quality, feedback, need_retry = self_evaluator.evaluate(task, result)
        result = json.dumps({
            "quality_score": quality,
            "feedback": feedback,
            "need_retry": need_retry
        }, ensure_ascii=False)
        logger.info(f"[evaluate_result] 评估完成，质量分数: {quality}")
        return result
    except Exception as e:
        logger.error(f"[evaluate_result] 评估失败: {e}", exc_info=True)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool(description="核查答案与参考资料的一致性，避免幻觉问题")
def fact_check(answer: str, sources: str) -> str:
    """
    核查答案与参考资料的一致性，避免幻觉问题
    
    :param answer: 生成的答案
    :param sources: 参考资料（JSON格式列表，每个元素包含 page_content 和 metadata）
    :return: JSON格式的核查结果（包含置信度、不一致项、建议）
    """
    logger.info(f"[fact_check] 开始事实核查，答案长度: {len(answer)}")
    try:
        sources_list = json.loads(sources)
        confidence, inconsistencies, suggestion = fact_checker.check(answer, sources_list)
        result = json.dumps({
            "confidence_score": confidence,
            "inconsistencies": inconsistencies,
            "suggestion": suggestion
        }, ensure_ascii=False)
        logger.info(f"[fact_check] 核查完成，置信度: {confidence}")
        return result
    except json.JSONDecodeError:
        logger.error("[fact_check] 参考资料格式错误")
        return json.dumps({"error": "参考资料格式错误"}, ensure_ascii=False)
    except Exception as e:
        logger.error(f"[fact_check] 核查失败: {e}", exc_info=True)
        return json.dumps({"error": str(e)}, ensure_ascii=False)
