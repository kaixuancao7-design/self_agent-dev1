"""
Fact Check Skill - 校验答案与参考来源的一致性
"""
import json
from langchain_core.tools import tool
from utils.logger_handler import logger


@tool(description="校验答案与检索到的原文是否一致，避免幻觉问题")
def fact_check(answer: str, sources: str) -> str:
    """
    校验答案与检索到的原文是否一致，避免"幻觉"问题
    
    :param answer: 需要核查的答案
    :param sources: 参考来源文本
    :return: JSON格式的核查结果
    """
    logger.info(f"[fact_check] 执行事实核查")
    
    try:
        from agent.modules.fact_checker import FactChecker
        
        checker = FactChecker()
        result = checker.check(answer, sources)
        logger.info(f"[fact_check] 核查完成，结果: {result.get('result', '未知')}")
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"[fact_check] 核查失败: {e}", exc_info=True)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


if __name__ == "__main__":
    # 测试
    result = fact_check.invoke({
        "answer": "AI Agent是2026年最热门的技术趋势",
        "sources": "参考资料1：根据2026年行业报告，AI Agent正在成为主流技术方向..."
    })
    print(result)
