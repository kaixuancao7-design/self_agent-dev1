"""
Content Generator Skill - 智能内容生成
支持基于知识库内容和对话上下文生成可复用的成果
"""
import json
import os
from datetime import datetime
from langchain_core.tools import tool
from utils.logger_handler import logger
from model.factory import chat_model


@tool(description="智能内容生成：基于知识库内容和对话上下文生成PPT、思维导图、报告等可复用成果")
def content_generator(task_type: str, content: str, style: str = "professional") -> str:
    """
    智能内容生成
    
    :param task_type: 生成类型 (ppt_outline: PPT大纲, mindmap: 思维导图, report: 长篇报告, summary: 摘要总结)
    :param content: 输入内容（可以是对话上下文、知识库内容或主题描述）
    :param style: 风格 (professional: 专业风格, casual: 轻松风格, academic: 学术风格)
    :return: JSON格式的生成结果
    """
    logger.info(f"[content_generator] 执行任务: {task_type}, 风格: {style}")
    
    try:
        # 根据任务类型生成不同格式的内容
        prompts = {
            "ppt_outline": f"""
你是一位专业的PPT设计师，请根据以下内容生成一份完整的PPT演示文稿大纲。

输入内容：
{content}

要求：
1. 结构清晰，逻辑严谨
2. 包含封面页、目录页、各章节内容、总结页
3. 每个页面要有明确的标题和要点
4. 使用{style}风格

请输出结构化的PPT大纲，格式如下：
{{
  "title": "PPT标题",
  "slides": [
    {{"type": "封面", "title": "...", "subtitle": "..."}},
    {{"type": "目录", "items": ["...", "..."]}},
    {{"type": "内容", "title": "...", "points": ["...", "..."]}},
    {{"type": "总结", "title": "...", "key_points": ["...", "..."]}}
  ]
}}
            """,
            
            "mindmap": f"""
你是一位思维导图专家，请根据以下内容生成一份结构化的思维导图。

输入内容：
{content}

要求：
1. 层次分明，主题突出
2. 使用{style}风格
3. 至少包含3-5个主分支，每个分支下有子节点

请输出思维导图结构，格式如下：
{{
  "central_topic": "中心主题",
  "branches": [
    {{"name": "分支1", "children": ["子节点1", "子节点2", ...]}},
    {{"name": "分支2", "children": ["子节点1", "子节点2", ...]}}
  ]
}}
            """,
            
            "report": f"""
你是一位专业报告撰写专家，请根据以下内容撰写一篇详细的长篇报告。

输入内容：
{content}

要求：
1. 结构完整，包含引言、正文、结论
2. 内容详实，分析深入
3. 使用{style}风格
4. 至少2000字

请输出完整报告内容。
            """,
            
            "summary": f"""
你是一位内容摘要专家，请根据以下内容生成一份精炼的摘要总结。

输入内容：
{content}

要求：
1. 提取核心要点
2. 突出关键信息
3. 使用{style}风格
4. 控制在300-500字

请输出摘要总结内容。
            """
        }
        
        if task_type not in prompts:
            return json.dumps({
                "success": False,
                "message": f"未知的任务类型: {task_type}，支持的类型: ppt_outline, mindmap, report, summary"
            }, ensure_ascii=False)
        
        # 调用模型生成内容
        prompt = prompts[task_type]
        response = chat_model.invoke(prompt)
        
        # 尝试解析JSON格式输出
        try:
            if isinstance(response, dict) and "content" in response:
                result_text = response["content"]
            else:
                result_text = str(response)
            
            # 清理输出，提取JSON部分
            if "{" in result_text and "}" in result_text:
                json_start = result_text.find("{")
                json_end = result_text.rfind("}") + 1
                json_content = result_text[json_start:json_end]
                parsed_result = json.loads(json_content)
            else:
                # 如果不是JSON格式，直接返回文本内容
                parsed_result = {
                    "success": True,
                    "task_type": task_type,
                    "content": result_text.strip()
                }
                
        except json.JSONDecodeError:
            # JSON解析失败，返回原始文本
            parsed_result = {
                "success": True,
                "task_type": task_type,
                "content": result_text.strip()
            }
        
        # 添加元信息
        parsed_result["generated_at"] = datetime.now().isoformat()
        parsed_result["style"] = style
        
        logger.info(f"[content_generator] 生成完成")
        return json.dumps(parsed_result, ensure_ascii=False, indent=2)
    
    except Exception as e:
        logger.error(f"[content_generator] 生成失败: {e}", exc_info=True)
        return json.dumps({"success": False, "message": str(e)}, ensure_ascii=False)


if __name__ == "__main__":
    # 测试
    test_content = """
AI Agent技术是2026年最热门的技术趋势之一，它具有以下特点：
1. 自主决策能力
2. 多模态整合
3. 垂直领域落地加速

主要应用场景包括：
- 智能客服
- 自动化办公
- 数据分析
"""
    result = content_generator.invoke({"task_type": "ppt_outline", "content": test_content})
    print(result)
