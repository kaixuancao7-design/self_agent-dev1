"""
Document Processor Skill - 文档处理
支持读取、解析和提取多种格式文档（PDF, Word, Excel, PPT）信息
"""
import json
import os
from datetime import datetime
from langchain_core.tools import tool
from utils.logger_handler import logger
from utils.file_handler import load_file_content, SUPPORTED_FILE_TYPES


@tool(description="文档处理：读取、解析和提取多种格式文档（PDF、Word、Excel、PPT）信息")
def document_processor(action: str, file_path: str = None, content: str = None, 
                       output_format: str = "text") -> str:
    """
    文档处理
    
    :param action: 操作类型 (extract_text: 提取文本, extract_tables: 提取表格, analyze: 分析文档, summarize: 文档摘要)
    :param file_path: 文件路径（本地文件）
    :param content: 文档内容（直接传入文本内容）
    :param output_format: 输出格式 (text: 纯文本, json: JSON格式, markdown: Markdown格式)
    :return: JSON格式的处理结果
    """
    logger.info(f"[document_processor] 执行操作: {action}")
    
    try:
        # 加载文档内容
        documents = []
        
        if file_path:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                return json.dumps({"success": False, "message": f"文件不存在: {file_path}"}, ensure_ascii=False)
            
            # 检查文件类型
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in SUPPORTED_FILE_TYPES:
                return json.dumps({
                    "success": False, 
                    "message": f"不支持的文件类型: {file_ext}，支持的类型: {SUPPORTED_FILE_TYPES}"
                }, ensure_ascii=False)
            
            # 加载文件内容
            documents = load_file_content(file_path)
            logger.info(f"[document_processor] 加载文件: {file_path}, 共 {len(documents)} 页")
        
        elif content:
            # 使用直接传入的内容
            from langchain_core.documents import Document
            documents = [Document(page_content=content, metadata={"source": "direct_input"})]
        else:
            return json.dumps({"success": False, "message": "必须提供 file_path 或 content 参数"}, ensure_ascii=False)
        
        # 根据操作类型处理文档
        results = []
        for doc in documents:
            page_content = doc.page_content
            metadata = doc.metadata
            
            if action == "extract_text":
                results.append({
                    "page": metadata.get("page", len(results) + 1),
                    "content": page_content[:5000],  # 限制单页内容长度
                    "source": metadata.get("source", "unknown")
                })
            
            elif action == "extract_tables":
                # 尝试从文本中提取表格数据
                tables = extract_tables_from_text(page_content)
                results.append({
                    "page": metadata.get("page", len(results) + 1),
                    "tables": tables,
                    "source": metadata.get("source", "unknown")
                })
            
            elif action == "analyze":
                # 分析文档结构
                analysis = analyze_document(page_content)
                results.append({
                    "page": metadata.get("page", len(results) + 1),
                    "analysis": analysis,
                    "source": metadata.get("source", "unknown")
                })
            
            elif action == "summarize":
                # 生成文档摘要
                summary = summarize_document(page_content)
                results.append({
                    "page": metadata.get("page", len(results) + 1),
                    "summary": summary,
                    "source": metadata.get("source", "unknown")
                })
            
            else:
                return json.dumps({"success": False, "message": f"未知操作类型: {action}"}, ensure_ascii=False)
        
        # 构建最终结果
        final_result = {
            "success": True,
            "action": action,
            "output_format": output_format,
            "total_pages": len(results),
            "processed_at": datetime.now().isoformat(),
            "data": results
        }
        
        # 根据输出格式返回结果
        if output_format == "json":
            return json.dumps(final_result, ensure_ascii=False, indent=2)
        elif output_format == "markdown":
            return format_as_markdown(final_result)
        else:
            return format_as_text(final_result)
    
    except Exception as e:
        logger.error(f"[document_processor] 处理失败: {e}", exc_info=True)
        return json.dumps({"success": False, "message": str(e)}, ensure_ascii=False)


def extract_tables_from_text(text: str) -> list:
    """
    从文本中提取表格数据（简单实现）
    """
    tables = []
    lines = text.split('\n')
    
    # 查找表格（通常包含多列数据，用空格或制表符分隔）
    current_table = []
    for line in lines:
        # 检测表格行（包含多个由空格/制表符分隔的列）
        if len(line.split()) >= 3 and any(c.isdigit() for c in line):
            current_table.append(line)
        elif current_table:
            if len(current_table) >= 2:
                tables.append({"rows": current_table.copy()})
            current_table = []
    
    if current_table and len(current_table) >= 2:
        tables.append({"rows": current_table})
    
    return tables


def analyze_document(text: str) -> dict:
    """
    分析文档结构
    """
    lines = text.split('\n')
    words = text.split()
    
    # 统计信息
    analysis = {
        "char_count": len(text),
        "word_count": len(words),
        "line_count": len(lines),
        "paragraph_count": len([l for l in lines if l.strip()]),
        "avg_sentence_length": sum(len(s.split()) for s in text.split('。')) / max(text.count('。'), 1)
    }
    
    # 提取标题（以数字或特殊字符开头的行）
    headings = []
    for line in lines[:20]:
        if line.strip() and (line[0].isdigit() or line.startswith(('【', '【', '#', '★', '●'))):
            headings.append(line.strip()[:50])
    
    if headings:
        analysis["headings"] = headings
    
    return analysis


def summarize_document(text: str) -> str:
    """
    生成文档摘要（使用模型）
    """
    from model.factory import chat_model
    
    prompt = f"""
请对以下文档内容进行摘要总结：

{text[:3000]}

要求：
1. 提取核心要点
2. 保持关键信息完整
3. 控制在200字以内
"""
    
    try:
        response = chat_model.invoke(prompt)
        if isinstance(response, dict) and "content" in response:
            return response["content"].strip()
        return str(response).strip()
    except Exception as e:
        logger.warning(f"[document_processor] 摘要生成失败，使用简单摘要: {e}")
        # 简单摘要：取前200字
        return text[:200] + "..."


def format_as_markdown(result: dict) -> str:
    """
    格式化为Markdown
    """
    md = f"# 文档处理结果\n\n"
    md += f"- **操作类型**: {result['action']}\n"
    md += f"- **总页数**: {result['total_pages']}\n"
    md += f"- **处理时间**: {result['processed_at']}\n\n"
    
    for item in result["data"]:
        md += f"## 第 {item['page']} 页\n\n"
        if "content" in item:
            md += f"{item['content'][:1000]}...\n\n"
        elif "summary" in item:
            md += f"**摘要**: {item['summary']}\n\n"
        elif "analysis" in item:
            md += "**分析结果**:\n"
            for key, value in item['analysis'].items():
                md += f"- {key}: {value}\n"
            md += "\n"
    
    return md


def format_as_text(result: dict) -> str:
    """
    格式化为纯文本
    """
    text = f"文档处理结果\n"
    text += f"操作类型: {result['action']}\n"
    text += f"总页数: {result['total_pages']}\n"
    text += f"处理时间: {result['processed_at']}\n\n"
    
    for item in result["data"]:
        text += f"=== 第 {item['page']} 页 ===\n"
        if "content" in item:
            text += f"{item['content'][:1000]}...\n\n"
        elif "summary" in item:
            text += f"摘要: {item['summary']}\n\n"
        elif "analysis" in item:
            text += "分析结果:\n"
            for key, value in item['analysis'].items():
                text += f"  {key}: {value}\n"
            text += "\n"
    
    return text


if __name__ == "__main__":
    # 测试
    test_content = """
人工智能（AI）是当今最热门的技术领域之一。

## 主要应用领域
1. 智能客服 - 自动回答用户问题
2. 自动驾驶 - 无人驾驶汽车技术
3. 医疗诊断 - 辅助医生诊断疾病

## 发展趋势
- 多模态AI整合
- 自主决策能力提升
- 垂直领域应用深化

表格示例：
年份 | 技术突破 | 代表产品
2023 | GPT-4发布 | ChatGPT
2024 | AI Agent兴起 | AutoGPT
2025 | 多模态融合 | Gemini
"""
    result = document_processor.invoke({"action": "analyze", "content": test_content})
    print(result)
