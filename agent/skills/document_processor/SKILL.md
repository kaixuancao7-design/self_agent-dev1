# Document Processor Skill

## Overview
文档处理技能，具备读取、解析和提取多种格式文档（PDF、Word、Excel、PPT）信息的能力。包括多页内容识别、表格数据提取、文本分析等功能。

## Capability
- **核心能力**: 文档处理与分析
- **输入**: 操作类型、文件路径或内容
- **输出**: 处理后的文本、表格或分析结果

## Parameters

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| action | string | 是 | - | 操作类型 |
| file_path | string | 否 | - | 本地文件路径（与content二选一） |
| content | string | 否 | - | 文档内容（与file_path二选一） |
| output_format | string | 否 | text | 输出格式 |

### Action Types

| Action | Description |
|--------|-------------|
| extract_text | 提取文档文本内容 |
| extract_tables | 提取文档中的表格数据 |
| analyze | 分析文档结构和统计信息 |
| summarize | 生成文档摘要 |

### Output Formats

| Format | Description |
|--------|-------------|
| text | 纯文本格式 |
| json | JSON格式 |
| markdown | Markdown格式 |

### Supported File Types
- PDF (.pdf)
- Word (.docx)
- Excel (.xlsx, .xls)
- CSV (.csv)
- Text (.txt)
- Markdown (.md)
- HTML (.html, .htm)
- 图片 (.png, .jpg, .jpeg, .gif)

## Returns
```json
{
  "success": true,
  "action": "extract_text",
  "output_format": "json",
  "total_pages": 5,
  "processed_at": "2026-04-16T10:30:00",
  "data": [
    {"page": 1, "content": "...", "source": "document.pdf"},
    {"page": 2, "content": "...", "source": "document.pdf"}
  ]
}
```

## Usage Example
```json
{
  "name": "document_processor",
  "parameters": {
    "action": "summarize",
    "file_path": "/path/to/document.pdf",
    "output_format": "markdown"
  }
}
```

## Use Cases
- 提取PDF文档中的文本内容
- 从Excel表格中提取数据
- 分析文档结构和统计信息
- 自动生成文档摘要

## Notes
- 文件路径需要是可访问的本地路径
- 大文件可能需要较长处理时间
- 表格提取功能适用于结构化表格数据
