# Content Generator Skill

## Overview
智能内容生成技能，能够基于知识库内容和对话上下文，直接生成可复用的成果，如PPT演示文稿、思维导图、知识图解或长篇报告。

## Capability
- **核心能力**: 智能内容生成
- **输入**: 任务类型、内容、风格
- **输出**: JSON格式的生成结果

## Parameters

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| task_type | string | 是 | - | 生成类型 |
| content | string | 是 | - | 输入内容（对话上下文、知识库内容或主题描述） |
| style | string | 否 | professional | 输出风格 |

### Task Types

| Type | Description |
|------|-------------|
| ppt_outline | 生成PPT演示文稿大纲 |
| mindmap | 生成思维导图结构 |
| report | 生成长篇报告 |
| summary | 生成摘要总结 |

### Style Options

| Style | Description |
|-------|-------------|
| professional | 专业风格（正式、严谨） |
| casual | 轻松风格（活泼、易懂） |
| academic | 学术风格（严谨、详细） |

## Returns
```json
{
  "success": true,
  "task_type": "ppt_outline",
  "style": "professional",
  "title": "PPT标题",
  "slides": [...],
  "generated_at": "2026-04-16T10:30:00"
}
```

## Usage Example
```json
{
  "name": "content_generator",
  "parameters": {
    "task_type": "ppt_outline",
    "content": "AI Agent技术趋势分析：自主决策能力、多模态整合、垂直领域落地",
    "style": "professional"
  }
}
```

## Use Cases
- 会议结束后将讨论要点制作成PPT
- 根据知识库内容生成培训材料
- 自动生成项目总结报告
- 创建思维导图帮助知识梳理

## Notes
- 输入内容越详细，生成结果越精准
- 支持多种输出风格以适应不同场景
- 生成的PPT大纲可直接用于制作演示文稿
