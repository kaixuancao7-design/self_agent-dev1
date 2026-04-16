# Knowledge Governance Skill

## Overview
持续知识治理技能，能够主动维护知识库的健康状态。包括生成质量报告、统计分析等功能。

## Capability
- **核心能力**: 知识库持续治理
- **输入**: 操作类型、参数
- **输出**: 治理结果报告

## Parameters

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| action | string | 是 | - | 操作类型 |
| query | string | 否 | - | 查询词（保留参数） |
| update_content | string | 否 | - | 更新内容（保留参数） |

### Action Types

| Action | Description | Required Parameters |
|--------|-------------|-------------------|
| generate_report | 生成质量报告 | 无 |
| analyze_stats | 分析统计信息 | 无 |

## Returns
```json
{
  "success": true,
  "message": "质量报告生成完成",
  "generated_at": "2026-04-16T10:30:00",
  "basic_stats": {...},
  "quality_metrics": {...},
  "recommendations": [...]
}
```

## Usage Example
```json
{
  "name": "knowledge_governance",
  "parameters": {
    "action": "generate_report"
  }
}
```

## Use Cases
- 生成知识库质量报告，了解健康状态
- 分析知识库统计信息和文件类型分布
- 获取改进建议，优化知识库内容

## Quality Metrics
质量报告包含以下指标：
- **total_chunks**: 知识库中Chunk的总数量
- **total_files**: 上传文件的总数量
- **health_score**: 知识库健康分数（0-100分）

## Notes
- 定期生成质量报告可以监控知识库健康状态
- 根据建议可以持续优化知识库内容
