# Task Decompose Skill

## Overview
将复杂目标自动拆解为可执行的子任务序列。

## Capability
- **核心能力**: 任务拆解与规划
- **输入**: 复杂目标描述
- **输出**: 结构化的子任务列表

## Parameters

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| goal | string | 是 | 复杂目标描述 |

## Returns
```json
{
  "task_list": [
    {
      "task_id": "task_1",
      "description": "子任务描述",
      "priority": 1,
      "dependencies": []
    }
  ]
}
```

## Usage Example
```json
{
  "name": "task_decompose",
  "parameters": {
    "goal": "帮我优化招聘流程"
  }
}
```

## Use Cases
- 复杂任务规划
- 项目分解
- 流程优化
- 多步骤任务管理

## Notes
- 支持中文目标描述
- 自动识别任务依赖关系
- 返回可执行的子任务序列
