# Web Search Skill

## Overview
进行网络搜索，获取最新的信息和新闻。

## Capability
- **核心能力**: 实时网络搜索
- **输入**: 搜索关键词、最大结果数
- **输出**: JSON格式的搜索结果列表

## Parameters

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| query | string | 是 | - | 搜索关键词 |
| max_results | int | 否 | 5 | 返回结果数量 |

## Returns
```json
[
  {
    "title": "文章标题",
    "link": "文章链接",
    "summary": "文章摘要",
    "source": "来源"
  }
]
```

## Usage Example
```json
{
  "name": "web_search",
  "parameters": {
    "query": "AI Agent 2026 趋势",
    "max_results": 3
  }
}
```

## Use Cases
- 获取最新新闻资讯
- 查询行业动态
- 收集实时信息
- 补充知识库中没有的内容

## Notes
- 需配置搜索引擎API密钥
- 搜索结果按相关性排序
- 支持中英文搜索
