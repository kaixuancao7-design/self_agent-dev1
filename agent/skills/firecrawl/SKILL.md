# Firecrawl Skill

## Overview
Firecrawl MCP 技能，基于 Firecrawl API 提供网页内容抓取和处理能力，支持 JavaScript 渲染页面、批量爬取和网页搜索。

## Capability
- **核心能力**: 网页内容抓取、网站爬取、网页搜索
- **输入**: URL、搜索关键词、输出格式等参数
- **输出**: JSON格式的处理结果

## Parameters

### firecrawl_scrape

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| url | string | 是 | - | 网页地址 |
| output_format | string | 否 | markdown | 输出格式 |

### firecrawl_crawl

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| url | string | 是 | - | 网站根URL |
| limit | integer | 否 | 10 | 爬取页面数量限制 |

### firecrawl_search

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| query | string | 是 | - | 搜索关键词 |
| limit | integer | 否 | 5 | 返回结果数量 |

### Output Formats

| Format | Description |
|--------|-------------|
| markdown | Markdown格式 |
| json | JSON格式 |
| html | HTML格式 |

## Returns

### firecrawl_scrape
```json
{
  "success": true,
  "url": "https://example.com",
  "format": "markdown",
  "content": "...",
  "title": "页面标题",
  "generated_at": "2026-04-16T10:30:00"
}
```

### firecrawl_crawl
```json
{
  "success": true,
  "url": "https://example.com",
  "pages_crawled": 10,
  "results": [...],
  "generated_at": "2026-04-16T10:30:00"
}
```

### firecrawl_search
```json
{
  "success": true,
  "query": "人工智能",
  "results": [...],
  "generated_at": "2026-04-16T10:30:00"
}
```

## Usage Example

### 抓取单个网页
```json
{
  "name": "firecrawl_scrape",
  "parameters": {
    "url": "https://example.com",
    "output_format": "markdown"
  }
}
```

### 爬取网站
```json
{
  "name": "firecrawl_crawl",
  "parameters": {
    "url": "https://example.com",
    "limit": 10
  }
}
```

### 网页搜索
```json
{
  "name": "firecrawl_search",
  "parameters": {
    "query": "人工智能最新趋势",
    "limit": 5
  }
}
```

## Use Cases
- 获取动态网页（SPA应用）的完整内容
- 批量获取网站多个页面内容
- 提取网页中的结构化信息
- 获取最新的新闻和资讯
- 补充知识库内容

## Notes
- 需要配置 Firecrawl API Key
- 支持 JavaScript 渲染页面
- 请遵守目标网站的 robots.txt 规则
- 建议对爬取频率进行限制