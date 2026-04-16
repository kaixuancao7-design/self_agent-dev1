# Knowledge Base Management Skill

## Overview
提供知识库管理的通用操作能力。

## Capability
- **核心能力**: 知识库管理
- **输入**: 操作类型和参数
- **输出**: 操作结果

## Parameters

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| action | string | 是 | 操作类型 |
| db_name | string | 否 | 数据库名称 |
| md5 | string | 否 | 文件MD5值 |
| file_path | string | 否 | 文件路径 |
| file_name | string | 否 | 文件名 |

### Action Types

| Action | Description | Required Parameters |
|--------|-------------|-------------------|
| get_stats | 获取知识库统计 | 无 |
| list_dbs | 获取数据库列表 | 无 |
| create_db | 创建数据库 | db_name |
| delete_db | 删除数据库 | db_name |
| switch_db | 切换数据库 | db_name |
| list_files | 获取文件列表 | 无 |
| remove_file | 删除文件 | md5 |
| reparse_file | 重新解析文件 | file_path, file_name |

## Returns
```json
{
  "success": true/false,
  "message": "操作结果描述",
  "data": {...}
}
```

## Usage Example
```json
{
  "name": "kb_management",
  "parameters": {
    "action": "create_db",
    "db_name": "my_knowledge_base"
  }
}
```

## Use Cases
- 管理知识库数据库
- 查询知识库统计信息
- 管理上传的文件
- 维护知识库内容

## Notes
- 删除操作需要谨慎
- 支持多数据库管理
- 提供完整的CRUD操作
