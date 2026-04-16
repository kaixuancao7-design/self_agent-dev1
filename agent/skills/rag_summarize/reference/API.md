# RAG Summarize API Reference

## Function Signature
```python
def rag_summarize(query: str) -> str:
    """
    检索知识库中的相关文档并返回格式化内容
    
    Args:
        query: 用户查询字符串
        
    Returns:
        格式化的参考文档内容字符串
    """
```

## Implementation Details

### 检索流程
1. 接收用户查询
2. 使用向量数据库进行语义检索
3. 获取Top-K相关文档
4. 格式化文档内容
5. 返回检索结果

### 向量数据库配置
- **数据库类型**: Chroma
- **检索策略**: 混合检索（向量 + BM25）
- **返回数量**: 3条（可配置）

### 输出格式
```
参考资料1：{文档内容}
参考资料2：{文档内容}
参考资料3：{文档内容}
```

## Error Handling

| 错误类型 | 错误信息 | 处理方式 |
|----------|----------|----------|
| 无相关文档 | "未检索到相关资料" | 返回空结果提示 |
| 数据库连接失败 | 异常日志 | 抛出异常 |
| 查询格式错误 | 异常日志 | 抛出异常 |

## Dependencies
- langchain_core
- chromadb
- rag_service

## Performance
- 检索响应时间: < 1秒
- 支持并发查询
- 自动重试机制
