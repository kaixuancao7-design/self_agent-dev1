# Fact Check Skill

## Overview
校验答案与检索到的原文是否一致，避免"幻觉"问题。

## Capability
- **核心能力**: 事实核查与验证
- **输入**: 答案文本、参考来源
- **输出**: 核查结果与置信度

## Parameters

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| answer | string | 是 | 需要核查的答案 |
| sources | string | 是 | 参考来源文本 |

## Returns
```json
{
  "result": "一致"/"不一致"/"部分一致",
  "confidence": 0.85,
  "explanation": "核查说明",
  "suggestion": "修改建议（如不一致）"
}
```

## Usage Example
```json
{
  "name": "fact_check",
  "parameters": {
    "answer": "AI Agent是2026年最热门的技术趋势",
    "sources": "参考资料1：根据2026年行业报告，AI Agent正在成为主流技术方向..."
  }
}
```

## Use Cases
- 验证AI回答的准确性
- 检查事实一致性
- 避免幻觉问题
- 提升回答可信度

## Notes
- 支持中文文本核查
- 返回置信度评分
- 提供修改建议
