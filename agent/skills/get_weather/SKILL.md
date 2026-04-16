# Get Weather Skill

## Overview
获取指定城市的天气信息。

## Capability
- **核心能力**: 查询天气信息
- **输入**: 城市名称
- **输出**: 天气信息字符串

## Parameters

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| city | string | 是 | 城市名称 |

## Returns
```
{城市}的天气是{天气}，温度{温度}度，湿度{湿度}%，风速{风速}公里每小时。
```

## Usage Example
```json
{
  "name": "get_weather",
  "parameters": {
    "city": "北京"
  }
}
```

## Use Cases
- 查询当前天气状况
- 出行前了解天气
- 为用户提供天气建议

## Notes
- 支持国内主要城市
- 后续可接入真实天气API
