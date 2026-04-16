"""
Skills 模块 - 按照 Anthropic Skills 标准设计

每个 Skill 包含:
- SKILL.md: 使用说明文档
- reference/: 详细参考文档
- script/: 可执行脚本

使用方式:
    from agent.skills import skill_loader
    tools = skill_loader.get_tool_list()
"""
from .skill_loader import SkillLoader, skill_loader

# 导出技能加载器
__all__ = ["SkillLoader", "skill_loader"]
