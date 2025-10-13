"""
模板模組 - 統一管理報告模板
"""
from .template_manager import TemplateManager

# 導出主要類別
__all__ = ["TemplateManager"]

# 便利函數


def get_template_manager() -> TemplateManager:
    """獲取模板管理器實例"""
    return TemplateManager()


def get_available_templates() -> list:
    """獲取所有可用的模板名稱"""
    manager = TemplateManager()
    return manager.get_available_templates()


def get_template(template_name: str) -> dict:
    """獲取指定模板"""
    manager = TemplateManager()
    return manager.get_template(template_name)


def recommend_template(topic: str, audience: str = "", purpose: str = "") -> str:
    """根據主題、受眾和用途推薦模板"""
    manager = TemplateManager()
    return manager.recommend_template(topic, audience, purpose)
