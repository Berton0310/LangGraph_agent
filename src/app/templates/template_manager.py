"""
模板管理工具
用於讀取、管理和應用報告模板
"""
import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path


class TemplateManager:
    """模板管理器"""

    def __init__(self, templates_file: str = "src/app/templates/templates.json"):
        """
        初始化模板管理器

        Args:
            templates_file: 統一模板檔案路徑
        """
        self.templates_file = Path(templates_file)
        self.templates = {}
        self.template_system = None
        self._load_templates()

    def _load_templates(self):
        """載入統一模板檔案"""
        try:
            if self.templates_file.exists():
                with open(self.templates_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 載入模板系統資訊
                self.template_system = data.get("template_system", {})

                # 載入模板
                self.templates = data.get("templates", {})

                # 載入其他配置
                self.selection_rules = data.get("template_selection", {})
                self.customization = data.get("customization", {})
                self.integration = data.get("integration", {})
                self.examples = data.get("examples", {})

                print(f"✅ 成功載入 {len(self.templates)} 個模板")
                print(f"📊 模板系統版本: {self.template_system.get('version', '未知')}")

            else:
                print(f"❌ 模板檔案不存在: {self.templates_file}")

        except Exception as e:
            print(f"❌ 載入模板失敗: {e}")

    def get_available_templates(self) -> List[str]:
        """獲取所有可用的模板名稱"""
        return list(self.templates.keys())

    def get_template(self, template_name: str) -> Optional[Dict[str, Any]]:
        """獲取指定模板"""
        return self.templates.get(template_name)

    def get_template_structure(self, template_name: str) -> List[str]:
        """獲取模板結構"""
        template = self.get_template(template_name)
        if template:
            return template.get("structure", [])
        return []

    def get_template_style(self, template_name: str) -> Dict[str, Any]:
        """獲取模板風格"""
        template = self.get_template(template_name)
        if template:
            return template.get("style", {})
        return {}

    def get_template_purpose(self, template_name: str) -> str:
        """獲取模板用途"""
        template = self.get_template(template_name)
        if template:
            return template.get("purpose", "未指定用途")
        return "模板不存在"

    def get_template_target_audience(self, template_name: str) -> str:
        """獲取目標受眾"""
        template = self.get_template(template_name)
        if template:
            return template.get("target_audience", "未指定受眾")
        return "模板不存在"

    def get_template_use_cases(self, template_name: str) -> List[str]:
        """獲取使用案例"""
        template = self.get_template(template_name)
        if template:
            return template.get("use_cases", [])
        return []

    def get_template_example_topics(self, template_name: str) -> List[str]:
        """獲取範例主題"""
        template = self.get_template(template_name)
        if template:
            return template.get("example_topics", [])
        return []

    def recommend_template(self, topic: str, audience: str = "", purpose: str = "") -> str:
        """根據主題、受眾和用途推薦模板"""
        recommendations = []

        for template_name, template in self.templates.items():
            score = 0

            # 根據受眾匹配
            if audience and audience.lower() in template.get("target_audience", "").lower():
                score += 2

            # 根據用途匹配
            if purpose and purpose.lower() in template.get("purpose", "").lower():
                score += 2

            # 根據範例主題匹配
            example_topics = template.get("example_topics", [])
            for example in example_topics:
                if any(word in topic.lower() for word in example.lower().split()):
                    score += 1

            if score > 0:
                recommendations.append((template_name, score))

        # 按分數排序
        recommendations.sort(key=lambda x: x[1], reverse=True)

        if recommendations:
            return recommendations[0][0]
        else:
            return "academic"  # 預設推薦學術模板

    def generate_custom_structure(self, template_name: str, topic: str) -> List[str]:
        """根據模板和主題生成自定義結構"""
        template = self.get_template(template_name)
        if not template:
            return []

        base_structure = template.get("structure", [])
        custom_structure = []

        for section in base_structure:
            # 根據主題自定義章節標題
            if "摘要" in section or "Abstract" in section:
                custom_structure.append(f"{section}: {topic}的概述")
            elif "背景" in section or "Background" in section:
                custom_structure.append(f"{section}: {topic}的發展背景")
            elif "方法" in section or "Methodology" in section:
                custom_structure.append(f"{section}: {topic}的研究方法")
            elif "結果" in section or "Results" in section:
                custom_structure.append(f"{section}: {topic}的分析結果")
            elif "結論" in section or "Conclusion" in section:
                custom_structure.append(f"{section}: {topic}的結論與建議")
            else:
                custom_structure.append(section)

        return custom_structure

    def get_template_system_info(self) -> Dict[str, Any]:
        """獲取模板系統資訊"""
        return self.template_system or {}

    def get_selection_rules(self) -> Dict[str, Any]:
        """獲取模板選擇規則"""
        return self.selection_rules or {}

    def get_customization_options(self) -> Dict[str, Any]:
        """獲取自定義選項"""
        return self.customization or {}

    def get_integration_info(self) -> Dict[str, Any]:
        """獲取整合資訊"""
        return self.integration or {}

    def get_examples(self) -> Dict[str, Any]:
        """獲取使用範例"""
        return self.examples or {}

    def get_template_metadata(self, template_name: str) -> Dict[str, Any]:
        """獲取模板元數據"""
        template = self.get_template(template_name)
        if template:
            return template.get("metadata", {})
        return {}

    def print_template_info(self, template_name: str):
        """列印模板詳細資訊"""
        template = self.get_template(template_name)
        if not template:
            print(f"❌ 模板 '{template_name}' 不存在")
            return

        print(f"\n📋 模板資訊: {template_name}")
        print("="*50)
        print(f"類別: {template.get('category', '未指定')}")
        print(f"用途: {template.get('purpose', '未指定')}")
        print(f"目標受眾: {template.get('target_audience', '未指定')}")

        print(f"\n📑 結構:")
        for i, section in enumerate(template.get('structure', []), 1):
            print(f"  {i}. {section}")

        print(f"\n🎨 風格:")
        style = template.get('style', {})
        for key, value in style.items():
            print(f"  {key}: {value}")

        print(f"\n✨ 主要特色:")
        for feature in template.get('key_features', []):
            print(f"  • {feature}")

        print(f"\n📝 使用案例:")
        for use_case in template.get('use_cases', []):
            print(f"  • {use_case}")

        print(f"\n💡 範例主題:")
        for topic in template.get('example_topics', []):
            print(f"  • {topic}")

    def print_all_templates(self):
        """列印所有模板的概覽"""
        if not self.templates:
            print("❌ 沒有可用的模板")
            return

        print("\n📚 可用模板概覽")
        print("="*50)

        for template_name, template in self.templates.items():
            print(f"\n📋 {template_name.upper()}")
            print(f"類別: {template.get('category', '未指定')}")
            print(f"用途: {template.get('purpose', '未指定')}")
            print(f"結構章節數: {len(template.get('structure', []))}")

    def export_template_config(self, output_file: str = "template_config.json"):
        """匯出模板配置"""
        config = {
            "template_system": self.template_system,
            "templates": self.templates,
            "template_selection": self.selection_rules,
            "customization": self.customization,
            "integration": self.integration,
            "examples": self.examples,
            "metadata": {
                "exported_at": "2024-12-19",
                "total_templates": len(self.templates),
                "template_names": list(self.templates.keys()),
                "system_version": self.template_system.get("version", "未知")
            }
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        print(f"✅ 模板配置已匯出至: {output_file}")


# 使用範例
def main():
    """主函數 - 示範模板管理器的使用"""
    print("🚀 模板管理器示範")
    print("="*50)

    # 創建模板管理器
    manager = TemplateManager()

    # 顯示所有模板
    manager.print_all_templates()

    # 顯示特定模板詳細資訊
    print("\n" + "="*50)
    manager.print_template_info("academic")

    print("\n" + "="*50)
    manager.print_template_info("business")

    # 模板推薦示範
    print("\n" + "="*50)
    print("🎯 模板推薦示範")

    topics = [
        "AI技術在教育領域的應用研究",
        "AI技術市場趨勢分析",
        "機器學習算法優化",
        "數位轉型策略建議"
    ]

    for topic in topics:
        recommended = manager.recommend_template(topic)
        print(f"主題: {topic}")
        print(f"推薦模板: {recommended}")
        print(f"用途: {manager.get_template_purpose(recommended)}")
        print()

    # 生成自定義結構示範
    print("="*50)
    print("🔧 自定義結構生成示範")

    topic = "AI技術在教育領域的應用研究"
    custom_structure = manager.generate_custom_structure("academic", topic)

    print(f"主題: {topic}")
    print("自定義結構:")
    for i, section in enumerate(custom_structure, 1):
        print(f"  {i}. {section}")

    # 匯出配置
    print("\n" + "="*50)
    manager.export_template_config()


if __name__ == "__main__":
    main()
