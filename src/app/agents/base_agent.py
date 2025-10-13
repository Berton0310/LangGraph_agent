# """
# Agent 基類和管理器
# 支援多個 Agent 的統一管理和配置
# """
# from abc import ABC, abstractmethod
# from typing import Dict, Any, Optional, List, Type, Union
# from dataclasses import dataclass
# from enum import Enum

# from ..config import ModelName, get_llm, get_current_model, create_runnable_config, AgentType


# @dataclass
# class AgentConfig:
#     """Agent 配置"""
#     agent_type: AgentType
#     model_name: ModelName
#     temperature: float = 0.2
#     max_tokens: Optional[int] = None
#     custom_prompt: Optional[str] = None
#     enabled: bool = True


# class BaseAgent(ABC):
#     """Agent 基類"""

#     def __init__(self, agent_type: AgentType, model_name: Optional[ModelName] = None):
#         """
#         初始化 Agent

#         Args:
#             agent_type: Agent 類型
#             model_name: 模型名稱，如果為 None 則使用預設模型
#         """
#         self.agent_type = agent_type
#         self.model_name = model_name or get_current_model()
#         self.llm = get_llm(self.model_name)
#         self.config = create_runnable_config(self.model_name)

#         # 初始化 Agent 特定配置
#         self._initialize_agent()

#     @abstractmethod
#     def _initialize_agent(self):
#         """初始化 Agent 特定配置"""
#         pass

#     @abstractmethod
#     def execute(self, task: str, **kwargs) -> Dict[str, Any]:
#         """
#         執行 Agent 任務

#         Args:
#             task: 任務描述
#             **kwargs: 其他參數

#         Returns:
#             執行結果
#         """
#         pass

#     def get_info(self) -> Dict[str, Any]:
#         """獲取 Agent 信息"""
#         config_info = None
#         if hasattr(self.config, 'configurable'):
#             config_info = self.config.configurable
#         elif hasattr(self.config, 'config'):
#             config_info = self.config.config

#         return {
#             "agent_type": self.agent_type.value,
#             "model_name": self.model_name,
#             "llm_type": type(self.llm).__name__,
#             "config": config_info
#         }


# class AgentManager:
#     """Agent 管理器"""

#     def __init__(self):
#         self._agents: Dict[AgentType, BaseAgent] = {}
#         self._agent_configs: Dict[AgentType, AgentConfig] = {}
#         self._initialize_default_configs()

#     def _initialize_default_configs(self):
#         """初始化預設 Agent 配置"""
#         # Plan Agent 配置
#         self._agent_configs[AgentType.PLAN] = AgentConfig(
#             agent_type=AgentType.PLAN,
#             model_name="gemini-2.5-pro",
#             temperature=0.2,
#             custom_prompt="你是一個專業的報告規劃專家"
#         )

#         # Research Agent 配置
#         self._agent_configs[AgentType.RESEARCH] = AgentConfig(
#             agent_type=AgentType.RESEARCH,
#             model_name="gpt-4o-mini",
#             temperature=0.1,
#             custom_prompt="你是一個專業的研究分析師"
#         )

#         # Writer Agent 配置
#         self._agent_configs[AgentType.WRITER] = AgentConfig(
#             agent_type=AgentType.WRITER,
#             model_name="gpt-4o",
#             temperature=0.3,
#             custom_prompt="你是一個專業的報告撰寫專家"
#         )

#         # Reviewer Agent 配置
#         self._agent_configs[AgentType.REVIEWER] = AgentConfig(
#             agent_type=AgentType.REVIEWER,
#             model_name="claude-3-sonnet-20240229",
#             temperature=0.1,
#             custom_prompt="你是一個專業的報告審查專家"
#         )

#         # Analyzer Agent 配置
#         self._agent_configs[AgentType.ANALYZER] = AgentConfig(
#             agent_type=AgentType.ANALYZER,
#             model_name="claude-3-opus-20240229",
#             temperature=0.2,
#             custom_prompt="你是一個專業的數據分析專家"
#         )

#         # Outline Agent 配置
#         self._agent_configs[AgentType.OUTLINE] = AgentConfig(
#             agent_type=AgentType.OUTLINE,
#             model_name="gemini-2.5-pro",
#             temperature=0.3,
#             custom_prompt="你是一個專業的大綱生成專家"
#         )

#     def register_agent(self, agent: BaseAgent):
#         """註冊 Agent"""
#         self._agents[agent.agent_type] = agent

#     def get_agent(self, agent_type: AgentType) -> Optional[BaseAgent]:
#         """獲取 Agent"""
#         return self._agents.get(agent_type)

#     def create_agent(self, agent_type: AgentType, model_name: Optional[ModelName] = None) -> BaseAgent:
#         """創建 Agent"""
#         config = self._agent_configs.get(agent_type)
#         if not config:
#             raise ValueError(f"不支援的 Agent 類型: {agent_type}")

#         # 使用配置中的模型名稱或指定的模型名稱
#         used_model = model_name or config.model_name

#         # 根據 Agent 類型創建對應的 Agent
#         if agent_type == AgentType.PLAN:
#             from .plan_agent import PlanAgent
#             return PlanAgent(agent_type, used_model)
#         elif agent_type == AgentType.RESEARCH:
#             from .research_agent import ResearchAgent
#             return ResearchAgent(agent_type, used_model)
#         elif agent_type == AgentType.WRITER:
#             from .writer_agent import WriterAgent
#             return WriterAgent(agent_type, used_model)
#         elif agent_type == AgentType.REVIEWER:
#             from .reviewer_agent import ReviewerAgent
#             return ReviewerAgent(agent_type, used_model)
#         elif agent_type == AgentType.ANALYZER:
#             from .analyzer_agent import AnalyzerAgent
#             return AnalyzerAgent(agent_type, used_model)
#         elif agent_type == AgentType.OUTLINE:
#             from .outline_agent import OutlineAgent
#             return OutlineAgent(agent_type, used_model)
#         else:
#             raise ValueError(f"不支援的 Agent 類型: {agent_type}")

#     def get_or_create_agent(self, agent_type: AgentType, model_name: Optional[ModelName] = None) -> BaseAgent:
#         """獲取或創建 Agent"""
#         agent = self.get_agent(agent_type)
#         if agent is None:
#             agent = self.create_agent(agent_type, model_name)
#             self.register_agent(agent)
#         return agent

#     def update_agent_config(self, agent_type: AgentType, **kwargs):
#         """更新 Agent 配置"""
#         if agent_type in self._agent_configs:
#             config = self._agent_configs[agent_type]
#             for key, value in kwargs.items():
#                 if hasattr(config, key):
#                     setattr(config, key, value)

#     def get_agent_config(self, agent_type: AgentType) -> Optional[AgentConfig]:
#         """獲取 Agent 配置"""
#         return self._agent_configs.get(agent_type)

#     def list_agents(self) -> List[AgentType]:
#         """列出所有已註冊的 Agent"""
#         return list(self._agents.keys())

#     def list_available_agent_types(self) -> List[AgentType]:
#         """列出所有可用的 Agent 類型"""
#         return list(self._agent_configs.keys())

#     def get_agent_info(self, agent_type: AgentType) -> Optional[Dict[str, Any]]:
#         """獲取 Agent 信息"""
#         agent = self.get_agent(agent_type)
#         if agent:
#             return agent.get_info()
#         return None


# # 全域 Agent 管理器實例
# agent_manager = AgentManager()


# # 便利函數
# def get_agent(agent_type: AgentType, model_name: Optional[ModelName] = None) -> BaseAgent:
#     """獲取 Agent"""
#     return agent_manager.get_or_create_agent(agent_type, model_name)


# def create_agent(agent_type: AgentType, model_name: Optional[ModelName] = None) -> BaseAgent:
#     """創建 Agent"""
#     return agent_manager.create_agent(agent_type, model_name)


# def register_agent(agent: BaseAgent):
#     """註冊 Agent"""
#     agent_manager.register_agent(agent)


# def update_agent_config(agent_type: AgentType, **kwargs):
#     """更新 Agent 配置"""
#     agent_manager.update_agent_config(agent_type, **kwargs)


# def list_agents() -> List[AgentType]:
#     """列出所有已註冊的 Agent"""
#     return agent_manager.list_agents()


# def list_available_agent_types() -> List[AgentType]:
#     """列出所有可用的 Agent 類型"""
#     return agent_manager.list_available_agent_types()


# if __name__ == "__main__":
#     # 測試 Agent 管理器
#     print("🧪 測試 Agent 管理器")
#     print("="*30)

#     # 列出可用的 Agent 類型
#     available_types = list_available_agent_types()
#     print(f"📋 可用的 Agent 類型: {[t.value for t in available_types]}")

#     # 測試創建 Plan Agent
#     try:
#         plan_agent = get_agent(AgentType.PLAN)
#         print(f"✅ Plan Agent 創建成功: {plan_agent.get_info()}")
#     except Exception as e:
#         print(f"❌ Plan Agent 創建失敗: {e}")

#     # 測試 Agent 配置
#     plan_config = agent_manager.get_agent_config(AgentType.PLAN)
#     if plan_config:
#         print(f"📊 Plan Agent 配置: {plan_config}")
