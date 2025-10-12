"""
Agents 模組 - 統一導入所有代理類別
"""
from .base_agent import (
    BaseAgent,
    AgentConfig,
    AgentManager,
    agent_manager,
    get_agent,
    create_agent,
    register_agent,
    update_agent_config,
    list_agents,
    list_available_agent_types
)

from .research_agent import (
    clarify_with_user,
    write_research_brief,
    test_research_agent
)

# 導出所有代理類別
__all__ = [
    # 基類和管理器
    "BaseAgent",
    "AgentConfig",
    "AgentManager",
    "agent_manager",

    # 研究代理函數
    "clarify_with_user",
    "write_research_brief",
    "test_research_agent",

    # 便利函數
    "get_agent",
    "create_agent",
    "register_agent",
    "update_agent_config",
    "list_agents",
    "list_available_agent_types"
]
