"""
Configuration management for the Open Deep Research system.
基於 Pydantic 的統一配置管理系統
"""
import os
from enum import Enum
from typing import Any, List, Optional, Dict, Literal, Union
from dataclasses import dataclass, fields

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# 抑制 Google 庫的警告訊息
os.environ["GOOGLE_API_USE_CLIENT_CERTIFICATE"] = "false"

# 優先從 .env 載入（若已由部署環境注入，dotenv 不會覆蓋現有環境變數）
load_dotenv(override=False)

# -- 模型類型定義
ModelProvider = Literal["gemini", "openai", "anthropic", "azure", "ollama"]
ModelName = Literal[
    "gemini-2.5-pro",
    "gpt-4o-mini",
    "gpt-4o",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-3-opus-20240229"
]

# Agent 類型枚舉


class AgentType(Enum):
    """Agent 類型枚舉"""
    PLAN = "plan"
    RESEARCH = "research"
    WRITER = "writer"
    REVIEWER = "reviewer"
    ANALYZER = "analyzer"
    OUTLINE = "outline"

# 搜尋 API 枚舉


class SearchAPI(Enum):
    """搜尋 API 類型"""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    TAVILY = "tavily"
    NONE = "none"

# MCP 配置


class MCPConfig(BaseModel):
    """Configuration for Model Context Protocol (MCP) servers."""

    url: Optional[str] = Field(
        default=None,
        optional=True,
    )
    """The URL of the MCP server"""
    tools: Optional[List[str]] = Field(
        default=None,
        optional=True,
    )
    """The tools to make available to the LLM"""
    auth_required: Optional[bool] = Field(
        default=False,
        optional=True,
    )
    """Whether the MCP server requires authentication"""

# 主要配置類別


class Configuration(BaseModel):
    """Main configuration class for the Deep Research agent."""

    # General Configuration
    max_structured_output_retries: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "Maximum number of retries for structured output calls from models"
            }
        }
    )
    allow_clarification: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Whether to allow the researcher to ask the user clarifying questions before starting research"
            }
        }
    )
    max_concurrent_research_units: int = Field(
        default=5,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "Maximum number of research units to run concurrently. This will allow the researcher to use multiple sub-agents to conduct research. Note: with more concurrency, you may run into rate limits."
            }
        }
    )
    # Research Configuration
    search_api: SearchAPI = Field(
        default=SearchAPI.TAVILY,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "tavily",
                "description": "Search API to use for research. NOTE: Make sure your Researcher Model supports the selected search API.",
                "options": [
                    {"label": "Tavily", "value": SearchAPI.TAVILY.value},
                    {"label": "OpenAI Native Web Search",
                        "value": SearchAPI.OPENAI.value},
                    {"label": "Anthropic Native Web Search",
                        "value": SearchAPI.ANTHROPIC.value},
                    {"label": "None", "value": SearchAPI.NONE.value}
                ]
            }
        }
    )
    max_researcher_iterations: int = Field(
        default=6,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 6,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Maximum number of research iterations for the Research Supervisor. This is the number of times the Research Supervisor will reflect on the research and ask follow-up questions."
            }
        }
    )
    max_react_tool_calls: int = Field(
        default=2,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 10,
                "min": 1,
                "max": 30,
                "step": 1,
                "description": "Maximum number of tool calling iterations to make in a single researcher step."
            }
        }
    )
    # Model Configuration
    summarization_model: str = Field(
        default="openai:gpt-4o-mini",  # 原始: "openai:gpt-4o-mini"
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4o-mini",  # 原始: "openai:gpt-4o-mini"
                "description": "Model for summarizing research results from Tavily search results"
            }
        }
    )
    summarization_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for summarization model"
            }
        }
    )
    max_content_length: int = Field(
        default=50000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 50000,
                "min": 1000,
                "max": 200000,
                "description": "Maximum character length for webpage content before summarization"
            }
        }
    )
    research_model: str = Field(
        default="gemini-2.5-pro",  # 原始: "openai:gpt-4o"
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "gemini-2.5-pro",  # 原始: "openai:gpt-4o"
                "description": "Model for conducting research. NOTE: Make sure your Researcher Model supports the selected search API."
            }
        }
    )
    research_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for research model"
            }
        }
    )
    compression_model: str = Field(
        default="gemini-2.5-pro",  # 原始: "openai:gpt-4o"
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "gemini-2.5-pro",  # 原始: "openai:gpt-4o"
                "description": "Model for compressing research findings from sub-agents. NOTE: Make sure your Compression Model supports the selected search API."
            }
        }
    )
    compression_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for compression model"
            }
        }
    )
    final_report_model: str = Field(
        default="gemini-2.5-pro",  # 原始: "openai:gpt-4o"
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "gemini-2.5-pro",  # 原始: "openai:gpt-4o"
                "description": "Model for writing the final report from all research findings"
            }
        }
    )
    final_report_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for final report model"
            }
        }
    )
    # Plan Agent 特定配置
    plan_model: str = Field(
        default="gemini-2.5-pro",  # 原始: "gemini:gemini-2.5-pro"
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "gemini-2.5-pro",  # 原始: "gemini:gemini-2.5-pro"
                "description": "Model for planning report structure and sections"
            }
        }
    )
    plan_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for plan model"
            }
        }
    )
    # MCP server configuration
    mcp_config: Optional[MCPConfig] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "mcp",
                "description": "MCP server configuration"
            }
        }
    )
    mcp_prompt: Optional[str] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "Any additional instructions to pass along to the Agent regarding the MCP tools that are available to it."
            }
        }
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(
                field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True


# 模型配置類別
@dataclass(kw_only=True)
class ModelConfiguration:
    """模型配置"""
    provider: ModelProvider
    model_name: ModelName
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    enabled: bool = True


# Agent 配置類別
@dataclass(kw_only=True)
class AgentConfiguration:
    """Agent 配置"""
    agent_type: AgentType
    model_name: ModelName
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    custom_prompt: Optional[str] = None
    enabled: bool = True
    max_retries: int = 3


class ConfigurationManager:
    """配置管理器"""

    def __init__(self):
        self._model_configs: Dict[str, ModelConfiguration] = {}
        self._agent_configs: Dict[AgentType, AgentConfiguration] = {}
        self._default_model = "gemini-2.5-pro"
        self._initialize_configurations()

    def _initialize_configurations(self):
        """初始化所有配置"""

        # 初始化模型配置
        self._model_configs = {
            "gemini-2.5-pro": ModelConfiguration(
                provider="gemini",
                model_name="gemini-2.5-pro",
                temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.2")),
                api_key=os.getenv("GEMINI_API_KEY")
            ),
            "gpt-4o-mini": ModelConfiguration(
                provider="openai",
                model_name="gpt-4o-mini",
                temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
                api_key=os.getenv("OPENAI_API_KEY")
            ),
            "gpt-4o": ModelConfiguration(
                provider="openai",
                model_name="gpt-4o",
                temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
                api_key=os.getenv("OPENAI_API_KEY")
            ),
            "claude-3-sonnet-20240229": ModelConfiguration(
                provider="anthropic",
                model_name="claude-3-sonnet-20240229",
                temperature=float(os.getenv("ANTHROPIC_TEMPERATURE", "0.2")),
                api_key=os.getenv("ANTHROPIC_API_KEY")
            ),
            "claude-3-haiku-20240307": ModelConfiguration(
                provider="anthropic",
                model_name="claude-3-haiku-20240307",
                temperature=float(os.getenv("ANTHROPIC_TEMPERATURE", "0.2")),
                api_key=os.getenv("ANTHROPIC_API_KEY")
            ),
            "claude-3-opus-20240229": ModelConfiguration(
                provider="anthropic",
                model_name="claude-3-opus-20240229",
                temperature=float(os.getenv("ANTHROPIC_TEMPERATURE", "0.2")),
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        }

        # 初始化 Agent 配置
        self._agent_configs = {
            AgentType.PLAN: AgentConfiguration(
                agent_type=AgentType.PLAN,
                model_name="gemini-2.5-pro",
                temperature=0.2,
                custom_prompt="你是一個專業的報告規劃專家"
            ),
            AgentType.RESEARCH: AgentConfiguration(
                agent_type=AgentType.RESEARCH,
                model_name="gemini-2.5-pro",  # 原始: "gpt-4o-mini"
                temperature=0.1,
                custom_prompt="你是一個專業的研究分析師"
            ),
            AgentType.WRITER: AgentConfiguration(
                agent_type=AgentType.WRITER,
                model_name="gpt-4o",  # 原始: "gpt-4o"
                temperature=0.3,
                custom_prompt="你是一個專業的報告撰寫專家"
            ),
            AgentType.REVIEWER: AgentConfiguration(
                agent_type=AgentType.REVIEWER,
                model_name="gemini-2.5-pro",  # 原始: "claude-3-sonnet-20240229"
                temperature=0.1,
                custom_prompt="你是一個專業的報告審查專家"
            ),
            AgentType.ANALYZER: AgentConfiguration(
                agent_type=AgentType.ANALYZER,
                model_name="gemini-2.5-pro",  # 原始: "claude-3-opus-20240229"
                temperature=0.2,
                custom_prompt="你是一個專業的數據分析專家"
            ),
            AgentType.OUTLINE: AgentConfiguration(
                agent_type=AgentType.OUTLINE,
                model_name="gemini-2.5-pro",
                temperature=0.3,
                custom_prompt="你是一個專業的大綱生成專家"
            )
        }

    def get_model_config(self, model_name: Optional[ModelName] = None) -> ModelConfiguration:
        """獲取模型配置"""
        if model_name is None:
            model_name = self._default_model

        if model_name not in self._model_configs:
            raise ValueError(f"不支援的模型: {model_name}")

        config = self._model_configs[model_name]

        # 檢查 API 金鑰
        if not config.api_key:
            raise RuntimeError(f"{config.provider.upper()}_API_KEY 未設定")

        return config

    def get_agent_config(self, agent_type: AgentType) -> Optional[AgentConfiguration]:
        """獲取 Agent 配置"""
        return self._agent_configs.get(agent_type)

    def set_default_model(self, model_name: ModelName):
        """設定預設模型"""
        if model_name not in self._model_configs:
            raise ValueError(f"不支援的模型: {model_name}")
        self._default_model = model_name

    def get_available_models(self) -> Dict[str, ModelConfiguration]:
        """獲取所有可用模型"""
        return self._model_configs.copy()

    def get_available_agents(self) -> Dict[AgentType, AgentConfiguration]:
        """獲取所有可用 Agent"""
        return self._agent_configs.copy()


# 全域配置管理器實例
config_manager = ConfigurationManager()


# 主要配置函數
def get_model_config(model_name: Optional[ModelName] = None) -> ModelConfiguration:
    """獲取模型配置"""
    return config_manager.get_model_config(model_name)


def get_agent_config(agent_type: AgentType) -> Optional[AgentConfiguration]:
    """獲取 Agent 配置"""
    return config_manager.get_agent_config(agent_type)


def get_llm(model_name: Optional[ModelName] = None):
    """根據模型名稱獲取 LLM 實例"""
    config = get_model_config(model_name)

    if config.provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=config.api_key
        )
    elif config.provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            openai_api_key=config.api_key
        )
    elif config.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            anthropic_api_key=config.api_key
        )
    else:
        raise ValueError(f"不支援的提供者: {config.provider}")


def get_api_key_for_model(model_name: str, config: Optional[RunnableConfig] = None) -> Optional[str]:
    """根據模型名稱獲取對應的 API 金鑰"""
    # 從模型名稱中提取提供者
    if ":" in model_name:
        provider, model = model_name.split(":", 1)
    else:
        # 根據模型名稱判斷提供者
        if model_name.startswith("gemini"):
            provider = "gemini"
        elif model_name.startswith("gpt"):
            provider = "openai"
        elif model_name.startswith("claude"):
            provider = "anthropic"
        else:
            provider = model_name.split("-")[0]  # 預設行為

    # 根據提供者獲取 API 金鑰
    if provider.lower() == "gemini":
        return os.getenv("GEMINI_API_KEY")
    elif provider.lower() == "openai":
        return os.getenv("OPENAI_API_KEY")
    elif provider.lower() == "anthropic":
        return os.getenv("ANTHROPIC_API_KEY")
    elif provider.lower() == "claude":
        return os.getenv("ANTHROPIC_API_KEY")
    else:
        # 嘗試從配置中獲取
        if config and "configurable" in config:
            return config["configurable"].get("api_key")
        return None


def get_llm_with_structured_output(model_name: Optional[ModelName] = None, model_class=None):
    """獲取支援 structured output 的 LLM 實例"""
    llm = get_llm(model_name)

    # 支援 structured output 的模型
    structured_output_models = ["gpt-4o-mini", "gpt-4o", "claude-3-sonnet-20240229",
                                "claude-3-haiku-20240307", "claude-3-opus-20240229", "gemini-2.5-pro"]

    config = get_model_config(model_name)

    if config.model_name in structured_output_models and model_class:
        try:
            return llm.with_structured_output(model_class)
        except AttributeError:
            # 如果模型不支援 with_structured_output，回退到原始 LLM
            pass

    return llm


def create_runnable_config(model_name: Optional[ModelName] = None, **kwargs) -> RunnableConfig:
    """創建 RunnableConfig"""
    config = get_model_config(model_name)

    runnable_config = RunnableConfig(
        configurable={
            "model_name": config.model_name,
            "provider": config.provider,
            "temperature": config.temperature,
            **kwargs
        }
    )

    return runnable_config


def get_current_model() -> ModelName:
    """獲取當前預設模型"""
    return config_manager._default_model


def set_current_model(model_name: ModelName):
    """設定當前預設模型"""
    config_manager.set_default_model(model_name)


def get_available_models() -> Dict[str, ModelConfiguration]:
    """獲取所有可用模型"""
    return config_manager.get_available_models()


def get_available_agents() -> Dict[AgentType, AgentConfiguration]:
    """獲取所有可用 Agent"""
    return config_manager.get_available_agents()


# Agent 配置管理
def get_agent_model_config(agent_type: str) -> ModelName:
    """獲取特定 Agent 的模型配置"""
    # 從環境變數獲取 Agent 特定配置
    env_key = f"{agent_type.upper()}_MODEL"
    model_name = os.getenv(env_key)

    if model_name and model_name in config_manager._model_configs:
        return model_name  # type: ignore

    # 回退到預設模型
    return get_current_model()


def set_agent_model_config(agent_type: str, model_name: ModelName):
    """設定特定 Agent 的模型配置"""
    env_key = f"{agent_type.upper()}_MODEL"
    os.environ[env_key] = model_name


def get_agent_temperature_config(agent_type: str) -> float:
    """獲取特定 Agent 的溫度配置"""
    env_key = f"{agent_type.upper()}_TEMPERATURE"
    return float(os.getenv(env_key, "0.2"))


def set_agent_temperature_config(agent_type: str, temperature: float):
    """設定特定 Agent 的溫度配置"""
    env_key = f"{agent_type.upper()}_TEMPERATURE"
    os.environ[env_key] = str(temperature)


# 便利函數
def get_gemini_llm() -> Any:
    """獲取 Gemini LLM"""
    return get_llm("gemini-2.5-pro")


def get_openai_llm(model: Literal["gpt-4o-mini", "gpt-4o"] = "gpt-4o-mini") -> Any:
    """獲取 OpenAI LLM"""
    return get_llm(model)


def get_claude_llm(model: Literal["claude-3-sonnet-20240229", "claude-3-haiku-20240307", "claude-3-opus-20240229"] = "claude-3-sonnet-20240229") -> Any:
    """獲取 Claude LLM"""
    return get_llm(model)


if __name__ == "__main__":
    # 測試配置
    print("🧪 測試新的配置系統")
    print("="*30)

    # 顯示可用模型
    models = get_available_models()
    print("📋 可用模型:")
    for name, config in models.items():
        print(f"  - {name} ({config.provider})")

    # 顯示可用 Agent
    agents = get_available_agents()
    print(f"\n📋 可用 Agent:")
    for agent_type, config in agents.items():
        print(f"  - {agent_type.value}: {config.model_name}")

    # 測試預設模型
    try:
        default_model = get_current_model()
        print(f"\n📊 預設模型: {default_model}")

        llm = get_llm()
        print(f"✅ LLM 創建成功: {type(llm).__name__}")

    except Exception as e:
        print(f"❌ LLM 創建失敗: {e}")

    # 測試主要配置
    try:
        main_config = Configuration.from_runnable_config()
        print(f"\n⚙️ 主要配置:")
        print(f"   搜尋 API: {main_config.search_api.value}")
        print(f"   研究模型: {main_config.research_model}")
        print(f"   規劃模型: {main_config.plan_model}")
        print(f"   最大重試次數: {main_config.max_structured_output_retries}")

    except Exception as e:
        print(f"❌ 主要配置失敗: {e}")
