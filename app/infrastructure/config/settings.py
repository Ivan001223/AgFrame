"""
Pydantic Settings 配置模块。

基于 Pydantic Settings 提供类型安全的配置管理，
支持自动环境变量映射和内置验证。
"""
import json
import logging
import os
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# 初始化环境变量
from app.infrastructure.config.env import init_env

init_env()


# ==================== 嵌套配置模型 ====================


class LLMConfig(BaseSettings):
    """LLM 配置"""
    api_key: str = Field(default="", alias="LLM_API_KEY")
    base_url: str = Field(default="https://api.openai.com/v1", alias="LLM_BASE_URL")
    model: str = Field(default="gpt-4o", alias="LLM_MODEL")
    structured_output_mode: str = "native_first"
    json_mode_response_format: bool = False


class ModelManagerConfig(BaseSettings):
    """模型管理器配置"""
    provider: str = Field(default="modelscope", alias="MODEL_MANAGER_PROVIDER")
    cache_dir: str = Field(default="", alias="MODEL_MANAGER_CACHE_DIR")
    revision: str = ""
    trust_remote_code: bool = True
    modelscope_fallback_to_hf: bool = True


class LocalModelsConfig(BaseSettings):
    """本地模型配置"""
    ocr_model: str = Field(default="", alias="MODEL_PATH_OCR")
    embedding_model: str = Field(default="", alias="MODEL_PATH_EMBEDDING")
    rerank_model: str = Field(default="", alias="MODEL_PATH_RERANKER")


class EmbeddingsConfig(BaseSettings):
    """嵌入模型配置"""
    provider: str = "modelscope"
    backend: str = "sentence_transformers"
    model_name: str = ""
    env_var: str = "MODEL_PATH_EMBEDDING"
    device: str = "auto"
    batch_size: int = 32
    max_length: int = 512
    pooling: str = "mean"
    normalize: bool = True
    query_prefix: str = ""
    doc_prefix: str = ""


class RerankerConfig(BaseSettings):
    """重排序模型配置"""
    provider: str = "modelscope"
    backend: str = "sentence_transformers"
    model_name: str = ""
    env_var: str = "MODEL_PATH_RERANKER"
    device: str = "auto"
    batch_size: int = 16
    max_length: int = 512
    query_prefix: str = ""
    doc_prefix: str = ""
    window_size: int | None = None
    stride: int | None = None
    transformers_model_type: str = "auto"


class SearchConfig(BaseSettings):
    """搜索配置"""
    provider: str = Field(default="duckduckgo", alias="SEARCH_PROVIDER")
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")
    serpapi_api_key: str = ""
    cache_ttl: int = 3600


class DatabaseConfig(BaseSettings):
    """数据库配置"""
    type: str = Field(default="postgres", alias="DB_TYPE")
    url: str = Field(default="", alias="DATABASE_URL")
    host: str = Field(default="localhost", alias="DB_HOST")
    port: int = Field(default=5432, alias="DB_PORT")
    user: str = Field(default="postgres", alias="DB_USER")
    password: str = Field(default="password", alias="DB_PASSWORD")
    db_name: str = Field(default="agent_app", alias="DB_NAME")


class QueueConfig(BaseSettings):
    """队列配置"""
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    rabbitmq_url: str = ""
    rabbitmq_management_url: str = ""


class StorageS3Config(BaseSettings):
    """S3 存储配置"""
    s3_endpoint: str = Field(default="", alias="S3_ENDPOINT")
    s3_access_key: str = Field(default="", alias="S3_ACCESS_KEY")
    s3_secret_key: str = Field(default="", alias="S3_SECRET_KEY")
    s3_bucket: str = Field(default="agframe", alias="S3_BUCKET")
    s3_secure: bool = False


class StorageLocalConfig(BaseSettings):
    """本地存储配置"""
    documents_dir: str = "data/documents"
    uploads_dir: str = "data/uploads"
    data_dir: str = "data"


class AuthConfig(BaseSettings):
    """认证配置"""
    secret_key: str = Field(default="your-secret-key-keep-it-secret", alias="AUTH_SECRET_KEY")
    algorithm: str = Field(default="HS256", alias="AUTH_ALGORITHM")
    access_token_expire_minutes: int = 30


class GeneralConfig(BaseSettings):
    """通用配置"""
    app_name: str = Field(default="My Agent App", alias="APP_NAME")


class RagRetrievalConfig(BaseSettings):
    """RAG 检索配置"""
    mode: str = "hybrid"
    dense_k: int = 20
    sparse_k: int = 20
    candidate_k: int = 20
    final_k: int = 3
    rrf_k: int = 60
    weights: list[float] = [0.5, 0.5]


class RagConfig(BaseSettings):
    """RAG 配置"""
    retrieval: RagRetrievalConfig = Field(default_factory=RagRetrievalConfig)


class PromptBudgetConfig(BaseSettings):
    """提示词预算配置"""
    max_recent_history_lines: int = 10
    max_docs: int = 3
    max_memories: int = 3
    max_doc_chars_total: int = 6000
    max_memory_chars_total: int = 3000
    max_item_chars: int = 2000


class PromptConfig(BaseSettings):
    """提示词配置"""
    budget: PromptBudgetConfig = Field(default_factory=PromptBudgetConfig)


class NodesConfig(BaseSettings):
    """节点配置"""
    enabled: list[str] = [
        "router",
        "retrieve_docs",
        "rerank_docs",
        "retrieve_memories",
        "assemble",
        "generate",
    ]


class FeatureFlagsConfig(BaseSettings):
    """特性开关配置"""
    enable_docs_rag: bool = Field(default=True, alias="ENABLE_DOCS_RAG")
    enable_chat_memory: bool = Field(default=True, alias="ENABLE_CHAT_MEMORY")
    enable_self_correction: bool = Field(default=True, alias="ENABLE_SELF_CORRECTION")
    enable_human_approval: bool = False
    allow_dangerous_deserialization: bool = False
    enable_tools_write_file: bool = False
    enable_tools_python_repl: bool = False
    enable_tools_python_executor: bool = False
    pgvector_dimension: int = 1024


class SandboxConfig(BaseSettings):
    """沙箱配置"""
    enabled: bool = False
    image: str = "python:3.11-slim"
    timeout: int = 30
    memory_limit: str = "256m"
    cpu_limit: float = 0.5


class SelfCorrectionConfig(BaseSettings):
    """自纠正配置"""
    max_attempts: int = 2


class ServerConfig(BaseSettings):
    """服务器配置"""
    host: str = Field(default="0.0.0.0", alias="SERVER_HOST")
    port: int = Field(default=8000, alias="SERVER_PORT")
    cors_origins: list[str] = Field(default_factory=lambda: ["*"], alias="CORS_ORIGINS")
    cors_allow_credentials: bool = True


# ==================== 主 Settings 类 ====================


class Settings(BaseSettings):
    """
    应用全局配置单例。

    使用 Pydantic BaseSettings 自动从环境变量加载配置，
    支持从 config.json 文件覆盖默认配置。

    环境变量优先级高于配置文件。

    用法:
        # 直接访问
        settings.llm.model
        settings.rag.retrieval.final_k

        # 快捷访问
        settings.get("llm.model")
        settings.get("database.host", "localhost")  # 带默认值
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 顶层配置项
    llm: LLMConfig = Field(default_factory=LLMConfig)
    model_manager: ModelManagerConfig = Field(default_factory=ModelManagerConfig)
    local_models: LocalModelsConfig = Field(default_factory=LocalModelsConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    storage_s3: StorageS3Config = Field(default_factory=StorageS3Config, alias="storage_s3")
    storage_local: StorageLocalConfig = Field(default_factory=StorageLocalConfig, alias="storage_local")
    auth: AuthConfig = Field(default_factory=AuthConfig)
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    rag: RagConfig = Field(default_factory=RagConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    nodes: NodesConfig = Field(default_factory=NodesConfig)
    feature_flags: FeatureFlagsConfig = Field(default_factory=FeatureFlagsConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    self_correction: SelfCorrectionConfig = Field(default_factory=SelfCorrectionConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    def __init__(self, **kwargs):
        """初始化配置，加载 config.json 文件覆盖"""
        super().__init__(**kwargs)
        self._load_file_config()

    def get(self, path: str, default: Any = None) -> Any:
        """
        快捷访问嵌套配置。

        Args:
            path: 点分路径，如 "llm.model", "rag.retrieval.final_k"
            default: 默认值

        Returns:
            配置值或默认值

        Example:
            settings.get("llm.model")
            settings.get("database.host", "localhost")
        """
        keys = path.split(".")
        obj = self
        for key in keys:
            if hasattr(obj, key):
                obj = getattr(obj, key)
            else:
                return default
        return obj

    def _load_file_config(self):
        """从 config.json 文件加载配置覆盖"""
        config_file = os.path.join("configs", "config.json")
        if os.path.exists(config_file):
            try:
                with open(config_file, encoding="utf-8") as f:
                    file_config = json.load(f)
                self._apply_file_config(file_config)
            except Exception as e:
                logger.error(f"Failed to load config file: {e}")

    def _apply_file_config(self, file_config: dict[str, Any]):
        """将文件配置应用到当前 Settings 对象（环境变量优先级更高）"""
        # 映射文件中的 storage 到 storage_s3 和 storage_local
        if "storage" in file_config:
            storage_data = file_config.pop("storage")
            # 区分 S3 配置和本地配置
            if "s3_endpoint" in storage_data or "s3_access_key" in storage_data:
                for key, value in storage_data.items():
                    if hasattr(self.storage_s3, key):
                        # 检查是否有对应的环境变量设置
                        env_key = self._get_env_alias(self.storage_s3, key)
                        if not os.getenv(env_key):
                            setattr(self.storage_s3, key, value)
            else:
                # 本地存储配置
                for key, value in storage_data.items():
                    if hasattr(self.storage_local, key):
                        setattr(self.storage_local, key, value)

        # 处理 env_overrides - 不再需要，pydantic 自动处理
        file_config.pop("env_overrides", None)

        # 应用其他配置
        for key, value in file_config.items():
            if hasattr(self, key):
                config_obj = getattr(self, key)
                if isinstance(value, dict) and hasattr(config_obj, "model_dump"):
                    # 嵌套配置对象，需要递归处理
                    self._apply_nested_config(config_obj, value)
                else:
                    setattr(self, key, value)

    def _apply_nested_config(self, config_obj: BaseSettings, value: dict[str, Any]):
        """递归应用嵌套配置"""
        for sub_key, sub_value in value.items():
            if hasattr(config_obj, sub_key):
                sub_config = getattr(config_obj, sub_key)
                if isinstance(sub_value, dict) and hasattr(sub_config, "model_dump"):
                    # 更深层的嵌套
                    self._apply_nested_config(sub_config, sub_value)
                else:
                    # 检查是否有对应的环境变量设置
                    env_key = self._get_env_alias(config_obj, sub_key)
                    if not os.getenv(env_key):
                        setattr(config_obj, sub_key, sub_value)

    def _get_env_alias(self, config_obj: BaseSettings, field_name: str) -> str | None:
        """获取字段的环境变量别名"""
        # 从 Pydantic 模型字段获取 alias
        fields = config_obj.model_fields
        if field_name in fields:
            alias = fields[field_name].alias
            if alias:
                return alias
        return field_name.upper()

    def validate_security(self) -> None:
        """验证安全配置"""
        import warnings

        # 检查 JWT 密钥
        insecure_keys = [
            "your-secret-key-keep-it-secret",
            "secret",
            "changeme",
            "password",
            "123456",
            "admin",
        ]
        if self.auth.secret_key in insecure_keys or len(self.auth.secret_key) < 32:
            raise ValueError(
                f"安全配置错误: auth.secret_key 使用了不安全的默认值 '{self.auth.secret_key}'。"
                f"请在 configs/config.json 中设置一个至少 32 位的随机字符串作为密钥，"
                f"或通过环境变量 AUTH_SECRET_KEY 设置。"
            )

        # 检查数据库密码
        db_password = self.database.password
        if db_password in insecure_keys or (db_password and len(db_password) < 8):
            raise ValueError(
                "安全配置错误: database.password 使用了不安全的默认值。"
                "请在 configs/config.json 中设置一个安全的数据库密码，"
                "或通过环境变量 DB_PASSWORD 设置。"
            )

        # 检查 API 密钥
        api_key = self.llm.api_key
        if api_key in insecure_keys or api_key == "":
            warnings.warn(
                "警告: llm.api_key 未配置，将无法使用 OpenAI 等云端 LLM。"
                "请在 configs/config.json 中设置 API Key，或通过环境变量 LLM_API_KEY 设置。"
            )


# 创建全局 Settings 单例
settings = Settings()
