import os
import json
from typing import Dict, Any, Optional
from app.infrastructure.config.env import init_env


class ConfigManager:
    """
    配置管理器（单例模式）。
    负责加载、合并和提供全局配置信息。
    支持从 config.json 文件加载 env_overrides 映射，并应用环境变量覆盖。
    """

    _instance = None
    CONFIG_FILE = os.path.join("configs", "config.json")

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._init_config()
        return cls._instance

    def _init_config(self):
        """初始化配置：加载默认值 -> 加载 env_overrides 映射 -> 应用环境变量覆盖"""
        init_env()
        self.config = self._load_defaults()

        file_config = self._load_from_file()
        if file_config:
            self._recursive_update(self.config, file_config)

        self._apply_env_overrides()

    def _load_defaults(self) -> Dict[str, Any]:
        """加载默认配置结构"""
        return {
            "llm": {
                "api_key": "",
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4o",
                "structured_output_mode": "native_first",
                "json_mode_response_format": False,
            },
            "model_manager": {
                "provider": "modelscope",
                "cache_dir": "",
                "revision": "",
                "trust_remote_code": True,
                "modelscope_fallback_to_hf": True,
            },
            "local_models": {
                "ocr_model": "",
                "embedding_model": "",
                "rerank_model": "",
            },
            "embeddings": {
                "provider": "modelscope",
                "backend": "sentence_transformers",
                "model_name": "",
                "env_var": "MODEL_PATH_EMBEDDING",
                "device": "auto",
                "batch_size": 32,
                "max_length": 512,
                "pooling": "mean",
                "normalize": True,
                "query_prefix": "",
                "doc_prefix": "",
            },
            "reranker": {
                "provider": "modelscope",
                "backend": "sentence_transformers",
                "model_name": "",
                "env_var": "MODEL_PATH_RERANKER",
                "device": "auto",
                "batch_size": 16,
                "max_length": 512,
                "query_prefix": "",
                "doc_prefix": "",
                "window_size": None,
                "stride": None,
                "transformers_model_type": "auto",
            },
            "search": {
                "provider": "duckduckgo",
                "tavily_api_key": "",
                "serpapi_api_key": "",
                "cache_ttl": 3600,
            },
            "database": {
                "type": "postgres",
                "url": "",
                "host": "localhost",
                "port": 5432,
                "user": "postgres",
                "password": "password",
                "db_name": "agent_app",
            },
            "queue": {
                "redis_url": "redis://localhost:6379/0",
                "rabbitmq_url": "",
                "rabbitmq_management_url": "",
            },
            "storage": {
                "s3_endpoint": "",
                "s3_access_key": "",
                "s3_secret_key": "",
                "s3_bucket": "agframe",
                "s3_secure": False,
            },
            "auth": {
                "secret_key": "your-secret-key-keep-it-secret",
                "algorithm": "HS256",
                "access_token_expire_minutes": 30,
            },
            "general": {"app_name": "My Agent App"},
            "rag": {
                "retrieval": {
                    "mode": "hybrid",
                    "dense_k": 20,
                    "sparse_k": 20,
                    "candidate_k": 20,
                    "final_k": 3,
                    "rrf_k": 60,
                    "weights": [0.5, 0.5],
                }
            },
            "prompt": {
                "budget": {
                    "max_recent_history_lines": 10,
                    "max_docs": 3,
                    "max_memories": 3,
                    "max_doc_chars_total": 6000,
                    "max_memory_chars_total": 3000,
                    "max_item_chars": 2000,
                }
            },
            "nodes": {
                "enabled": [
                    "router",
                    "retrieve_docs",
                    "rerank_docs",
                    "retrieve_memories",
                    "assemble",
                    "generate",
                ]
            },
            "feature_flags": {
                "enable_docs_rag": True,
                "enable_chat_memory": True,
                "enable_self_correction": True,
                "enable_human_approval": False,
                "allow_dangerous_deserialization": False,
                "enable_tools_write_file": False,
                "enable_tools_python_repl": False,
                "enable_tools_python_executor": False,
                "pgvector_dimension": 1024,
            },
            "sandbox": {
                "enabled": False,
                "image": "python:3.11-slim",
                "timeout": 30,
                "memory_limit": "256m",
                "cpu_limit": 0.5,
            },
            "self_correction": {
                "max_attempts": 2,
            },
            "server": {
                "host": "0.0.0.0",
                "port": 8000,
                "cors_origins": ["*"],
            },
            "storage": {
                "documents_dir": "data/documents",
                "uploads_dir": "data/uploads",
                "data_dir": "data",
            },
        }

    def _apply_env_overrides(self):
        """根据 env_overrides 映射应用环境变量覆盖"""
        env_overrides = self.config.get("env_overrides", {})
        if not env_overrides:
            return

        for path_str, env_var in env_overrides.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                self._set_nested_value(path_str, env_value)

    def _set_nested_value(self, path_str: str, value: Any):
        """根据点分路径设置嵌套值"""
        keys = path_str.split(".")
        target = self.config
        for key in keys[:-1]:
            if key not in target:
                return
            target = target[key]
        last_key = keys[-1]
        if last_key in target:
            original = target[last_key]
            if isinstance(original, bool):
                target[last_key] = value.lower() == "true"
            elif isinstance(original, int):
                target[last_key] = int(value)
            elif isinstance(original, float):
                target[last_key] = float(value)
            elif isinstance(original, list) and value.startswith("[") and value.endswith("]"):
                pass
            else:
                target[last_key] = value

    def _load_from_file(self) -> Optional[Dict[str, Any]]:
        """从 configs/config.json 文件加载配置"""
        if os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载配置文件失败：{e}")
                return None
        return None

    def _recursive_update(self, target: Dict, source: Dict):
        """递归更新目标字典"""
        for key, value in source.items():
            if (
                isinstance(value, dict)
                and key in target
                and isinstance(target[key], dict)
            ):
                self._recursive_update(target[key], value)
            else:
                target[key] = value

    def get_config(self) -> Dict[str, Any]:
        """获取当前配置字典"""
        return self.config

    def update_config(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """更新配置并持久化到文件"""
        self._recursive_update(self.config, new_config)
        self._save_to_file()
        return self.config

    def _save_to_file(self):
        """保存配置到 config.json"""
        try:
            with open(self.CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存配置文件失败：{e}")


config_manager = ConfigManager()
