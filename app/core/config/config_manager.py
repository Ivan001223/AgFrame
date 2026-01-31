import os
import json
from typing import Dict, Any, Optional
from app.core.config.env import init_env

class ConfigManager:
    """
    配置管理器（单例模式）。
    负责加载、合并和提供全局配置信息。
    支持从 config.json 文件加载，并回退到默认值和环境变量。
    """
    _instance = None
    CONFIG_FILE = "config.json"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._init_config()
        return cls._instance

    def _init_config(self):
        """初始化配置：加载环境变量 -> 加载文件 -> 合并默认值"""
        init_env()
        self.config = self._load_from_file() or self._load_defaults()
        defaults = self._load_defaults()
        self._deep_merge(self.config, defaults)
        
    def _load_defaults(self) -> Dict[str, Any]:
        """加载默认配置结构，优先使用环境变量"""
        return {
            "llm": {
                "api_key": os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", "")),
                "base_url": os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
                "model": os.getenv("LLM_MODEL", "gpt-4o")
            },
            "model_manager": {
                "provider": os.getenv("MODEL_MANAGER_PROVIDER", "modelscope"),
                "cache_dir": os.getenv("MODEL_MANAGER_CACHE_DIR", ""),
                "revision": os.getenv("MODEL_MANAGER_REVISION", ""),
                "trust_remote_code": os.getenv("MODEL_MANAGER_TRUST_REMOTE_CODE", "true").lower() == "true",
                "modelscope_fallback_to_hf": os.getenv("MODELSCOPE_FALLBACK_TO_HF", "true").lower() == "true",
            },
            "local_models": {
                "ocr_model": os.getenv("MODEL_PATH_OCR", ""),
                "embedding_model": os.getenv("MODEL_PATH_EMBEDDING", ""),
                "rerank_model": os.getenv("MODEL_PATH_RERANKER", ""),
            },
            "embeddings": {
                "provider": os.getenv("EMBEDDINGS_PROVIDER", "modelscope"),
                "backend": os.getenv("EMBEDDINGS_BACKEND", "sentence_transformers"),
                "model_name": os.getenv("EMBEDDINGS_MODEL_NAME", ""),
                "env_var": os.getenv("EMBEDDINGS_ENV_VAR", "MODEL_PATH_EMBEDDING"),
                "device": os.getenv("EMBEDDINGS_DEVICE", "auto"),
                "batch_size": int(os.getenv("EMBEDDINGS_BATCH_SIZE", "32")),
                "max_length": int(os.getenv("EMBEDDINGS_MAX_LENGTH", "512")),
                "pooling": os.getenv("EMBEDDINGS_POOLING", "mean"),
                "normalize": os.getenv("EMBEDDINGS_NORMALIZE", "true").lower() == "true",
                "query_prefix": os.getenv("EMBEDDINGS_QUERY_PREFIX", ""),
                "doc_prefix": os.getenv("EMBEDDINGS_DOC_PREFIX", ""),
            },
            "reranker": {
                "provider": os.getenv("RERANKER_PROVIDER", "modelscope"),
                "backend": os.getenv("RERANKER_BACKEND", "sentence_transformers"),
                "model_name": os.getenv("RERANKER_MODEL_NAME", ""),
                "env_var": os.getenv("RERANKER_ENV_VAR", "MODEL_PATH_RERANKER"),
                "device": os.getenv("RERANKER_DEVICE", "auto"),
                "batch_size": int(os.getenv("RERANKER_BATCH_SIZE", "16")),
                "max_length": int(os.getenv("RERANKER_MAX_LENGTH", "512")),
                "query_prefix": os.getenv("RERANKER_QUERY_PREFIX", ""),
                "doc_prefix": os.getenv("RERANKER_DOC_PREFIX", ""),
                "window_size": None,
                "stride": None,
                "transformers_model_type": os.getenv("RERANKER_TRANSFORMERS_MODEL_TYPE", "auto"),
            },
            "search": {
                "provider": os.getenv("SEARCH_PROVIDER", "duckduckgo"),
                "tavily_api_key": os.getenv("TAVILY_API_KEY", ""),
            },
            "database": {
                "type": "mysql",
                "host": os.getenv("DB_HOST", "localhost"),
                "port": int(os.getenv("DB_PORT", 3306)),
                "user": os.getenv("DB_USER", "root"),
                "password": os.getenv("DB_PASSWORD", "password"),
                "db_name": os.getenv("DB_NAME", "agent_app")
            },
            "general": {
                "app_name": os.getenv("APP_NAME", "My Agent App")
            },
            "prompt": {
                "budget": {
                    "max_recent_history_lines": int(os.getenv("PROMPT_MAX_RECENT_HISTORY_LINES", "10")),
                    "max_docs": int(os.getenv("PROMPT_MAX_DOCS", "3")),
                    "max_memories": int(os.getenv("PROMPT_MAX_MEMORIES", "3")),
                    "max_doc_chars_total": int(os.getenv("PROMPT_MAX_DOC_CHARS_TOTAL", "6000")),
                    "max_memory_chars_total": int(os.getenv("PROMPT_MAX_MEMORY_CHARS_TOTAL", "3000")),
                    "max_item_chars": int(os.getenv("PROMPT_MAX_ITEM_CHARS", "2000")),
                }
            },
            "nodes": {
                "enabled": ["router", "retrieve_docs", "retrieve_memories", "assemble", "generate"]
            },
            "feature_flags": {
                "enable_docs_rag": os.getenv("ENABLE_DOCS_RAG", "true").lower() == "true",
                "enable_chat_memory": os.getenv("ENABLE_CHAT_MEMORY", "true").lower() == "true",
                "allow_dangerous_deserialization": os.getenv("ALLOW_DANGEROUS_DESERIALIZATION", "false").lower() == "true",
                "enable_tools_write_file": os.getenv("ENABLE_TOOLS_WRITE_FILE", "false").lower() == "true",
                "enable_tools_python_repl": os.getenv("ENABLE_TOOLS_PYTHON_REPL", "false").lower() == "true",
            },
        }

    def _load_from_file(self) -> Optional[Dict[str, Any]]:
        """从 config.json 文件加载配置"""
        if os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载配置文件失败：{e}")
                return None
        return None

    def _deep_merge(self, target: Dict, source: Dict):
        """递归合并字典，确保默认值补充缺失字段"""
        for key, value in source.items():
            if key not in target:
                target[key] = value
            elif isinstance(value, dict) and isinstance(target.get(key), dict):
                self._deep_merge(target[key], value)

    def get_config(self) -> Dict[str, Any]:
        """获取当前配置字典"""
        return self.config

    def update_config(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """更新配置并持久化到文件"""
        self._recursive_update(self.config, new_config)
        self._save_to_file()
        return self.config

    def _recursive_update(self, target: Dict, source: Dict):
        """递归更新目标字典"""
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._recursive_update(target[key], value)
            else:
                target[key] = value

    def _save_to_file(self):
        """保存配置到 config.json"""
        try:
            with open(self.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存配置文件失败：{e}")

config_manager = ConfigManager()
