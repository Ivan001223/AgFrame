import os
import json
from typing import Dict, Any, Optional
from app.core.config.env import init_env

class ConfigManager:
    _instance = None
    CONFIG_FILE = "config.json"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._init_config()
        return cls._instance

    def _init_config(self):
        init_env()
        self.config = self._load_from_file() or self._load_defaults()
        defaults = self._load_defaults()
        self._deep_merge(self.config, defaults)
        
    def _load_defaults(self) -> Dict[str, Any]:
        return {
            "llm": {
                "api_key": os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", "")),
                "base_url": os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
                "model": os.getenv("LLM_MODEL", "gpt-4o")
            },
            "database": {
                "type": "mysql",
                "host": os.getenv("DB_HOST", "localhost"),
                "port": int(os.getenv("DB_PORT", 3306)),
                "user": os.getenv("DB_USER", "root"),
                "password": os.getenv("DB_PASSWORD", "password"),
                "db_name": os.getenv("DB_NAME", "agent_app")
            },
            # 在此添加自定义模块配置
        }

    def _load_from_file(self) -> Optional[Dict[str, Any]]:
        if os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载配置文件失败：{e}")
                return None
        return None

    def _deep_merge(self, target: Dict, source: Dict):
        for key, value in source.items():
            if key not in target:
                target[key] = value
            elif isinstance(value, dict) and isinstance(target.get(key), dict):
                self._deep_merge(target[key], value)

    def get_config(self) -> Dict[str, Any]:
        return self.config

    def update_config(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        self._recursive_update(self.config, new_config)
        self._save_to_file()
        return self.config

    def _recursive_update(self, target: Dict, source: Dict):
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._recursive_update(target[key], value)
            else:
                target[key] = value

    def _save_to_file(self):
        try:
            with open(self.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存配置文件失败：{e}")

config_manager = ConfigManager()
