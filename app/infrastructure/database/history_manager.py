import os
import json
import time
from typing import List, Dict, Any, Optional

from app.infrastructure.database.conversation_utils import derive_session_title, should_bump_updated_at

HISTORY_FILE = os.path.join("data", "chat_history.json")

class HistoryManager:
    """
    基于 JSON 文件的本地对话历史管理器。
    用于在没有数据库环境时的降级存储方案。
    """
    def __init__(self):
        self._ensure_data_dir()
        
    def _ensure_data_dir(self):
        """确保数据目录和历史文件存在"""
        os.makedirs("data", exist_ok=True)
        if not os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def _load_data(self) -> Dict[str, Any]:
        """加载 JSON 数据"""
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _save_data(self, data: Dict[str, Any]):
        """保存 JSON 数据"""
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_history(self, user_id: str) -> List[Dict[str, Any]]:
        """获取用户的会话列表，按时间戳倒序排列。"""
        data = self._load_data()
        user_sessions = data.get(user_id, {})
        # 字典转列表
        sessions_list = list(user_sessions.values())
        # 按 updated_at 倒序排序
        sessions_list.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
        return sessions_list

    def save_session(self, user_id: str, session_id: str, messages: List[Dict[str, Any]], title: Optional[str] = None):
        """保存或更新一次聊天会话。"""
        data = self._load_data()
        if user_id not in data:
            data[user_id] = {}
            
        now = int(time.time())
        
        # 若会话已存在则更新
        if session_id in data[user_id]:
            session = data[user_id][session_id]
            # 只有当消息数量增加时（即产生了新对话），才更新 updated_at
            # 前端自动保存可能会频繁调用，但如果只是加载历史记录点击查看，不应更新时间戳
            # 简单的判断逻辑：如果传入的 messages 长度比已存的长，或者是全新的会话，才更新时间
            old_messages = session.get("messages", [])
            
            session["messages"] = messages
            session["title"] = derive_session_title(messages, title)
                
            # 只有产生新内容时才置顶 (更新 updated_at)
            if should_bump_updated_at(old_messages, messages):
                session["updated_at"] = now
        else:
            # 创建新会话
            # 若未提供标题则自动生成
            title = derive_session_title(messages, title)
            
            data[user_id][session_id] = {
                "id": session_id,
                "title": title,
                "created_at": now,
                "updated_at": now,
                "messages": messages
            }
            
        self._save_data(data)
        return data[user_id][session_id]

    def delete_session(self, user_id: str, session_id: str):
        """删除会话"""
        data = self._load_data()
        if user_id in data and session_id in data[user_id]:
            del data[user_id][session_id]
            self._save_data(data)
            return True
        return False

history_manager = HistoryManager()
