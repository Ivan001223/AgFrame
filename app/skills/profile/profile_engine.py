from __future__ import annotations

import json
import logging
import time
from typing import Any

from app.infrastructure.database.stores import MySQLProfileStore
from app.infrastructure.utils.json_parser import parse_json_from_llm
from app.runtime.llm.llm_factory import get_llm

logger = logging.getLogger(__name__)


def _default_profile() -> dict[str, Any]:
    """返回默认的空用户画像结构"""
    return {
        "basic_info": {"name": None, "role": None, "location": None},
        "tech_profile": {"languages": [], "tools": []},
        "preferences": {
            "communication_style": None,
            "language": "中文",
            "interaction_protocol": None,
            "tone_instruction": None,
        },
        "facts": [],
    }


def normalize_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """
    标准化用户画像数据，确保所有必要字段都存在且格式正确。
    
    Args:
        profile: 原始用户画像数据
        
    Returns:
        Dict: 标准化后的用户画像
    """
    base = _default_profile()
    for k, v in base.items():
        if k not in profile or profile[k] is None:
            profile[k] = v
    if not isinstance(profile.get("basic_info"), dict):
        profile["basic_info"] = base["basic_info"]
    if not isinstance(profile.get("tech_profile"), dict):
        profile["tech_profile"] = base["tech_profile"]
    if not isinstance(profile.get("preferences"), dict):
        profile["preferences"] = base["preferences"]
    if not isinstance(profile.get("facts"), list):
        profile["facts"] = []

    profile["tech_profile"]["languages"] = list(profile["tech_profile"].get("languages") or [])
    profile["tech_profile"]["tools"] = list(profile["tech_profile"].get("tools") or [])

    # 标准化 facts 列表
    normalized_facts: list[dict[str, Any]] = []
    for f in profile["facts"]:
        if isinstance(f, str):
            normalized_facts.append({"text": f, "confidence_score": 0.6, "last_verified_at": None})
        elif isinstance(f, dict) and f.get("text"):
            normalized_facts.append(
                {
                    "text": str(f.get("text")),
                    "confidence_score": float(f.get("confidence_score") or 0.6),
                    "last_verified_at": f.get("last_verified_at"),
                }
            )
    profile["facts"] = normalized_facts
    return profile


def apply_forgetting(profile: dict[str, Any], now: int | None = None, max_age_days: int = 90) -> dict[str, Any]:
    """
    应用遗忘机制：降低旧事实的置信度，移除置信度过低的事实。
    
    Args:
        profile: 用户画像
        now: 当前时间戳
        max_age_days: 开始衰减的天数阈值
        
    Returns:
        Dict: 更新后的用户画像
    """
    now_ts = int(now or time.time())
    cutoff = now_ts - max_age_days * 24 * 3600
    kept: list[dict[str, Any]] = []
    for f in profile.get("facts", []):
        last = f.get("last_verified_at")
        conf = float(f.get("confidence_score") or 0.6)
        if last is None:
            f["last_verified_at"] = now_ts
            kept.append(f)
            continue
        try:
            last_int = int(last)
        except (ValueError, TypeError) as e:
            logger.debug(f"Failed to parse last_verified_at: {e}")
            f["last_verified_at"] = now_ts
            kept.append(f)
            continue
        # 如果事实很久未验证，降低置信度
        if last_int < cutoff:
            conf *= 0.8
            f["confidence_score"] = conf
        # 移除置信度过低的事实
        if conf >= 0.2:
            kept.append(f)
    profile["facts"] = kept
    return profile


def extract_base_profile(conversation_history: str) -> dict[str, Any]:
    """
    从完整的对话历史中提取初始用户画像。
    
    Args:
        conversation_history: 对话历史文本
        
    Returns:
        Dict: 提取出的用户画像
    """
    llm = get_llm(temperature=0, streaming=False, json_mode=True)
    prompt = (
        "你是一个专业的“用户画像侧写师”。\n"
        "阅读用户与 AI 的对话历史，从中提取关键结构化信息。\n"
        "如果对话中没有体现某项信息，请填写 null；严禁编造或推测。\n"
        "facts 仅保留长期有效的事实或偏好，不要记录寒暄与即时情绪。\n"
        "facts 元素使用对象格式：{\"text\": string, \"confidence_score\": number, \"last_verified_at\": number|null}。\n"
        "仅输出一个标准 JSON 对象，不要包含其他文字。\n\n"
        "<conversation_history>\n"
        f"{conversation_history}\n"
        "</conversation_history>\n\n"
        "输出 JSON Schema:\n"
        "{\n"
        '  \"basic_info\": {\"name\": \"string or null\", \"role\": \"string or null\", \"location\": \"string or null\"},\n'
        '  \"tech_profile\": {\"languages\": [\"string\"], \"tools\": [\"string\"]},\n'
        '  \"preferences\": {\n'
        '    \"communication_style\": \"string or null\",\n'
        '    \"language\": \"string (e.g., 中文, English)\",\n'
        '    \"interaction_protocol\": null,\n'
        '    \"tone_instruction\": null\n'
        "  },\n"
        '  \"facts\": [{\"text\": \"string\", \"confidence_score\": 0.0, \"last_verified_at\": null}]\n'
        "}"
    )
    raw = llm.invoke(prompt).content
    data = parse_json_from_llm(str(raw))
    if not isinstance(data, dict):
        return _default_profile()
    return apply_forgetting(normalize_profile(data))


def incremental_update_profile(old_profile: dict[str, Any], chat_log: str) -> dict[str, Any]:
    """
    根据新的对话片段增量更新用户画像。
    
    Args:
        old_profile: 旧的用户画像
        chat_log: 新的对话片段
        
    Returns:
        Dict: 更新后的用户画像
    """
    llm = get_llm(temperature=0, streaming=False, json_mode=True)
    old_profile_json = json.dumps(old_profile, ensure_ascii=False)
    prompt = (
        "你是一个数据库更新逻辑控制器。你的任务是根据【最新的对话片段】来更新【已有的用户画像】。\n\n"
        "<old_profile>\n"
        f"{old_profile_json}\n"
        "</old_profile>\n\n"
        "<chat_log>\n"
        f"{chat_log}\n"
        "</chat_log>\n\n"
        "规则：\n"
        "1) ADD：发现旧画像中不存在的新事实则新增。\n"
        "2) UPDATE：新对话与旧画像冲突，以新对话为准。\n"
        "3) KEEP：未提及字段保持不变。\n"
        "4) IGNORE：忽略即时性、非长期有效信息。\n"
        "5) facts 仅保留长期有效事实/偏好；重复则合并；每条 fact 维护 confidence_score 与 last_verified_at。\n\n"
        "只输出更新后的完整 JSON，结构与旧画像一致，不要输出其他文字。"
    )
    raw = llm.invoke(prompt).content
    data = parse_json_from_llm(str(raw))
    if not isinstance(data, dict):
        return apply_forgetting(normalize_profile(old_profile))
    merged = normalize_profile(data)
    return apply_forgetting(merged)


def analyze_interaction_protocol(conversation_samples: str) -> dict[str, Any]:
    """
    分析用户的交互偏好和风格。
    
    Args:
        conversation_samples: 对话样本
        
    Returns:
        Dict: 分析结果（偏好配置）
    """
    llm = get_llm(temperature=0, streaming=False, json_mode=True)
    prompt = (
        "你是一个沟通心理学专家。分析用户的提问方式，总结他对 AI 回答风格的偏好。\n"
        "仅输出 JSON，不要输出其他文字。\n\n"
        "<conversation_samples>\n"
        f"{conversation_samples}\n"
        "</conversation_samples>\n\n"
        "输出格式：\n"
        "{\n"
        '  \"conciseness_preference\": \"Low|Medium|High\",\n'
        '  \"code_style\": \"Snippet|Full-Solution\",\n'
        '  \"theoretical_depth\": \"Theory-First|Practical-First\",\n'
        '  \"tone_instruction\": \"string\"\n'
        "}"
    )
    raw = llm.invoke(prompt).content
    data = parse_json_from_llm(str(raw))
    return data if isinstance(data, dict) else {}


class UserProfileEngine:
    """
    用户画像引擎。
    负责从数据库获取、更新和管理用户画像。
    """
    def __init__(self):
        self.store = MySQLProfileStore()

    def get_profile(self, user_id: str) -> dict[str, Any]:
        """获取指定用户的画像"""
        row = self.store.get_profile(user_id)
        if not row:
            return _default_profile()
        profile = row.get("profile") or {}
        return apply_forgetting(normalize_profile(profile))

    def upsert_profile(self, user_id: str, profile: dict[str, Any], version: int) -> None:
        """更新或插入用户画像"""
        self.store.upsert_profile(user_id, normalize_profile(profile), version=version)
