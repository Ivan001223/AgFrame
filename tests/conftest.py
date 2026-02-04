"""
AgFrame 测试配置
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


@pytest.fixture(autouse=True)
def suppress_warnings():
    """自动抑制测试中的警告"""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture
def mock_config():
    """提供测试用的配置"""
    return {
        "llm": {
            "model": "test-model",
            "base_url": "http://localhost:8000",
            "api_key": "test-key",
        },
        "database": {
            "type": "postgres",
            "host": "localhost",
            "port": 5432,
            "user": "test",
            "password": "test",
            "db_name": "test_db",
        },
        "redis": {
            "host": "localhost",
            "port": 6379,
        },
    }


@pytest.fixture
def sample_agent_state():
    """提供测试用的 AgentState"""
    from langchain_core.messages import HumanMessage, AIMessage

    return {
        "messages": [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ],
        "next_step": "general",
        "reasoning": "Test reasoning",
        "context": {},
        "user_id": "test_user",
        "session_id": "test_session",
    }


@pytest.fixture
def sample_conversation_history():
    """提供测试用的对话历史"""
    return [
        {"role": "user", "content": "Hello", "created_at": 1234567890},
        {"role": "assistant", "content": "Hi there!", "created_at": 1234567891},
    ]
