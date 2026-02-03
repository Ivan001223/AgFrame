import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, AIMessage


class TestAgentCore:
    """核心 Agent 集成测试 - 测试数据结构而非实际导入"""

    def test_state_initialization(self):
        """测试 AgentState 初始化"""
        from app.runtime.graph.state import AgentState

        state: AgentState = {
            "messages": [],
            "next_step": "",
            "reasoning": "",
            "context": {},
            "user_id": "test_user",
        }

        assert state["messages"] == []
        assert state["user_id"] == "test_user"
        assert isinstance(state["context"], dict)

    def test_route_decision_structure(self):
        """测试 RouteDecision 结构"""
        from app.runtime.graph.state import RouteDecision

        decision: RouteDecision = {
            "needs_docs": True,
            "needs_history": False,
            "reasoning": "用户请求需要检索文档",
        }

        assert decision["needs_docs"] is True
        assert decision["needs_history"] is False
        assert "文档" in decision["reasoning"]

    def test_citation_structure(self):
        """测试 Citation 结构"""
        from app.runtime.graph.state import Citation

        citation: Citation = {
            "kind": "doc",
            "label": "Python 异步编程指南",
            "doc_id": "doc_001",
            "session_id": "session_001",
            "page": 1,
            "source": "/data/docs/python_async.md",
        }

        assert citation["kind"] == "doc"
        assert citation["doc_id"] == "doc_001"

    def test_action_required_structure(self):
        """测试 ActionRequired 结构"""
        from app.runtime.graph.state import ActionRequired

        action: ActionRequired = {
            "action_type": "file_upload",
            "description": "需要用户确认上传文件",
            "payload": {"file_path": "/tmp/test.pdf"},
            "requires_approval": True,
            "approved": False,
        }

        assert action["action_type"] == "file_upload"
        assert action["requires_approval"] is True

    def test_full_message_flow(self):
        """测试完整消息流"""
        from app.runtime.graph.state import AgentState

        messages = [
            HumanMessage(content="你好"),
            AIMessage(content="你好！有什么可以帮助你的？"),
            HumanMessage(content="帮我查一下文档"),
        ]

        state: AgentState = {
            "messages": messages,
            "next_step": "rag",
            "reasoning": "用户请求检索文档",
            "context": {},
            "user_id": "test_user",
            "retrieved_docs": [],
            "retrieved_memories": [],
            "citations": [],
        }

        assert len(state["messages"]) == 3
        assert state["next_step"] == "rag"

    def test_error_handling_in_state(self):
        """测试状态中的错误处理"""
        from app.runtime.graph.state import AgentState

        state: AgentState = {
            "messages": [],
            "next_step": "",
            "reasoning": "",
            "context": {},
            "user_id": "test_user",
            "errors": ["Error: Connection timeout"],
        }

        assert len(state["errors"]) == 1
        assert "timeout" in state["errors"][0]

    def test_trace_debug_info(self):
        """测试调试追踪信息"""
        from app.runtime.graph.state import AgentState

        state: AgentState = {
            "messages": [],
            "next_step": "general",
            "reasoning": "",
            "context": {},
            "user_id": "test_user",
            "trace": {
                "node": "orchestrator",
                "timestamp": "2025-01-01T00:00:00Z",
                "duration_ms": 150,
            },
        }

        assert state["trace"]["node"] == "orchestrator"
        assert "duration_ms" in state["trace"]
