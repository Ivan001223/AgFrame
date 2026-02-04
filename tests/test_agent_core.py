"""
Agent Core 测试 - 测试 Agent 核心数据结构
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAgentCoreDataTypes:
    """Agent 核心数据类型测试"""

    def test_agent_state_type_exists(self):
        """测试 AgentState 类型存在"""
        from app.runtime.graph.state import AgentState
        assert AgentState is not None

    def test_route_decision_type_exists(self):
        """测试 RouteDecision 类型存在"""
        from app.runtime.graph.state import RouteDecision
        assert RouteDecision is not None

    def test_citation_type_exists(self):
        """测试 Citation 类型存在"""
        from app.runtime.graph.state import Citation
        assert Citation is not None

    def test_action_required_type_exists(self):
        """测试 ActionRequired 类型存在"""
        from app.runtime.graph.state import ActionRequired
        assert ActionRequired is not None


class TestAgentStateStructure:
    """AgentState 结构测试"""

    def test_empty_state(self):
        """测试空状态"""
        from app.runtime.graph.state import AgentState

        state: AgentState = {
            "messages": [],
            "next_step": "",
            "reasoning": "",
            "context": {},
            "user_id": "test_user",
            "session_id": "test_session",
        }

        assert state["messages"] == []
        assert state["user_id"] == "test_user"
        assert isinstance(state["context"], dict)

    def test_state_with_messages(self, sample_agent_state):
        """测试带消息的状态"""
        from app.runtime.graph.state import AgentState

        state: AgentState = sample_agent_state
        assert len(state["messages"]) == 2
        assert state["next_step"] == "general"


class TestRouteDecision:
    """路由决策测试"""

    def test_route_decision_creation(self):
        """测试创建路由决策"""
        from app.runtime.graph.state import RouteDecision

        decision: RouteDecision = {
            "needs_docs": True,
            "needs_history": False,
            "needs_profile": True,
            "reasoning": "需要文档和用户画像",
        }

        assert decision["needs_docs"] is True
        assert decision["needs_history"] is False
        assert decision["needs_profile"] is True

    def test_route_decision_empty(self):
        """测试空路由决策"""
        from app.runtime.graph.state import RouteDecision

        decision: RouteDecision = {
            "needs_docs": False,
            "needs_history": False,
            "needs_profile": False,
            "reasoning": "",
        }

        assert all(v is False for k, v in decision.items() if k.startswith("needs_"))


class TestCitations:
    """引用测试"""

    def test_citation_structure(self):
        """测试引用结构"""
        from app.runtime.graph.state import Citation

        citation: Citation = {
            "kind": "doc",
            "label": "Python 编程指南",
            "doc_id": "doc_001",
            "session_id": "session_001",
            "page": 1,
            "source": "/data/docs/python.md",
        }

        assert citation["kind"] == "doc"
        assert citation["doc_id"] == "doc_001"
        assert citation["page"] == 1

    def test_citation_types(self):
        """测试引用类型"""
        from app.runtime.graph.state import Citation

        doc_citation: Citation = {"kind": "doc", "doc_id": "1", "source": "/docs/1.md"}
        memory_citation: Citation = {"kind": "memory", "session_id": "sess_1"}
        profile_citation: Citation = {"kind": "profile", "user_id": "user_1"}

        assert doc_citation["kind"] == "doc"
        assert memory_citation["kind"] == "memory"
        assert profile_citation["kind"] == "profile"


class TestActionRequired:
    """所需操作测试"""

    def test_action_required_structure(self):
        """测试所需操作结构"""
        from app.runtime.graph.state import ActionRequired

        action: ActionRequired = {
            "action_type": "file_upload",
            "description": "请上传需要处理的文档",
            "payload": {"accepted_types": [".pdf", ".docx"]},
            "requires_approval": True,
            "approved": False,
        }

        assert action["action_type"] == "file_upload"
        assert action["requires_approval"] is True
        assert action["approved"] is False

    def test_action_approved(self):
        """测试已批准的操作"""
        from app.runtime.graph.state import ActionRequired

        action: ActionRequired = {
            "action_type": "confirm_delete",
            "description": "确认删除操作",
            "payload": {"item_id": "item_123"},
            "requires_approval": True,
            "approved": True,
        }

        assert action["approved"] is True


class TestTraceStructure:
    """追踪结构测试"""

    def test_trace_structure(self):
        """测试追踪结构"""
        from app.runtime.graph.state import AgentState

        state: AgentState = {
            "messages": [],
            "next_step": "generate",
            "reasoning": "生成回答",
            "context": {},
            "user_id": "test_user",
            "session_id": "test_session",
            "trace": {
                "node": "generate",
                "timestamp": "2025-01-01T00:00:00Z",
                "duration_ms": 150,
                "self_correction_attempts": 0,
            },
        }

        assert state["trace"]["node"] == "generate"
        assert state["trace"]["duration_ms"] == 150
        assert state["trace"]["self_correction_attempts"] == 0
