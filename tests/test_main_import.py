from __future__ import annotations


def test_server_main_imports_chat_routes():
    from app.server.main import app

    paths = {route.path for route in app.routes}
    assert "/chat/workbench-invoke" in paths
    assert "/chat/invoke" in paths
