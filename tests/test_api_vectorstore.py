from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.server.api import vectorstore as vectorstore_api


def test_vectorstore_clear(monkeypatch: pytest.MonkeyPatch):
    import sys
    import types

    cleared = {"v": False}

    class _E:
        def clear(self):
            cleared["v"] = True

    fake_mod = types.ModuleType("app.skills.rag.rag_engine")
    fake_mod.get_rag_engine = lambda: _E()
    monkeypatch.setitem(sys.modules, "app.skills.rag.rag_engine", fake_mod)

    app = FastAPI()
    app.include_router(vectorstore_api.router)
    c = TestClient(app)
    r = c.post("/vectorstore/docs/clear")
    assert r.status_code == 200
    assert r.json()["message"] == "cleared"
    assert cleared["v"] is True
