from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.server.api import health


def test_health_endpoints():
    app = FastAPI()
    app.include_router(health.router)
    c = TestClient(app)

    r1 = c.get("/health")
    assert r1.status_code == 200
    assert r1.json()["status"] == "healthy"

    r2 = c.get("/health/ready")
    assert r2.status_code == 200
    assert r2.json()["status"] == "ready"

    r3 = c.get("/health/live")
    assert r3.status_code == 200
    assert r3.json()["status"] == "alive"

