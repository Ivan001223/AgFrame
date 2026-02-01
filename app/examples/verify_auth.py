import sys
from unittest.mock import MagicMock

# Mock heavy dependencies BEFORE imports
sys.modules["torch"] = MagicMock()
sys.modules["torch.nn"] = MagicMock()
sys.modules["torch.nn.functional"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["timm"] = MagicMock()
sys.modules["einops"] = MagicMock()
sys.modules["pdf2image"] = MagicMock()

from app.infrastructure.config.config_manager import config_manager

config_manager.config["database"] = {
    "type": "sqlite",
    "url": "sqlite:///./test.db",
    "db_name": "test.db",
}

from fastapi.testclient import TestClient
from app.infrastructure.database.schema import ensure_schema
from app.server.main import app
import time

# Force schema creation
try:
    print("Creating schema...")
    ensure_schema()
    print("Schema created.")
except Exception as e:
    print(f"Schema creation failed: {e}")

client = TestClient(app)


def test_auth_flow():
    # 1. Test Unprotected access
    # /chat requires auth
    print("Testing 401 on /chat...")
    # LangServe mounts /chat/invoke usually
    response = client.post(
        "/chat/invoke",
        json={"input": {"messages": [{"role": "user", "content": "hi"}]}},
    )
    assert response.status_code == 401, f"Expected 401, got {response.status_code}"
    print("PASS: Unprotected access denied.")

    # 2. Register
    username = f"user_{int(time.time())}"
    password = "password123"
    print(f"Registering {username}...")
    response = client.post(
        "/auth/register", json={"username": username, "password": password}
    )
    assert response.status_code == 200, f"Register failed: {response.text}"
    user_data = response.json()
    assert user_data["username"] == username
    assert user_data["role"] == "user"
    print("PASS: Register success.")

    # 3. Login
    print("Logging in...")
    response = client.post(
        "/auth/token", data={"username": username, "password": password}
    )
    assert response.status_code == 200, f"Login failed: {response.text}"
    token_data = response.json()
    access_token = token_data["access_token"]
    headers = {"Authorization": f"Bearer {access_token}"}
    print("PASS: Login success.")

    # 4. Access Protected User Endpoint (History)
    print("Accessing /history/u1...")
    response = client.get("/history/u1", headers=headers)
    # History endpoint might return empty list or 200
    # Note: history.router is protected
    # It might return 200 even if empty
    assert response.status_code in [200, 404], (
        f"Expected 200 or 404, got {response.status_code}"
    )
    print("PASS: Protected endpoint accessible.")

    # 5. Access Admin Endpoint (Upload)
    print("Accessing /upload/ (Admin only)...")
    # Need to mock a file upload
    # But checking 403 is enough
    # If I try to upload without file, it might contain validation error 422 before 403 if dependencies run after?
    # Dependencies run first.
    # But Upload router might have logic.
    # Let's try GET /settings which is also admin
    response = client.get("/settings", headers=headers)
    assert response.status_code == 403, (
        f"Expected 403 for Admin route, got {response.status_code}"
    )
    print("PASS: Admin endpoint denied for user.")

    print("\nALL TESTS PASSED!")


if __name__ == "__main__":
    # Mock Redis for RateLimiter if needed or assume it's running?
    # fastap-limiter needs init. TestClient lifespan handling is tricky.
    # With TestClient(app), lifespan is called.
    # So Redis must be available.
    try:
        test_auth_flow()
    except Exception as e:
        print(f"FAILED: {e}")
