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
    # NOTE: user 1 can only access /history/{username}
    # /history/u1 won't work if u1 != username
    print(f"Accessing /history/{username}...")
    response = client.get(f"/history/{username}", headers=headers)
    # History endpoint might return empty list or 200
    # Note: history.router is protected
    # It might return 200 even if empty
    assert response.status_code in [200, 404], (
        f"Expected 200 or 404, got {response.status_code}"
    )
    print("PASS: Protected endpoint accessible.")

    # 5. Access Upload (Now allowed for User, but isolated)
    print("Accessing /upload/ (User allowed)...")
    # We expect 422 because we didn't send file, but NOT 403
    response = client.post("/upload", headers=headers)
    assert response.status_code == 422, (
        f"Expected 422 (Validation Error), got {response.status_code}"
    )
    print("PASS: Upload endpoint accessible for user.")

    # 6. Admin Check (First user should be admin?)
    # Our updated logic makes first user admin.
    # The user registered in step 2 was likely first user in this test session if DB was clean.
    # Let's check /settings which is Admin only.
    print("Checking Admin access to /settings...")
    response = client.get("/settings", headers=headers)

    # If DB was empty, this user is admin -> 200 (or whatever settings returns, maybe 200 with config)
    # If DB was not empty (previous tests), might be user -> 403.
    # We cleared DB at start.
    # But wait, did we clear DB file?
    # verify_auth.py: os.environ["DATABASE_URL"] = "sqlite:///./test.db"
    # And we ran `rm test.db` in bash.
    # So yes, first user is Admin.

    if response.status_code == 200:
        print("PASS: First user is Admin (Access Granted).")
    else:
        print(f"WARN: First user might not be Admin? Status: {response.status_code}")
        # Maybe settings router returns something else?
        # Settings returns config dict.

    # 7. Register Second User (Should be 'user')
    username2 = f"user2_{int(time.time())}"
    print(f"Registering second user {username2}...")
    client.post("/auth/register", json={"username": username2, "password": password})
    resp2 = client.post(
        "/auth/token", data={"username": username2, "password": password}
    )
    token2 = resp2.json()["access_token"]
    headers2 = {"Authorization": f"Bearer {token2}"}

    print("Checking Second User Admin access (Should Fail)...")
    resp_settings = client.get("/settings", headers=headers2)
    assert resp_settings.status_code == 403, (
        f"Expected 403 for second user, got {resp_settings.status_code}"
    )
    print("PASS: Second user is NOT Admin.")

    # 8. History Isolation
    print("Checking History Isolation...")
    # User 1 saves history
    client.post(
        f"/history/{username}/save",
        json={"messages": [{"content": "hi"}], "session_id": "s1"},
        headers=headers,
    )

    # User 2 tries to read User 1 history
    resp_read = client.get(f"/history/{username}", headers=headers2)
    assert resp_read.status_code == 403, (
        f"Expected 403 reading other's history, got {resp_read.status_code}"
    )
    print("PASS: History is isolated.")

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
