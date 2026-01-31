import requests
import base64
import json
import uuid
from datetime import datetime

SECRET_KEY = "sk-lf-99299b0d-ebb2-409c-9238-3fb2e6d2d650"
PUBLIC_KEY = "pk-lf-151b2be3-f847-4e4f-8428-6d4068671248"
HOST = "http://localhost:3000"


def test_http():
    print(f"Testing connection to {HOST}...")

    # 1. Health Check
    try:
        resp = requests.get(f"{HOST}/api/public/health", timeout=5)
        if resp.status_code == 200:
            print(f"[OK] Health Check Passed: {resp.status_code}")
        else:
            print(f"[WARN] Health Check Warning: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"[ERR] Health Check Failed: {e}")
        # Continue anyway to see if it's just the health endpoint

    # 2. Auth & Ingestion Check
    url = f"{HOST}/api/public/ingestion"
    auth_str = f"{PUBLIC_KEY}:{SECRET_KEY}"
    b64_auth = base64.b64encode(auth_str.encode()).decode()

    headers = {"Authorization": f"Basic {b64_auth}", "Content-Type": "application/json"}

    payload = {
        "batch": [
            {
                "id": str(uuid.uuid4()),
                "type": "trace-create",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "body": {
                    "name": "http-connection-test",
                    "userId": "http-tester",
                    "metadata": {"source": "python-requests"},
                },
            }
        ]
    }

    try:
        print(f"Sending test trace to {url}...")
        resp = requests.post(url, headers=headers, json=payload, timeout=5)
        if resp.status_code in [200, 201, 207]:
            print(f"[OK] Ingestion Test Passed! Status: {resp.status_code}")
            print(f"Response: {resp.json()}")
        else:
            print(f"[ERR] Ingestion Test Failed. Status: {resp.status_code}")
            print(f"Response: {resp.text}")
    except Exception as e:
        print(f"[ERR] Ingestion Request Failed: {e}")


if __name__ == "__main__":
    test_http()
