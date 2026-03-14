#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from fpdf import FPDF


def _request(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=data, method=method)
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            payload = {"raw": body}
        return exc.code, payload


def _json_request(url: str, *, method: str, payload: dict, token: str | None = None) -> tuple[int, dict]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return _request(url, method=method, headers=headers, data=json.dumps(payload).encode("utf-8"))


def _form_request(url: str, *, fields: dict[str, str]) -> tuple[int, dict]:
    body = urllib.parse.urlencode(fields).encode("utf-8")
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    return _request(url, method="POST", headers=headers, data=body)


def _multipart_request(url: str, *, file_field: str, file_name: str, file_bytes: bytes, token: str) -> tuple[int, dict]:
    boundary = f"----AgFrameBoundary{uuid.uuid4().hex}"
    content_type = mimetypes.guess_type(file_name)[0] or "application/octet-stream"
    lines = [
        f"--{boundary}",
        f'Content-Disposition: form-data; name="{file_field}"; filename="{file_name}"',
        f"Content-Type: {content_type}",
        "",
    ]
    body = "\r\n".join(lines).encode("utf-8") + b"\r\n" + file_bytes + b"\r\n" + f"--{boundary}--\r\n".encode("utf-8")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }
    return _request(url, method="POST", headers=headers, data=body)


def _assert(status: int, expected: int, payload: dict, step: str) -> dict:
    if status != expected:
        raise RuntimeError(f"{step} failed: expected {expected}, got {status}, payload={payload}")
    return payload


def _build_smoke_pdf_bytes() -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 10, "Smoke test document for AgFrame.\nThis PDF should be parseable.")
    raw = pdf.output(dest="S")
    if isinstance(raw, bytearray):
        return bytes(raw)
    if isinstance(raw, str):
        return raw.encode("latin-1")
    return raw


def main() -> int:
    parser = argparse.ArgumentParser(description="Run live AgFrame workbench smoke checks against a running server.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Running API base URL")
    parser.add_argument("--username", default=f"smoke_{uuid.uuid4().hex[:8]}", help="Smoke test username")
    parser.add_argument("--password", default="smoke-pass-123", help="Smoke test password")
    parser.add_argument("--task-timeout", type=int, default=90, help="Seconds to wait for upload task completion")
    parser.add_argument("--skip-upload", action="store_true", help="Skip upload/task/document checks")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    username = args.username
    password = args.password
    session_id = f"smoke-session-{uuid.uuid4().hex[:8]}"

    print(f"[1/7] register user: {username}")
    status, payload = _json_request(f"{base}/auth/register", method="POST", payload={"username": username, "password": password})
    if status not in {200, 400}:
        raise RuntimeError(f"register failed: status={status}, payload={payload}")
    if status == 400 and payload.get("detail") != "Username already registered":
        raise RuntimeError(f"unexpected register response: {payload}")

    print("[2/7] login")
    status, payload = _form_request(f"{base}/auth/token", fields={"username": username, "password": password})
    token = _assert(status, 200, payload, "login")["access_token"]

    if not args.skip_upload:
        print("[3/7] upload pdf")
        pdf_bytes = _build_smoke_pdf_bytes()
        status, payload = _multipart_request(
            f"{base}/upload",
            file_field="files",
            file_name="smoke.pdf",
            file_bytes=pdf_bytes,
            token=token,
        )
        body = _assert(status, 200, payload, "upload")
        result = body["results"][0]
        if result["status"] not in {"queued", "duplicate"}:
            raise RuntimeError(f"unexpected upload status: {result}")

        if result["status"] == "queued":
            print("[4/7] poll task")
            task_id = result["task_id"]
            deadline = time.time() + args.task_timeout
            task_payload = {}
            while time.time() < deadline:
                status, task_payload = _request(
                    f"{base}/tasks/{task_id}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                _assert(status, 200, task_payload, "task poll")
                if task_payload.get("status") in {"succeeded", "failed"}:
                    break
                time.sleep(2)
            if task_payload.get("status") != "succeeded":
                raise RuntimeError(f"task did not succeed: {task_payload}")

        print("[5/7] verify documents")
        status, docs_payload = _request(
            f"{base}/documents?q=smoke",
            headers={"Authorization": f"Bearer {token}"},
        )
        docs = _assert(status, 200, docs_payload, "documents list")["documents"]
        if not docs:
            raise RuntimeError("documents list is empty after upload")
        doc_id = docs[0]["doc_id"]
        status, doc_payload = _request(
            f"{base}/documents/{doc_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        _assert(status, 200, doc_payload, "document detail")
    else:
        print("[3/7] upload skipped")

    print("[6/7] verify history flow")
    status, payload = _json_request(
        f"{base}/history/{username}/save",
        method="POST",
        token=token,
        payload={
            "session_id": session_id,
            "title": "smoke session",
            "messages": [{"role": "user", "content": "smoke question"}],
        },
    )
    _assert(status, 200, payload, "history save")
    status, payload = _request(
        f"{base}/history/{username}?q=smoke",
        headers={"Authorization": f"Bearer {token}"},
    )
    history = _assert(status, 200, payload, "history list")["history"]
    if not history:
        raise RuntimeError("history search returned no sessions")

    print("[7/7] verify memory profile")
    status, payload = _request(
        f"{base}/memory/profile",
        headers={"Authorization": f"Bearer {token}"},
    )
    _assert(status, 200, payload, "memory profile")

    print("smoke passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
