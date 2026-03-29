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


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


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
    parser.add_argument("--skip-chat", action="store_true", help="Skip normal chat invoke/history checks")
    parser.add_argument("--skip-interrupt", action="store_true", help="Skip chat interrupt/approve/resume checks")
    parser.add_argument("--exercise-reject", action="store_true", help="Also verify interrupt reject/resume-blocked behavior")
    args = parser.parse_args()
    if args.exercise_reject and args.skip_interrupt:
        parser.error("--exercise-reject requires interrupt checks to be enabled")

    base = args.base_url.rstrip("/")
    username = args.username
    password = args.password
    session_id = f"smoke-session-{uuid.uuid4().hex[:8]}"
    chat_session_id = f"smoke-chat-{uuid.uuid4().hex[:8]}"
    interrupt_session_id = f"smoke-interrupt-{uuid.uuid4().hex[:8]}"
    reject_session_id = f"smoke-reject-{uuid.uuid4().hex[:8]}"

    step = 1

    def _step(label: str) -> None:
        nonlocal step
        print(f"[{step}] {label}")
        step += 1

    _step(f"register user: {username}")
    status, payload = _json_request(f"{base}/auth/register", method="POST", payload={"username": username, "password": password})
    if status not in {200, 400}:
        raise RuntimeError(f"register failed: status={status}, payload={payload}")
    if status == 400 and payload.get("detail") != "Username already registered":
        raise RuntimeError(f"unexpected register response: {payload}")

    _step("login")
    status, payload = _form_request(f"{base}/auth/token", fields={"username": username, "password": password})
    token = _assert(status, 200, payload, "login")["access_token"]

    if not args.skip_upload:
        _step("upload pdf")
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
            _step("poll task")
            task_id = result["task_id"]
            deadline = time.time() + args.task_timeout
            task_payload = {}
            while time.time() < deadline:
                status, task_payload = _request(
                    f"{base}/tasks/{task_id}",
                    headers=_auth_headers(token),
                )
                _assert(status, 200, task_payload, "task poll")
                if task_payload.get("status") in {"succeeded", "failed"}:
                    break
                time.sleep(2)
            if task_payload.get("status") != "succeeded":
                raise RuntimeError(f"task did not succeed: {task_payload}")

        _step("verify documents")
        status, docs_payload = _request(
            f"{base}/documents?q=smoke",
            headers=_auth_headers(token),
        )
        docs = _assert(status, 200, docs_payload, "documents list")["documents"]
        if not docs:
            raise RuntimeError("documents list is empty after upload")
        doc_id = docs[0]["doc_id"]
        status, doc_payload = _request(
            f"{base}/documents/{doc_id}",
            headers=_auth_headers(token),
        )
        _assert(status, 200, doc_payload, "document detail")
    else:
        _step("upload skipped")

    if not args.skip_chat:
        _step("invoke normal chat workbench flow")
        status, payload = _json_request(
            f"{base}/chat/workbench-invoke",
            method="POST",
            token=token,
            payload={
                "input": {
                    "messages": [{"role": "user", "content": "Reply briefly so the smoke test can verify persistence."}],
                    "context": {
                        "session_id": chat_session_id,
                        "context_focus_hint": "smoke test persistence",
                    },
                },
                "config": {
                    "configurable": {
                        "thread_id": chat_session_id,
                    }
                },
            },
        )
        chat_payload = _assert(status, 200, payload, "chat workbench invoke")
        messages = chat_payload.get("messages") or []
        if len(messages) < 2:
            raise RuntimeError(f"chat invoke did not return persisted conversation: {chat_payload}")

        _step("verify chat history persistence")
        status, payload = _request(
            f"{base}/history/{username}/{chat_session_id}",
            headers=_auth_headers(token),
        )
        session_payload = _assert(status, 200, payload, "chat history detail")
        session_messages = session_payload.get("messages") or []
        if len(session_messages) < 2:
            raise RuntimeError(f"chat history missing persisted assistant reply: {session_payload}")

        _step("verify chat history search")
        status, payload = _request(
            f"{base}/history/{username}?q=smoke",
            headers=_auth_headers(token),
        )
        history = _assert(status, 200, payload, "history list")["history"]
        if not any(item.get("id") == chat_session_id for item in history):
            raise RuntimeError(f"history search did not include chat session: {history}")
    else:
        _step("chat flow skipped")

    if not args.skip_interrupt:
        _step("invoke chat with human approval")
        status, payload = _json_request(
            f"{base}/chat/workbench-invoke",
            method="POST",
            token=token,
            payload={
                "input": {
                    "messages": [{"role": "user", "content": "Please prepare a response that needs approval."}],
                    "context": {
                        "session_id": interrupt_session_id,
                        "require_human_approval": True,
                        "interrupt_action_type": "live_smoke_approval",
                        "interrupt_description": "Approve live smoke response",
                        "interrupt_payload": {"next_step": "generate"},
                    },
                },
                "config": {
                    "configurable": {
                        "thread_id": interrupt_session_id,
                    }
                },
            },
        )
        _assert(status, 200, payload, "chat invoke")

        _step("verify interrupt status")
        status, payload = _request(
            f"{base}/interrupt/{interrupt_session_id}",
            headers=_auth_headers(token),
        )
        interrupt_payload = _assert(status, 200, payload, "interrupt status")
        if not interrupt_payload.get("interrupted"):
            raise RuntimeError(f"expected interrupted session, got payload={interrupt_payload}")

        _step("approve interrupt")
        status, payload = _json_request(
            f"{base}/interrupt/{interrupt_session_id}/approve",
            method="POST",
            token=token,
            payload={"approved": True},
        )
        approve_payload = _assert(status, 200, payload, "interrupt approve")
        if not approve_payload.get("approved"):
            raise RuntimeError(f"interrupt approval was not accepted: {approve_payload}")

        _step("resume approved session")
        status, payload = _json_request(
            f"{base}/interrupt/{interrupt_session_id}/resume",
            method="POST",
            token=token,
            payload={},
        )
        resume_payload = _assert(status, 200, payload, "interrupt resume")
        if not resume_payload.get("resumed"):
            raise RuntimeError(f"resume did not complete successfully: {resume_payload}")

        _step("verify resumed history persistence")
        status, payload = _request(
            f"{base}/history/{username}/{interrupt_session_id}",
            headers=_auth_headers(token),
        )
        session_payload = _assert(status, 200, payload, "resumed history detail")
        messages = session_payload.get("messages") or []
        if len(messages) < 2:
            raise RuntimeError(f"resumed history missing assistant output: {session_payload}")

        if args.exercise_reject:
            _step("invoke chat with rejectable approval")
            status, payload = _json_request(
                f"{base}/chat/workbench-invoke",
                method="POST",
                token=token,
                payload={
                    "input": {
                        "messages": [{"role": "user", "content": "Prepare a response that the smoke test will reject."}],
                        "context": {
                            "session_id": reject_session_id,
                            "require_human_approval": True,
                            "interrupt_action_type": "live_smoke_reject",
                            "interrupt_description": "Reject live smoke response",
                            "interrupt_payload": {"next_step": "generate"},
                        },
                    },
                    "config": {
                        "configurable": {
                            "thread_id": reject_session_id,
                        }
                    },
                },
            )
            _assert(status, 200, payload, "rejectable chat invoke")

            _step("reject interrupt")
            status, payload = _json_request(
                f"{base}/interrupt/{reject_session_id}/approve",
                method="POST",
                token=token,
                payload={"approved": False},
            )
            reject_payload = _assert(status, 200, payload, "interrupt reject")
            if reject_payload.get("approved") is not False:
                raise RuntimeError(f"interrupt reject did not remain rejected: {reject_payload}")

            _step("verify rejected interrupt status")
            status, payload = _request(
                f"{base}/interrupt/{reject_session_id}",
                headers=_auth_headers(token),
            )
            rejected_status = _assert(status, 200, payload, "rejected interrupt status")
            if not rejected_status.get("interrupted"):
                raise RuntimeError(f"rejected interrupt unexpectedly cleared: {rejected_status}")
            action_required = rejected_status.get("action_required") or {}
            if action_required.get("approved") is not False:
                raise RuntimeError(f"rejected interrupt status lost approval decision: {rejected_status}")

            _step("verify rejected resume is blocked")
            status, payload = _request(
                f"{base}/interrupt/{reject_session_id}/resume",
                headers=_auth_headers(token),
            )
            if status != 400:
                raise RuntimeError(f"rejected interrupt resume should be blocked, got status={status}, payload={payload}")

            _step("verify rejected history remains draft")
            status, payload = _request(
                f"{base}/history/{username}/{reject_session_id}",
                headers=_auth_headers(token),
            )
            rejected_history = _assert(status, 200, payload, "rejected history detail")
            rejected_messages = rejected_history.get("messages") or []
            if len(rejected_messages) < 2:
                raise RuntimeError(f"rejected history missing persisted draft: {rejected_history}")
            if rejected_messages[-1].get("role") != "assistant":
                raise RuntimeError(f"rejected history lost assistant draft: {rejected_history}")

    _step("verify memory profile")
    status, payload = _request(
        f"{base}/memory/profile",
        headers=_auth_headers(token),
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
