import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import select

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.infrastructure.config.settings import settings
from app.infrastructure.database.models import UserProfile
from app.infrastructure.database.orm import get_session
from app.infrastructure.database.schema import ensure_schema_if_possible
from app.memory.long_term.user_memory_engine import UserMemoryEngine
from app.memory.vector_stores.faiss_store import load_faiss
from app.runtime.llm.embeddings import ModelEmbeddings


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _migrate_faiss_chat_summaries(engine: UserMemoryEngine) -> dict[str, int]:
    base_dir = os.path.join(os.getcwd(), "data", "vector_store_chat_summary")
    if not os.path.isdir(base_dir):
        return {"users": 0, "docs": 0}

    flags = settings.feature_flags
    emb = ModelEmbeddings()
    users = 0
    total_docs = 0

    for user_id in os.listdir(base_dir):
        user_dir = os.path.join(base_dir, user_id)
        if not os.path.isdir(user_dir):
            continue
        store = load_faiss(
            user_dir,
            emb,
            allow_dangerous_deserialization=flags.allow_dangerous_deserialization,
        )
        if store is None:
            continue
        docstore = getattr(store, "docstore", None)
        docs = list(getattr(docstore, "_dict", {}).values()) if docstore is not None else []
        if not docs:
            continue

        users += 1
        total_docs += len(docs)

        rows: list[dict[str, Any]] = []
        for d in docs:
            text = str(getattr(d, "page_content", "") or "").strip()
            if not text:
                continue
            meta = dict(getattr(d, "metadata", {}) or {})
            session_id = str(meta.get("session_id") or "unknown")
            start_msg_id = meta.get("start_msg_id")
            end_msg_id = meta.get("end_msg_id")
            created_at = meta.get("created_at")
            item_hash = _sha256_hex(f"chat_summary|{user_id}|{session_id}|{start_msg_id}|{end_msg_id}|{text}")
            rows.append(
                {
                    "user_id": str(user_id),
                    "kind": "episodic",
                    "subkind": "chat_summary",
                    "session_id": session_id,
                    "text": text,
                    "item_hash": item_hash,
                    "confidence_score": None,
                    "last_verified_at": int(created_at) if created_at is not None else None,
                    "metadata_json": {
                        "type": "chat_summary",
                        "session_id": session_id,
                        "start_msg_id": start_msg_id,
                        "end_msg_id": end_msg_id,
                        "created_at": int(created_at) if created_at is not None else None,
                        "source": "faiss_migration",
                    },
                }
            )

        if not rows:
            continue
        embeddings = engine.embeddings.embed_documents([r["text"] for r in rows])
        for r, e in zip(rows, embeddings):
            r["embedding"] = e
        engine.store.upsert_items(rows)

    return {"users": users, "docs": total_docs}


def _migrate_profiles(engine: UserMemoryEngine) -> int:
    count = 0
    with get_session() as session:
        rows = session.execute(select(UserProfile)).scalars().all()
        for r in rows:
            try:
                engine.replace_profile_semantic_memory(user_id=str(r.user_id), profile=r.profile_json or {})
                count += 1
            except Exception:
                continue
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-faiss", action="store_true")
    parser.add_argument("--skip-profiles", action="store_true")
    args = parser.parse_args()

    if not ensure_schema_if_possible():
        raise RuntimeError("database not ready")

    engine = UserMemoryEngine()

    if not args.skip_faiss:
        r = _migrate_faiss_chat_summaries(engine)
        print(f"migrated faiss chat summaries: users={r['users']} docs={r['docs']}")

    if not args.skip_profiles:
        n = _migrate_profiles(engine)
        print(f"migrated profiles: users={n}")


if __name__ == "__main__":
    main()

