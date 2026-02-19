import argparse
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.infrastructure.database.schema import ensure_schema_if_possible
from app.memory.long_term.user_memory_engine import UserMemoryEngine
from app.runtime.prompts.prompt_builder import PromptBudget, build_system_prompt
from app.skills.profile.profile_engine import UserProfileEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--query", required=True)
    args = parser.parse_args()

    if not ensure_schema_if_possible():
        raise RuntimeError("database not ready")

    user_id = str(args.user_id)
    query = str(args.query)
    profile = UserProfileEngine().get_profile(user_id)
    engine = UserMemoryEngine()

    profile_items = engine.retrieve_profile_items(user_id=user_id, query=query, k=6, fetch_k=30)
    memories = engine.retrieve_chat_summaries(user_id=user_id, query=query, k=3, fetch_k=20)

    prefs = profile.get("preferences") if isinstance(profile, dict) else {}
    if not isinstance(prefs, dict):
        prefs = {}
    pinned = {}
    for key in ["language", "communication_style", "interaction_protocol", "tone_instruction"]:
        if prefs.get(key) is not None:
            pinned[key] = prefs.get(key)

    profile_view = {
        "basic_info": (profile.get("basic_info") if isinstance(profile, dict) else {}) or {},
        "tech_profile": (profile.get("tech_profile") if isinstance(profile, dict) else {}) or {},
        "preferences": pinned,
        "retrieved_profile_items": profile_items,
    }

    system_prompt, _ = build_system_prompt(
        profile=profile_view,
        recent_history_lines=[],
        docs=[],
        memories=memories,
        budget=PromptBudget(),
    )

    print("profile_items:")
    for i, it in enumerate(profile_items, start=1):
        text = str(it.get("text") or "")
        score = it.get("rerank_score") or (it.get("metadata_json") or {}).get("rerank_score")
        print(f"- {i}. {text} score={score}")

    print("\nmemories:")
    for i, m in enumerate(memories, start=1):
        meta = getattr(m, "metadata", {}) or {}
        print(f"- {i}. session_id={meta.get('session_id')} range={meta.get('start_msg_id')}..{meta.get('end_msg_id')}")

    print(f"\nsystem_prompt_chars={len(system_prompt)}")


if __name__ == "__main__":
    main()

