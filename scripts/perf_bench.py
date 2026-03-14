from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Callable

from langchain_core.documents import Document

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("AUTH_SECRET_KEY", "x" * 64)

from app.infrastructure.utils.security import create_access_token, decode_access_token
from app.infrastructure.utils.text_split import split_text_by_chars
from app.runtime.prompts.prompt_builder import PromptBudget, build_system_prompt


@dataclass(frozen=True)
class BenchResult:
    name: str
    runs: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if p <= 0:
        return values[0]
    if p >= 100:
        return values[-1]
    k = (len(values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    d0 = values[f] * (c - k)
    d1 = values[c] * (k - f)
    return d0 + d1


def bench(name: str, fn: Callable[[], Any], *, warmup: int, runs: int) -> BenchResult:
    for _ in range(max(0, warmup)):
        fn()
    times_ms: list[float] = []
    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
    times_ms.sort()
    return BenchResult(
        name=name,
        runs=len(times_ms),
        mean_ms=float(statistics.mean(times_ms)),
        p50_ms=float(_percentile(times_ms, 50)),
        p95_ms=float(_percentile(times_ms, 95)),
        min_ms=float(times_ms[0]),
        max_ms=float(times_ms[-1]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()

    budget = PromptBudget()
    docs = [
        Document(
            page_content=("x" * 2500),
            metadata={"doc_id": f"doc_{i}", "parent_chunk_id": f"p_{i}", "page_num": i},
        )
        for i in range(1, 4)
    ]
    memories = [
        Document(
            page_content=("y" * 1800),
            metadata={"session_id": "sess_1", "start_msg_id": i * 10, "end_msg_id": i * 10 + 3},
        )
        for i in range(1, 4)
    ]
    history_lines = [f"u: line {i}" for i in range(50)]
    profile = "user_profile:" + ("z" * 5000)

    def _prompt_build():
        build_system_prompt(
            profile=profile,
            recent_history_lines=history_lines,
            docs=docs,
            memories=memories,
            web_search={"query": "q" * 300, "result": "r" * 4000},
            self_correction="c" * 4000,
            budget=budget,
        )

    big_text = ("abcde " * 40000).strip()

    def _split_text():
        split_text_by_chars(big_text, chunk_size=1000, overlap=200)

    def _jwt_roundtrip():
        token = create_access_token({"sub": "bench_user", "role": "user"})
        decode_access_token(token)

    results = [
        bench("prompt_builder.build_system_prompt", _prompt_build, warmup=args.warmup, runs=args.runs),
        bench("text_split.split_text_by_chars", _split_text, warmup=args.warmup, runs=args.runs),
        bench("security.jwt_roundtrip", _jwt_roundtrip, warmup=args.warmup, runs=args.runs),
    ]

    payload: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "results": [asdict(r) for r in results],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
