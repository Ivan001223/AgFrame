from typing import List


def split_text_by_chars(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        return [text]
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)
    parts: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        parts.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return [p for p in parts if p.strip()]

