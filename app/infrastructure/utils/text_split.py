

def split_text_by_chars(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    按字符数切分文本，支持重叠。
    
    Args:
        text: 待切分的文本
        chunk_size: 每个切片的最大字符数
        overlap: 相邻切片间的重叠字符数
        
    Returns:
        List[str]: 切分后的文本列表
    """
    if chunk_size <= 0:
        return [text]
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)
    parts: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        parts.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return [p for p in parts if p.strip()]

