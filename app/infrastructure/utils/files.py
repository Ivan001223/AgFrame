import hashlib


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    """
    计算文件的 SHA-256 哈希值。
    
    Args:
        path: 文件路径
        chunk_size: 读取块大小 (默认 1MB)
        
    Returns:
        str: 十六进制哈希字符串
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

