import re
from typing import List

import jieba


class Tokenizer:
    def __init__(self, language: str = "mixed"):
        self.language = language

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []

        text = text.lower()

        if self.language == "en":
            words = re.findall(r"\b[a-zA-Z]+\b", text)
        elif self.language == "zh":
            words = list(jieba.cut(text))
        else:
            en_words = re.findall(r"\b[a-zA-Z]+\b", text)
            zh_words = list(jieba.cut(text.replace(" ", "")))
            words = en_words + zh_words

        return [w for w in words if len(w) > 1]
