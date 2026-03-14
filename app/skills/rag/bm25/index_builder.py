import json
import os
from typing import List, Optional

from app.skills.rag.bm25.tokenizer import Tokenizer
from app.skills.rag.bm25.inverted_index import InvertedIndex
from app.skills.rag.bm25.bm25_scorer import BM25Scorer


class IndexBuilder:
    def __init__(self, tokenizer: Tokenizer, persist_path: Optional[str] = None):
        self.tokenizer = tokenizer
        self.index = InvertedIndex(tokenizer)
        self.scorer = BM25Scorer(self.index)
        self.documents: List[str] = []
        self.persist_path = persist_path

    def build(self, documents: List[str]) -> None:
        self.documents = documents
        self.index.build(documents)

    def save(self) -> None:
        if not self.persist_path:
            return

        os.makedirs(self.persist_path, exist_ok=True)
        data = {
            "version": 1,
            "doc_count": self.index.doc_count,
            "avg_doc_len": self.index.avg_doc_len,
            "term_dict": self.index.term_dict,
            "df": self.index.df,
            "idf": self.index.idf,
            "documents": self.documents,
            "posting_lists": self.index.posting_lists,
            "doc_lens": self.index.doc_lens,
        }

        path = os.path.join(self.persist_path, "bm25_index.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, persist_path: str) -> None:
        path = os.path.join(persist_path, "bm25_index.json")
        if not os.path.exists(path):
            return

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self.persist_path = persist_path
        self.documents = data.get("documents", [])
        self.index.doc_count = data.get("doc_count", 0)
        self.index.avg_doc_len = data.get("avg_doc_len", 0.0)
        self.index.term_dict = data.get("term_dict", {})
        self.index.df = data.get("df", {})
        self.index.idf = data.get("idf", {})
        self.index.posting_lists = data.get("posting_lists", [])
        self.index.doc_lens = data.get("doc_lens", {})
