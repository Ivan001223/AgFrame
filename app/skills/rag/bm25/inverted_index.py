import math
from collections import defaultdict
from typing import Dict, List, Tuple

from app.skills.rag.bm25.tokenizer import Tokenizer


class InvertedIndex:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.term_dict: Dict[str, int] = {}
        self.posting_lists: List[List[Tuple[int, int]]] = []
        self.doc_count: int = 0
        self.avg_doc_len: float = 0.0
        self.doc_lens: Dict[int, int] = {}
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    def build(self, documents: List[str]) -> None:
        self.doc_count = len(documents)
        term_doc_freq: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        for doc_id, doc in enumerate(documents):
            tokens = self.tokenizer.tokenize(doc)
            self.doc_lens[doc_id] = len(tokens)

            seen_terms = set()
            for term in tokens:
                term_doc_freq[term][doc_id] += 1
                if term not in seen_terms:
                    self.df[term] = self.df.get(term, 0) + 1
                    seen_terms.add(term)

        self._build_term_dict(term_doc_freq)
        self._compute_idf()
        self.avg_doc_len = sum(self.doc_lens.values()) / max(1, self.doc_count)

    def _build_term_dict(self, term_doc_freq: Dict[str, Dict[int, int]]) -> None:
        for term, doc_freqs in term_doc_freq.items():
            term_id = len(self.term_dict)
            self.term_dict[term] = term_id
            posting_list = [(doc_id, tf) for doc_id, tf in doc_freqs.items()]
            self.posting_lists.append(posting_list)

    def _compute_idf(self) -> None:
        for term, df in self.df.items():
            self.idf[term] = math.log(
                (self.doc_count - df + 0.5) / (df + 0.5) + 1
            )

    def get_postings(self, term: str) -> List[Tuple[int, int]]:
        term_id = self.term_dict.get(term)
        if term_id is None:
            return []
        return self.posting_lists[term_id]
