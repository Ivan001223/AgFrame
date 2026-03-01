from app.skills.rag.bm25.inverted_index import InvertedIndex


class BM25Scorer:
    def __init__(self, index: InvertedIndex, k1: float = 1.5, b: float = 0.75):
        self.index = index
        self.k1 = k1
        self.b = b

    def score(self, doc_id: int, query: str) -> float:
        query_terms = self.index.tokenizer.tokenize(query)
        if not query_terms:
            return 0.0

        doc_len = self.index.doc_lens.get(doc_id, 0)
        avgdl = self.index.avg_doc_len or 1.0

        score = 0.0
        for term in query_terms:
            if term not in self.index.term_dict:
                continue

            tf = 0
            for d, tffreq in self.index.get_postings(term):
                if d == doc_id:
                    tf = tffreq
                    break

            idf = self.index.idf.get(term, 0)

            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / avgdl)

            score += idf * numerator / denominator if denominator > 0 else 0

        return score
