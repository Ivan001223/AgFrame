import os

from app.skills.rag.bm25.index_builder import IndexBuilder
from app.skills.rag.bm25.tokenizer import Tokenizer


def test_index_builder_persist(tmp_path):
    builder = IndexBuilder(Tokenizer(), persist_path=str(tmp_path))
    docs = ["hello world", "python programming"]

    builder.build(docs)
    builder.save()

    assert os.path.exists(os.path.join(tmp_path, "bm25_index.json"))


def test_index_builder_load(tmp_path):
    builder = IndexBuilder(Tokenizer(), persist_path=str(tmp_path))
    docs = ["hello world", "python programming"]

    builder.build(docs)
    builder.save()

    builder2 = IndexBuilder(Tokenizer(), persist_path=str(tmp_path))
    builder2.load(str(tmp_path))

    assert builder2.index.doc_count == 2
    assert builder2.documents == docs
    assert "hello" in builder2.index.term_dict


def test_index_builder_load_nonexistent():
    builder = IndexBuilder(Tokenizer(), persist_path="/nonexistent/path")
    builder.load("/nonexistent/path")

    assert builder.index.doc_count == 0
    assert builder.documents == []
