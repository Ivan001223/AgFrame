import os
import tempfile

import pytest
from app.skills.rag.bm25.tokenizer import Tokenizer
from app.skills.rag.bm25.index_builder import IndexBuilder


def test_index_builder_persist():
    with tempfile.TemporaryDirectory() as tmpdir:
        builder = IndexBuilder(Tokenizer(), persist_path=tmpdir)
        docs = ["hello world", "python programming"]

        builder.build(docs)
        builder.save()

        assert os.path.exists(os.path.join(tmpdir, "bm25_index.json"))


def test_index_builder_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        builder = IndexBuilder(Tokenizer(), persist_path=tmpdir)
        docs = ["hello world", "python programming"]

        builder.build(docs)
        builder.save()

        builder2 = IndexBuilder(Tokenizer(), persist_path=tmpdir)
        builder2.load(tmpdir)

        assert builder2.index.doc_count == 2
        assert builder2.documents == docs
        assert "hello" in builder2.index.term_dict


def test_index_builder_load_nonexistent():
    builder = IndexBuilder(Tokenizer(), persist_path="/nonexistent/path")
    builder.load("/nonexistent/path")

    assert builder.index.doc_count == 0
    assert builder.documents == []
