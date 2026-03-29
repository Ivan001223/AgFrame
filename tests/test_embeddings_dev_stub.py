from __future__ import annotations

from app.runtime.llm.embeddings import ModelEmbeddings


def test_dev_stub_embeddings_are_deterministic_and_dimensioned():
    client = ModelEmbeddings(
        config={
            "embeddings": {
                "model_name": "dev-stub",
                "normalize": True,
            },
            "local_models": {
                "embedding_model": "",
            },
            "feature_flags": {
                "pgvector_dimension": 1024,
            },
            "model_manager": {
                "provider": "hf",
            },
        }
    )

    doc_vectors = client.embed_documents(["alpha", "alpha", "beta"])
    query_vector = client.embed_query("alpha")

    assert len(doc_vectors) == 3
    assert len(doc_vectors[0]) == 1024
    assert doc_vectors[0] == doc_vectors[1]
    assert doc_vectors[0] != doc_vectors[2]
    assert query_vector == doc_vectors[0]
