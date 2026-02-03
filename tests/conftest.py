import os
import sys
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def mock_external_deps():
    mocks = {}

    for module in [
        "pgvector", "pgvector.sqlalchemy",
        "unstructured", "unstructured.partition",
        "unstructured.partition.pdf",
        "pdfminer", "pdfminer.layout",
    ]:
        sys.modules[module] = MagicMock()
        mocks[module] = sys.modules[module]

    yield mocks


@pytest.fixture
def mock_app_modules(mock_external_deps):
    modules_to_clear = [
        k for k in sys.modules.keys()
        if k.startswith("app") or k.startswith("pgvector") or k.startswith("unstructured")
    ]
    for k in modules_to_clear:
        del sys.modules[k]

    app = MagicMock()
    sys.modules["app"] = app

    infrastructure = MagicMock()
    app.infrastructure = infrastructure
    sys.modules["app.infrastructure"] = infrastructure

    database = MagicMock()
    infrastructure.database = database
    sys.modules["app.infrastructure.database"] = database

    models = MagicMock()
    database.models = models
    sys.modules["app.infrastructure.database.models"] = models

    stores = MagicMock()
    database.stores = stores
    sys.modules["app.infrastructure.database.stores"] = stores

    orm = MagicMock()
    database.orm = orm
    sys.modules["app.infrastructure.database.orm"] = orm

    schema = MagicMock()
    database.schema = schema
    sys.modules["app.infrastructure.database.schema"] = schema
    schema.ensure_schema_if_possible.return_value = True

    utils = MagicMock()
    infrastructure.utils = utils
    sys.modules["app.infrastructure.utils"] = utils

    logging = MagicMock()
    utils.logging = logging
    sys.modules["app.infrastructure.utils.logging"] = logging

    files = MagicMock()
    utils.files = files
    sys.modules["app.infrastructure.utils.files"] = files
    files.sha256_file = MagicMock(return_value="dummy_hash")

    text_split = MagicMock()
    utils.text_split = text_split
    sys.modules["app.infrastructure.utils.text_split"] = text_split
    text_split.split_text_by_chars = MagicMock(return_value=["chunk1", "chunk2"])

    config = MagicMock()
    infrastructure.config = config
    sys.modules["app.infrastructure.config"] = config

    memory = MagicMock()
    app.memory = memory
    sys.modules["app.memory"] = memory

    vector_stores = MagicMock()
    memory.vector_stores = vector_stores
    sys.modules["app.memory.vector_stores"] = vector_stores

    pg_vs = MagicMock()
    vector_stores.pgvector_vectorstore = pg_vs
    sys.modules["app.memory.vector_stores.pgvector_vectorstore"] = pg_vs

    runtime = MagicMock()
    app.runtime = runtime
    sys.modules["app.runtime"] = runtime

    llm = MagicMock()
    runtime.llm = llm
    sys.modules["app.runtime.llm"] = llm

    embeddings = MagicMock()
    llm.embeddings = embeddings
    sys.modules["app.runtime.llm.embeddings"] = embeddings

    reranker = MagicMock()
    llm.reranker = reranker
    sys.modules["app.runtime.llm.reranker"] = reranker

    llm_factory = MagicMock()
    llm.llm_factory = llm_factory
    sys.modules["app.runtime.llm.llm_factory"] = llm_factory

    skills = MagicMock()
    app.skills = skills
    sys.modules["app.skills"] = skills

    ocr = MagicMock()
    skills.ocr = ocr
    sys.modules["app.skills.ocr"] = ocr

    yield {
        "app": app,
        "infrastructure": infrastructure,
        "database": database,
        "models": models,
        "stores": stores,
        "runtime": runtime,
        "llm": llm,
    }


@pytest.fixture
def sample_user_id():
    return "test_user_001"


@pytest.fixture
def sample_session_id():
    return "test_session_001"


@pytest.fixture
def sample_conversation_history():
    return [
        {"role": "user", "content": "你好，我想了解机器学习"},
        {"role": "assistant", "content": "当然可以，请问你想了解哪个方面？"},
    ]


@pytest.fixture
def sample_golden_cases():
    return [
        {
            "id": "case_001",
            "input": "查找关于 Python 异步编程的文档",
            "expected_tool": "rag",
            "expected_keywords": ["Python", "异步", "async"],
            "description": "RAG 查询场景 - 应该触发文档检索",
        },
        {
            "id": "case_002",
            "input": "今天的新闻有哪些？",
            "expected_tool": "web_search",
            "expected_keywords": ["新闻", "最新"],
            "description": "实时查询场景 - 应该触发网络搜索",
        },
        {
            "id": "case_003",
            "input": "帮我写一个快速排序算法",
            "expected_tool": "python_executor",
            "expected_keywords": ["排序", "算法", "代码"],
            "description": "代码生成场景 - 应该触发代码执行",
        },
        {
            "id": "case_004",
            "input": "分析这张图片的内容",
            "expected_tool": "ocr",
            "expected_keywords": ["图片", "图像"],
            "description": "图像分析场景 - 应该触发 OCR",
        },
        {
            "id": "case_005",
            "input": "我们之前讨论过什么？",
            "expected_tool": "memory",
            "expected_keywords": ["之前", "讨论", "记忆"],
            "description": "记忆检索场景 - 应该触发记忆查询",
        },
    ]
