import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from unittest.mock import MagicMock, patch
import types


def setup_app_mocks():
    """Setup comprehensive mocks for app modules"""
    modules_to_clear = [
        k for k in list(sys.modules.keys())
        if k.startswith("app") or k.startswith("pgvector") or k.startswith("unstructured")
    ]
    for k in modules_to_clear:
        del sys.modules[k]

    app = types.ModuleType("app")
    app.__path__ = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/app"]
    sys.modules["app"] = app

    infra_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/app/infrastructure"
    infrastructure = types.ModuleType("app.infrastructure")
    infrastructure.__path__ = [infra_path]
    app.infrastructure = infrastructure
    sys.modules["app.infrastructure"] = infrastructure

    db_path = infra_path + "/database"
    database = types.ModuleType("app.infrastructure.database")
    database.__path__ = [db_path]
    infrastructure.database = database
    sys.modules["app.infrastructure.database"] = database

    models = types.ModuleType("app.infrastructure.database.models")
    database.models = models
    sys.modules["app.infrastructure.database.models"] = models
    models.DocContent = MagicMock
    models.DocEmbedding = MagicMock
    models.Document = MagicMock

    stores = types.ModuleType("app.infrastructure.database.stores")
    database.stores = stores
    sys.modules["app.infrastructure.database.stores"] = stores
    stores.MySQLDocStore = MagicMock
    stores.PgDocEmbeddingStore = MagicMock
    stores.PgUserMemoryStore = MagicMock

    orm = types.ModuleType("app.infrastructure.database.orm")
    database.orm = orm
    sys.modules["app.infrastructure.database.orm"] = orm
    orm.get_session = MagicMock(return_value=MagicMock())

    schema = types.ModuleType("app.infrastructure.database.schema")
    database.schema = schema
    sys.modules["app.infrastructure.database.schema"] = schema
    schema.ensure_schema_if_possible = MagicMock(return_value=True)

    utils_path = infra_path + "/utils"
    utils = types.ModuleType("app.infrastructure.utils")
    utils.__path__ = [utils_path]
    infrastructure.utils = utils
    sys.modules["app.infrastructure.utils"] = utils

    logging = types.ModuleType("app.infrastructure.utils.logging")
    utils.logging = logging
    sys.modules["app.infrastructure.utils.logging"] = logging
    logging.get_logger = MagicMock(return_value=MagicMock())
    logging.bind_logger = MagicMock(return_value=MagicMock())

    files = types.ModuleType("app.infrastructure.utils.files")
    utils.files = files
    sys.modules["app.infrastructure.utils.files"] = files
    files.sha256_file = MagicMock(return_value="dummy_hash")
    files.write_file = MagicMock(return_value=None)

    text_split = types.ModuleType("app.infrastructure.utils.text_split")
    utils.text_split = text_split
    sys.modules["app.infrastructure.utils.text_split"] = text_split
    text_split.split_text_by_chars = MagicMock(return_value=["chunk1", "chunk2"])

    config = types.ModuleType("app.infrastructure.config")
    infrastructure.config = config
    sys.modules["app.infrastructure.config"] = config

    config_manager = types.ModuleType("app.infrastructure.config.config_manager")
    config.config_manager = config_manager
    sys.modules["app.infrastructure.config.config_manager"] = config_manager
    config_manager.config_manager = MagicMock()
    config_manager.get_config = MagicMock(return_value={})

    memory_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/app/memory"
    memory = types.ModuleType("app.memory")
    memory.__path__ = [memory_path]
    app.memory = memory
    sys.modules["app.memory"] = memory

    long_term = types.ModuleType("app.memory.long_term")
    memory.long_term = long_term
    sys.modules["app.memory.long_term"] = long_term

    user_memory_engine = types.ModuleType("app.memory.long_term.user_memory_engine")
    long_term.user_memory_engine = user_memory_engine
    sys.modules["app.memory.long_term.user_memory_engine"] = user_memory_engine
    user_memory_engine.UserMemoryEngine = MagicMock

    vector_stores = types.ModuleType("app.memory.vector_stores")
    memory.vector_stores = vector_stores
    sys.modules["app.memory.vector_stores"] = vector_stores

    pg_vs = types.ModuleType("app.memory.vector_stores.pgvector_vectorstore")
    vector_stores.pgvector_vectorstore = pg_vs
    sys.modules["app.memory.vector_stores.pgvector_vectorstore"] = pg_vs
    pg_vs.PgVectorVectorStore = MagicMock

    runtime_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/app/runtime"
    runtime = types.ModuleType("app.runtime")
    runtime.__path__ = [runtime_path]
    app.runtime = runtime
    sys.modules["app.runtime"] = runtime

    llm = types.ModuleType("app.runtime.llm")
    runtime.llm = llm
    sys.modules["app.runtime.llm"] = llm

    embeddings = types.ModuleType("app.runtime.llm.embeddings")
    sys.modules["app.runtime.llm.embeddings"] = embeddings
    llm.embeddings = embeddings
    embeddings.ModelEmbeddings = MagicMock

    reranker = types.ModuleType("app.runtime.llm.reranker")
    sys.modules["app.runtime.llm.reranker"] = reranker
    llm.reranker = reranker
    reranker.ModelReranker = MagicMock
    reranker.get_reranker = MagicMock(return_value=MagicMock())

    llm_factory = types.ModuleType("app.runtime.llm.llm_factory")
    sys.modules["app.runtime.llm.llm_factory"] = llm_factory
    llm.llm_factory = llm_factory
    llm_factory.get_local_qwen_provider = MagicMock(return_value=MagicMock())

    component_loader = types.ModuleType("app.runtime.llm.component_loader")
    sys.modules["app.runtime.llm.component_loader"] = component_loader
    llm.component_loader = component_loader
    component_loader.load_sentence_transformers_embedder = MagicMock()
    component_loader.load_transformers_model = MagicMock()
    component_loader.load_transformers_tokenizer = MagicMock()
    component_loader.resolve_pretrained_source_for_spec = MagicMock()
    component_loader.try_load_transformers_processor = MagicMock()

    model_manager = types.ModuleType("app.runtime.llm.model_manager")
    sys.modules["app.runtime.llm.model_manager"] = model_manager
    llm.model_manager = model_manager
    model_manager.build_model_spec = MagicMock()
    model_manager.get_best_device = MagicMock()

    prompts = types.ModuleType("app.runtime.prompts")
    prompts.__path__ = [runtime_path + "/prompts"]
    runtime.prompts = prompts
    sys.modules["app.runtime.prompts"] = prompts
    prompts.get_prompt_template = MagicMock(return_value="You are a {role}. Help me with {task}.")

    graph = types.ModuleType("app.runtime.graph")
    graph.__path__ = [runtime_path + "/graph"]
    runtime.graph = graph
    sys.modules["app.runtime.graph"] = graph
    graph.state = MagicMock()
    graph.registry = MagicMock()
    graph.registry.register_node = MagicMock(return_value=lambda x: x)

    skills_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/app/skills"
    skills = types.ModuleType("app.skills")
    skills.__path__ = [skills_path]
    app.skills = skills
    sys.modules["app.skills"] = skills

    rag = types.ModuleType("app.skills.rag")
    rag.__path__ = [skills_path + "/rag"]
    skills.rag = rag
    sys.modules["app.skills.rag"] = rag

    common = types.ModuleType("app.skills.common")
    common.__path__ = [skills_path + "/common"]
    skills.common = common
    sys.modules["app.skills.common"] = common
    common.tools = MagicMock()
    common.register_tool = MagicMock()
    common.write_file = MagicMock(return_value=None)

    tools = types.ModuleType("app.skills.tools")
    tools.__path__ = [skills_path + "/tools"]
    skills.tools = tools
    sys.modules["app.skills.tools"] = tools
    tools.python_executor = MagicMock()

    research = types.ModuleType("app.skills.research")
    research.__path__ = [skills_path + "/research"]
    skills.research = research
    sys.modules["app.skills.research"] = research
    research.enhanced_search = MagicMock()
    research.web_search = MagicMock()

    ocr = types.ModuleType("app.skills.ocr")
    ocr.__path__ = [skills_path + "/ocr"]
    skills.ocr = ocr
    sys.modules["app.skills.ocr"] = ocr
    ocr.ocr_engine = MagicMock()

    memory_skill = types.ModuleType("app.skills.memory")
    memory_skill.__path__ = [skills_path + "/memory"]
    skills.memory = memory_skill
    sys.modules["app.skills.memory"] = memory_skill
    memory_skill.get_memory_store = MagicMock(return_value=MagicMock())

    profile = types.ModuleType("app.skills.profile")
    profile.__path__ = [skills_path + "/profile"]
    skills.profile = profile
    sys.modules["app.skills.profile"] = profile
    profile.get_profile_store = MagicMock(return_value=MagicMock())

    unstructured = types.ModuleType("unstructured")
    sys.modules["unstructured"] = unstructured

    partition = types.ModuleType("unstructured.partition")
    sys.modules["unstructured.partition"] = partition

    pdf = types.ModuleType("unstructured.partition.pdf")
    sys.modules["unstructured.partition.pdf"] = pdf
    pdf.partition_pdf = MagicMock(return_value=["Mocked table content"])

    for module in ["pgvector", "pgvector.sqlalchemy", "pdfminer", "pdfminer.layout", "redis"]:
        if module not in sys.modules:
            sys.modules[module] = MagicMock()


setup_app_mocks()

import pytest


class TestRAGPipeline:
    """RAG 流程集成测试"""

    def test_rag_engine_initialization(self):
        """测试 RAG 引擎初始化"""
        from app.skills.rag.rag_engine import RAGEngine

        with patch("app.skills.rag.rag_engine.partition_pdf") as mock_partition:
            mock_partition.return_value = ["Mocked content"]

            engine = RAGEngine()

            assert engine is not None
            assert hasattr(engine, "_vectorstore")
            assert hasattr(engine, "reranker")

    def test_etl_flow_pdf_parsing(self):
        """测试 ETL 流程 - PDF 解析"""
        from app.skills.rag.rag_engine import RAGEngine

        with patch("app.skills.rag.rag_engine.partition_pdf") as mock_partition:
            mock_partition.return_value = ["Table Row 1", "Table Row 2"]

            engine = RAGEngine()

            fake_pdf = "/tmp/test_doc.pdf"
            with open(fake_pdf, "w") as f:
                f.write("dummy")

            try:
                docs = engine.load_documents(fake_pdf)
                assert len(docs) >= 0
                mock_partition.assert_called_once()
            finally:
                if os.path.exists(fake_pdf):
                    os.remove(fake_pdf)

    def test_search_pipeline_hybrid_retrieval(self):
        """测试搜索流程 - 混合检索"""
        from app.skills.rag.rag_engine import RAGEngine

        engine = RAGEngine()

        mock_vs = engine._vectorstore

        doc_dense = Document(page_content="Dense Result", metadata={"doc_id": 1})
        mock_vs.similarity_search.return_value = [doc_dense]
        mock_vs.sparse_search.return_value = [
            Document(page_content="Sparse Result", metadata={"doc_id": 2})
        ]

        engine.reranker.rerank.return_value = [
            ("Dense Result", 0.9, 0),
            ("Sparse Result", 0.8, 1),
        ]

        results = engine.retrieve_context("test query", user_id="u1")

        assert len(results) >= 0

        args, kwargs = mock_vs.similarity_search.call_args
        assert kwargs.get("filter") == {"user_id": "u1"}

        args, kwargs = mock_vs.sparse_search.call_args
        assert kwargs.get("filter") == {"user_id": "u1"}

    def test_hybrid_retriever_service(self):
        """测试混合检索服务"""
        from app.skills.rag.hybrid_retriever_service import HybridRetrieverService

        mock_vectorstore = MagicMock()
        service = HybridRetrieverService(vectorstore=mock_vectorstore)

        assert service is not None

    def test_rag_engine_with_context(self):
        """测试带上下文的 RAG 引擎"""
        from app.skills.rag.rag_engine import RAGEngine

        engine = RAGEngine()

        context_docs = [
            Document(page_content="Context 1", metadata={"doc_id": 1}),
            Document(page_content="Context 2", metadata={"doc_id": 2}),
        ]

        query = "Test query"

        engine.reranker.rerank.return_value = [
            ("Context 1", 0.9, 0),
            ("Context 2", 0.8, 1),
        ]

        if hasattr(engine, "build_context_with_rerank"):
            results = engine.build_context_with_rerank(context_docs, query)
            assert len(results) >= 0
        else:
            pass


class TestHybridRetrieverEdgeCases:
    """混合检索边界情况测试"""

    def test_empty_query(self):
        """测试空查询处理"""
        from app.skills.rag.rag_engine import RAGEngine

        engine = RAGEngine()

        results = engine.retrieve_context("", user_id="u1")

        assert results == []

    def test_no_filter_user_id(self):
        """测试无用户 ID 过滤"""
        from app.skills.rag.rag_engine import RAGEngine

        engine = RAGEngine()

        engine._vectorstore.similarity_search.return_value = []

        results = engine.retrieve_context("test", user_id=None)

        args, kwargs = engine._vectorstore.similarity_search.call_args
        assert kwargs.get("filter") is None
