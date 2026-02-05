
import os
import sys
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.getcwd())

# --- ROBUST MOCKING STRATEGY ---
# 1. Clear any existing 'app' modules from sys.modules to ensure a guaranteed clean slate for mocks
# This prevents "ImportError: cannot import name X from app" if app was partially loaded really
for k in list(sys.modules.keys()):
    if k.startswith("app") or k.startswith("pgvector") or k.startswith("unstructured"):
        del sys.modules[k]

# 2. Mock External Heavy Dependencies
sys.modules["pgvector"] = MagicMock()
sys.modules["pgvector.sqlalchemy"] = MagicMock()
sys.modules["unstructured"] = MagicMock()
sys.modules["unstructured.partition"] = MagicMock()
sys.modules["unstructured.partition.pdf"] = MagicMock()
sys.modules["pdfminer"] = MagicMock()
sys.modules["pdfminer.layout"] = MagicMock()

import types

# 3. Mock Application Subsystems
# We use MagicMock for modules.
# We DO NOT mock 'app' itself in sys.modules, we let Python create the namespace package if needed,
# OR we mock the leaf modules.
# However, to be safe against import traversal:
# We will mock the intermediate packages too.

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

# Mock utils as a PACKAGE so submodules works
utils = types.ModuleType("app.infrastructure.utils")
utils.__path__ = [] # Critical for package behavior
infrastructure.utils = utils
sys.modules["app.infrastructure.utils"] = utils

# Explicitly mock utils logger because it's imported
logging = MagicMock()
utils.logging = logging
sys.modules["app.infrastructure.utils.logging"] = logging

# Explicitly mock utils files
files = MagicMock()
utils.files = files
sys.modules["app.infrastructure.utils.files"] = files
files.sha256_file = MagicMock(return_value="dummy_hash")

# Mock text_split
text_split = MagicMock()
utils.text_split = text_split
sys.modules["app.infrastructure.utils.text_split"] = text_split
text_split.split_text_by_chars = MagicMock(return_value=["chunk1", "chunk2"])

# Mock config
config = MagicMock()
infrastructure.config = config
sys.modules["app.infrastructure.config"] = config
config_manager = MagicMock()
config.config_manager = config_manager
sys.modules["app.infrastructure.config.config_manager"] = config_manager


# Memory
memory = MagicMock()
app.memory = memory
sys.modules["app.memory"] = memory
vector_stores = MagicMock()
memory.vector_stores = vector_stores
sys.modules["app.memory.vector_stores"] = vector_stores

# Specific vector store mock
pg_vs = MagicMock()
vector_stores.pgvector_vectorstore = pg_vs
sys.modules["app.memory.vector_stores.pgvector_vectorstore"] = pg_vs
# The class inside the module
pg_vs.PgVectorVectorStore = MagicMock

# Runtime
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

# Skills
skills = MagicMock()
app.skills = skills
sys.modules["app.skills"] = skills
ocr = MagicMock()
skills.ocr = ocr
sys.modules["app.skills.ocr"] = ocr
ocr_engine = MagicMock()
ocr.ocr_engine = ocr_engine
sys.modules["app.skills.ocr.ocr_engine"] = ocr_engine

# --- BEHAVIOR SETUP ---
schema.ensure_schema_if_possible.return_value = True

# --- IMPORT TARGET ---
# Now we import RAGEngine. 
# BUT wait. RAGEngine is a real file we want to test.
# If we mocked `app.skills`, then `from app.skills.rag.rag_engine import ...` will just return a attribute of the mock.
# IT WILL NOT READ THE FILE.
# So we CANNOT mock `app.skills` or `app` if we want to import the real file via `app.skills.rag`.

# CORRECT STRATEGY FOR TESTING A MODULE WHEN PARENTS ARE MOCKED:
# We must NOT mock `app.skills` or `app.skills.rag`.
# We MUST mock everything else.
# So we remove `sys.modules['app']`, `sys.modules['app.skills']` from our manual setting above.
# We also remove `sys.modules['app.skills.ocr']`? No, we need that mocked.
# We can Mock `app.skills.ocr` without mocking `app.skills`.
pass

# REWIND Logic (simplified for safety):
# We needed to ensure 'app' and 'app.skills' take traffic from disk, but 'app.infrastructure' etc take from mocks.

# Safely remove 'app' and 'app.skills' if they exist in sys.modules, so they reload from disk/real pkg
if "app" in sys.modules:
    del sys.modules["app"]
if "app.skills" in sys.modules:
    del sys.modules["app.skills"]

# However, we mocked submodules like `app.skills.ocr`.
# If we delete `app.skills`, these might be orphaned or reloaded?
# No, `sys.modules['app.skills.ocr']` remains.
# When `app.skills` is re-imported from disk, it will see `app.skills.ocr` is already in sys.modules and use it?
# Yes, usually.

# Ensure submodules we want mocked ARE in sys.modules
sys.modules["app.skills.ocr"] = ocr
sys.modules["app.skills.ocr.ocr_engine"] = ocr_engine

# Also ensure app.infrastructure IS in sys.modules (we set it above).
# But if `app` is NOT in sys.modules, can we import `app.infrastructure`?
# Yes, if we import `app` from disk, and then it looks for `infrastructure`...
# WAIT. If `app` loads from disk, it might not know `infrastructure` is already in sys.modules if `app` isn't a simple namespace.
# However, if we blindly import `from app.skills.rag.rag_engine import ...`,
# Python imports `app`. (Loads real app).
# Then `app.skills`. (Loads real app.skills).
# Then `app.skills.rag`. (Loads real app.skills.rag).
# Then `rag_engine`.
# INSIDE rag_engine: `from app.infrastructure.database.models import ...`
# Python imports `app`. (Already loaded).
# Then `app.infrastructure`. (Checks sys.modules... FOUND OUR MOCK!).
# So it uses our mock.
# THIS SHOULD WORK.

# The previous error "ImportError: cannot import name 'infrastructure' from 'app'"
# happens if `app` package (real) does not expose `infrastructure` (which is a Mock in sys.modules).
# If `app` is a regular package, `import app` creates a module object.
# `import app.infrastructure` should just work if it's in sys.modules?
# Usually yes.

# Let's try.

from app.skills.rag.rag_engine import RAGEngine
from langchain_core.documents import Document

def test_etl_flow():
    print("\n>>> Testing ETL Flow (PDF Parsing) ...")
    
    with patch("app.skills.rag.rag_engine.partition_pdf") as mock_partition:
        mock_partition.return_value = ["Table Row 1", "Table Row 2"]
        
        engine = RAGEngine()
        
        fake_pdf = "test_doc.pdf"
        with open(fake_pdf, "w") as f: f.write("dummy")
        
        try:
            docs = engine.load_documents(fake_pdf)
            print(f"Loaded {len(docs)} documents.")
            content = docs[0].page_content
            if "Table Row 1" in content:
                print("SUCCESS: Unstructured partition_pdf was used.")
            else:
                print("FAILURE: Unstructured partition_pdf was NOT used.")
        finally:
            if os.path.exists(fake_pdf): os.remove(fake_pdf)

def test_search_pipeline():
    print("\n>>> Testing Search Pipeline (Hybrid + Rerank) ...")
    
    engine = RAGEngine()
    
    # Configure Mock VectorStore
    mock_vs = engine._vectorstore # Instance of mocked PgVectorVectorStore
    
    doc_dense = Document(page_content="Dense Result", metadata={"doc_id":1})
    mock_vs.similarity_search.return_value = [doc_dense]
    mock_vs.sparse_search.return_value = [Document(page_content="Sparse Result", metadata={"doc_id":2})]
    
    # Configure Mock Reranker
    engine.reranker.rerank.return_value = [("Dense Result", 0.9, 0), ("Sparse Result", 0.8, 1)]
    
    results = engine.retrieve_context("test", user_id="u1")
    print(f"Retrieved {len(results)} results.")
    
    # Verify
    print("Verifying Filters...")
    args, kwargs = mock_vs.similarity_search.call_args
    if kwargs.get("filter") == {"user_id": "u1"}:
        print("SUCCESS: Dense search filter correct.")
    else:
        print(f"FAILURE: Dense search filter: {kwargs}")

    args, kwargs = mock_vs.sparse_search.call_args
    if kwargs.get("filter") == {"user_id": "u1"}:
        print("SUCCESS: Sparse search filter correct.")
    else:
        print(f"FAILURE: Sparse search filter: {kwargs}")

if __name__ == "__main__":
    try:
        test_etl_flow()
        test_search_pipeline()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"TEST FAILED: {e}")
