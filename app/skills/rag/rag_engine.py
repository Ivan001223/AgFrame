import os
from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import (
    TextLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
)
from langchain_community.document_loaders import UnstructuredFileLoader
from unstructured.partition.pdf import partition_pdf
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document

# 自定义本地模型
from app.runtime.llm.embeddings import ModelEmbeddings
from app.runtime.llm.reranker import ModelReranker
from app.skills.ocr.ocr_engine import ocr_engine
from app.infrastructure.database.schema import ensure_schema_if_possible
from app.infrastructure.database.orm import get_session
from app.infrastructure.database.models import (
    DocContent,
    DocEmbedding,
    Document as DocumentRow,
)
from app.infrastructure.database.stores import MySQLDocStore, PgDocEmbeddingStore
from app.infrastructure.config.config_manager import config_manager
from app.skills.rag.hybrid_retriever_service import (
    HybridRetrieverService,
    HybridRetrievalConfig,
)
from app.memory.vector_stores.pgvector_vectorstore import PgVectorVectorStore
from app.infrastructure.utils.files import sha256_file
from app.infrastructure.utils.text_split import split_text_by_chars


class RAGEngine:
    """
    RAG (Retrieval-Augmented Generation) 引擎核心类。
    负责管理文档的摄取、切片、向量化存储以及检索增强。
    支持多种文件格式，并集成了 OCR 能力。
    """

    def __init__(self):
        # 初始化 Embeddings（本地模型）
        print("正在初始化 RAG 引擎（本地向量模型）...")
        self.embeddings = ModelEmbeddings()

        # 初始化重排器（Reranker）
        self.reranker = ModelReranker()

        self._vectorstore = None
        self._hybrid_retriever: Optional[HybridRetrieverService] = None
        if ensure_schema_if_possible():
            self._vectorstore = PgVectorVectorStore(embeddings=self.embeddings)
            self._hybrid_retriever = HybridRetrieverService(
                vectorstore=self._vectorstore
            )

    def _get_hybrid_config(self) -> HybridRetrievalConfig:
        # TODO: 从配置中心通过 config_manager 读取
        # 暂时返回默认配置
        return HybridRetrievalConfig()

    def load_documents(self, file_path: str) -> List[Document]:
        """
        根据文件扩展名加载文档内容。
        支持 PDF/图片 (OCR), DOCX, XLSX, MD, TXT。

        Args:
            file_path: 文件绝对路径

        Returns:
            List[Document]: 加载的文档对象列表
        """
        docs: List[Document] = []
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            print(f"正在使用 Unstructured 处理 PDF：{file_path}...")
            try:
                # 尝试使用 unstructured 进行高级解析（支持表格）
                elements = partition_pdf(
                    filename=file_path,
                    infer_table_structure=True,
                    strategy="hi_res",
                )
                text = "\n\n".join([str(e) for e in elements])
                docs = [Document(page_content=text, metadata={"source": file_path})]
            except Exception as e:
                print(f"Unstructured 解析失败，降级为 OCR: {e}")
                # 降级到原有的 OCR 逻辑
                text = ocr_engine.process_file(file_path)
                if text:
                    docs = [Document(page_content=text, metadata={"source": file_path})]
                else:
                    print(f"警告：OCR 未从 {file_path} 提取到文本")
                    docs = []

        elif ext in [".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"]:
            print(f"正在使用本地 OCR 处理：{file_path}...")
            text = ocr_engine.process_file(file_path)
            if text:
                docs = [Document(page_content=text, metadata={"source": file_path})]
            else:
                print(f"警告：OCR 未从 {file_path} 提取到文本")
                docs = []
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
        elif ext == ".xlsx":
            loader = UnstructuredExcelLoader(file_path)
            docs = loader.load()
        elif ext == ".md":
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
        else:
            raise ValueError(f"不支持的文件类型: {ext}")

        return docs

    def add_knowledge_base(self, file_path: str, user_id: str = None):
        """
        将文件摄取到知识库中。

        流程：
        1. 加载文档（文本或 OCR）。
        2. 如果数据库可用，使用 Parent Retrieval 策略：
           - 将大块父文档存入 MySQL。
           - 将子切片存入 FAISS 向量库。
        3. 如果数据库不可用，降级为普通向量存储模式（语义切分）。
        4. 持久化向量索引。

        Args:
            file_path: 文件路径
            user_id: 用户 ID (用于多租户隔离)

        Returns:
            bool: 是否成功添加
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"未找到文件: {file_path}")

        try:
            # 1. 加载文档
            try:
                docs = self.load_documents(file_path)
            except Exception as e:
                print(f"加载文件 {file_path} 错误: {e}")
                return False

            if not docs:
                return False

            use_parent_retrieval = ensure_schema_if_possible()
            splits: List[Document] = []
            if not use_parent_retrieval:
                print("未检测到可用数据库，无法写入 pgvector")
                return False

            doc_store = MySQLDocStore()
            checksum = sha256_file(file_path)
            # 传入 user_id 写入 Document 表
            doc_id = doc_store.upsert_document(
                source_path=file_path, checksum=checksum, user_id=user_id
            )

            parent_chunks: List[Dict[str, Any]] = []
            for d in docs:
                parent_parts = split_text_by_chars(
                    d.page_content, chunk_size=6000, overlap=400
                )
                for p in parent_parts:
                    parent_chunks.append(
                        {"content": p, "page_num": d.metadata.get("page")}
                    )

            parent_ids = doc_store.insert_parent_chunks(doc_id, parent_chunks)
            for parent_id, parent in zip(parent_ids, parent_chunks):
                child_parts = split_text_by_chars(
                    parent["content"], chunk_size=1400, overlap=120
                )
                for idx, cp in enumerate(child_parts):
                    splits.append(
                        Document(
                            page_content=cp,
                            metadata={
                                "type": "doc_fragment",
                                "doc_id": doc_id,
                                "parent_chunk_id": parent_id,
                                "child_index": idx,
                                "source": file_path,
                                "user_id": user_id or "",  # 写入 vector metadata
                            },
                        )
                    )

            # ... (rest of logic) ...

            PgDocEmbeddingStore().delete_by_doc_id(doc_id)
            vectors = self.embeddings.embed_documents([d.page_content for d in splits])
            rows: List[Dict[str, Any]] = []
            for d, v in zip(splits, vectors):
                meta = dict(getattr(d, "metadata", {}) or {})
                rows.append(
                    {
                        "doc_id": meta.get("doc_id"),
                        "parent_chunk_id": meta.get("parent_chunk_id"),
                        "child_index": meta.get("child_index"),
                        "source_path": meta.get("source"),
                        "content": d.page_content,
                        "embedding": v,
                        "metadata_json": meta,
                    }
                )
            PgDocEmbeddingStore().add_embeddings(rows)

            if self._vectorstore is None:
                self._vectorstore = PgVectorVectorStore(embeddings=self.embeddings)
                self._hybrid_retriever = HybridRetrieverService(
                    vectorstore=self._vectorstore
                )

            print(f"成功添加了来自 {file_path} 的 {len(splits)} 个块")
            return True
        except Exception as e:
            print(f"添加到向量存储失败：{e}")
            return False

    def retrieve_candidates(
        self, query: str, *, fetch_k: int = 20, user_id: str = None
    ) -> List[Document]:
        if self._vectorstore is None:
            return []
        cfg = self._get_hybrid_config()
        # TODO: Pass filter to config or directly to retrieve_candidates
        # HybridRetrievalConfig doesn't seem to support filter yet,
        # but PgVectorVectorStore.similarity_search does.
        # HybridRetrieverService needs update to accept filter.

        # We need to pass filter to retrieve_candidates
        filter_dict = {"user_id": user_id} if user_id else None

        if self._hybrid_retriever is None:
            self._hybrid_retriever = HybridRetrieverService(
                vectorstore=self._vectorstore
            )

        return self._hybrid_retriever.retrieve_candidates(
            query, config=cfg, filter=filter_dict
        )

    def retrieve_context(
        self, query: str, k: int = 3, fetch_k: int = 20, user_id: str = None
    ) -> List[Document]:
        """
        检索查询的前 k 个相关文档。
        ...
        Args:
            user_id: 用户 ID 用于隔离
        ...
        """
        try:
            candidates = self.retrieve_candidates(
                query, fetch_k=fetch_k, user_id=user_id
            )
            if not candidates:
                return []
            print(f"正在对 {len(candidates)} 条候选文档进行重排...")
            reranked = self.rerank_candidates(query, candidates, k=k)
            return self.restore_parents(reranked, k=k)
        except Exception as e:
            print(f"检索上下文错误: {e}")
            return []

    def rerank_candidates(
        self, query: str, candidates: List[Document], *, k: int
    ) -> List[Document]:
        if not candidates or k <= 0:
            return []
        candidate_texts = [doc.page_content for doc in candidates]
        reranked_results = self.reranker.rerank(query, candidate_texts, top_k=k)
        out: List[Document] = []
        for _, score, idx in reranked_results:
            doc = candidates[idx]
            meta = dict(getattr(doc, "metadata", {}) or {})
            meta["rerank_score"] = score
            doc.metadata = meta
            out.append(doc)
        return out

    def restore_parents(self, docs: List[Document], *, k: int) -> List[Document]:
        if not docs:
            return []
        use_parent_retrieval = ensure_schema_if_possible()
        if not use_parent_retrieval:
            return docs[:k]

        parent_scores: Dict[int, float] = {}
        parent_order: List[int] = []
        fallback_docs: List[Document] = []

        for doc in docs:
            meta = dict(getattr(doc, "metadata", {}) or {})
            score = float(meta.get("rerank_score") or 0.0)
            parent_id = meta.get("parent_chunk_id")
            if parent_id is None:
                fallback_docs.append(doc)
                continue
            try:
                parent_id_int = int(parent_id)
            except Exception:
                fallback_docs.append(doc)
                continue
            if parent_id_int not in parent_scores:
                parent_scores[parent_id_int] = score
                parent_order.append(parent_id_int)
            else:
                parent_scores[parent_id_int] = max(parent_scores[parent_id_int], score)

        out: List[Document] = list(fallback_docs)
        if parent_order:
            parent_order = parent_order[:k]
            doc_store = MySQLDocStore()
            try:
                parents = doc_store.fetch_parent_chunks(parent_order)
                for p in parents:
                    parent_id = int(p["parent_chunk_id"])
                    out.append(
                        Document(
                            page_content=p["content"],
                            metadata={
                                "type": "doc_parent",
                                "doc_id": int(p["doc_id"]),
                                "parent_chunk_id": parent_id,
                                "page_num": p.get("page_num"),
                                "rerank_score": parent_scores.get(parent_id),
                            },
                        )
                    )
            except Exception as e:
                print(f"获取父文档失败，降级返回原切片: {e}")

        out.sort(key=lambda x: x.metadata.get("rerank_score", 0), reverse=True)
        return out[:k]



    def clear(self):
        """
        清除向量存储（危险操作！）。
        删除磁盘上的 FAISS 索引文件并重置内存中的实例。
        """
        try:
            if not ensure_schema_if_possible():
                self._vectorstore = None
                return
            with get_session() as session:
                session.execute(DocEmbedding.__table__.delete())
                session.execute(DocContent.__table__.delete())
                session.execute(DocumentRow.__table__.delete())
            self._vectorstore = PgVectorVectorStore(embeddings=self.embeddings)
            self._hybrid_retriever = HybridRetrieverService(
                vectorstore=self._vectorstore
            )
        except Exception as e:
            print(f"清空向量库失败：{e}")


_rag_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    """获取 RAGEngine 单例"""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine
