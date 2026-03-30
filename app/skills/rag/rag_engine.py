import os
import subprocess
import tempfile
from dataclasses import replace
from importlib import import_module
from shutil import which
from typing import Any

os.environ.setdefault("OBJC_PRINT_DUPLICATE_CLASSES", "NO")

from langchain_community.document_loaders import (
    Docx2txtLoader,
    TextLoader,
    UnstructuredExcelLoader,
)
from langchain_core.documents import Document
from pypdf import PdfReader

from app.infrastructure.config.settings import settings

from app.infrastructure.database.models import (
    DocContent,
    DocEmbedding,
)
from app.infrastructure.database.models import (
    Document as DocumentRow,
)
from app.infrastructure.database.orm import get_session
from app.infrastructure.database.schema import ensure_schema_if_possible
from app.infrastructure.database.stores import MySQLDocStore, PgDocEmbeddingStore
from app.infrastructure.utils.files import sha256_file
from app.infrastructure.utils.logging import get_logger
from app.infrastructure.utils.text_split import split_text_by_chars
from app.memory.vector_stores.pgvector_vectorstore import PgVectorVectorStore

from app.runtime.llm.embeddings import get_embeddings
from app.skills.rag.hybrid_retriever_service import (
    HybridRetrievalConfig,
    HybridRetrieverService,
)

logger = get_logger("rag_engine")


def _partition_pdf(
    *,
    filename: str,
    infer_table_structure: bool,
    strategy: str,
    languages: list[str] | None = None,
):
    # On macOS, unstructured's import chain may load both cv2 and av. Suppress the
    # duplicate Objective-C class noise before the first import.
    os.environ.setdefault("OBJC_PRINT_DUPLICATE_CLASSES", "NO")
    try:
        partition_pdf = import_module("unstructured.partition.pdf").partition_pdf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "缺少可选依赖 'unstructured'。如需高精度 PDF 解析，请执行 "
            "`uv sync --group document-ai`。"
        ) from exc
    return partition_pdf(
        filename=filename,
        infer_table_structure=infer_table_structure,
        strategy=strategy,
        languages=languages or ["eng"],
    )


def _tesseract_available() -> bool:
    return which("tesseract") is not None


def _poppler_available() -> bool:
    return which("pdfinfo") is not None


def _ensure_mpl_config_dir() -> None:
    if os.getenv("MPLCONFIGDIR"):
        return
    mpl_dir = os.path.join(tempfile.gettempdir(), "agframe-matplotlib")
    os.makedirs(mpl_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = mpl_dir


def _extract_pdf_text_with_pypdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    page_texts = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n\n".join([page for page in page_texts if page])


def _run_tesseract(image_path: str) -> str:
    try:
        completed = subprocess.run(
            ["tesseract", image_path, "stdout"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return ""
    if completed.returncode != 0:
        logger.warning(f"tesseract 识别失败: {completed.stderr.strip()}")
        return ""
    return completed.stdout.strip()


def _cleanup_temp_files(paths: list[str]) -> None:
    for path in paths:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError as exc:
                logger.debug(f"Failed to remove temp file {path}: {exc}")


def _extract_image_text_with_tesseract(file_path: str) -> str:
    return _run_tesseract(file_path)


def _extract_pdf_text_with_tesseract(file_path: str) -> str:
    from pdf2image import convert_from_path

    images = convert_from_path(file_path, dpi=200)
    temp_paths: list[str] = []
    try:
        for image in images:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image.save(tmp.name, format="PNG")
                temp_paths.append(tmp.name)
        page_texts = [_extract_image_text_with_tesseract(path) for path in temp_paths]
        return "\n\n".join(text for text in page_texts if text)
    finally:
        _cleanup_temp_files(temp_paths)


def _get_ocr_engine():
    from app.skills.ocr.ocr_engine import ocr_engine

    return ocr_engine


def load_documents_from_path(file_path: str) -> list[Document]:
    """
    根据文件扩展名加载文档内容。
    支持 PDF/图片 (OCR), DOCX, XLSX, MD, TXT。
    """
    docs: list[Document] = []
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        text = ""
        try:
            text = _extract_pdf_text_with_pypdf(file_path)
        except Exception as pdf_error:
            logger.warning(f"pypdf 提取失败，尝试本地 tesseract OCR: {pdf_error}")
            text = ""

        if not text and _tesseract_available() and _poppler_available():
            logger.info(f"PDF 未提取到可复制文本，使用本地 tesseract OCR 处理：{file_path}...")
            try:
                text = _extract_pdf_text_with_tesseract(file_path)
            except Exception as ocr_error:
                logger.warning(f"本地 tesseract OCR 解析失败，降级为本地 OCR: {ocr_error}")
                text = ""
        elif not text and not _tesseract_available():
            logger.info(f"未检测到 tesseract，跳过本地 tesseract OCR：{file_path}")
        elif not text and not _poppler_available():
            logger.info(f"未检测到 poppler(pdfinfo)，跳过本地 tesseract OCR：{file_path}")

        if text:
            docs = [Document(page_content=text, metadata={"source": file_path})]
        else:
            text = _get_ocr_engine().process_file(file_path)
            if text:
                docs = [Document(page_content=text, metadata={"source": file_path})]
            else:
                logger.warning(f"OCR 未从 {file_path} 提取到文本")
                docs = []

    elif ext in [".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"]:
        logger.info(f"正在处理图片文本提取：{file_path}...")
        text = _extract_image_text_with_tesseract(file_path) if _tesseract_available() else ""
        if not text:
            text = _get_ocr_engine().process_file(file_path)
        if text:
            docs = [Document(page_content=text, metadata={"source": file_path})]
        else:
            logger.warning(f"OCR 未从 {file_path} 提取到文本")
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


def extract_text_from_file(file_path: str) -> str:
    docs = load_documents_from_path(file_path)
    return "\n\n".join(
        doc.page_content.strip()
        for doc in docs
        if str(getattr(doc, "page_content", "")).strip()
    )


class RAGEngine:
    """
    RAG (Retrieval-Augmented Generation) 引擎核心类。
    负责管理文档的摄取、切片、向量化存储以及检索增强。
    支持多种文件格式，并集成了 OCR 能力。
    """

    def __init__(self):
        logger.info("Initializing RAG engine...")
        self.embeddings = get_embeddings()

        self._vectorstore = None
        self._hybrid_retriever: HybridRetrieverService | None = None
        if ensure_schema_if_possible():
            self._vectorstore = PgVectorVectorStore(embeddings=self.embeddings)
            self._hybrid_retriever = HybridRetrieverService(
                vectorstore=self._vectorstore
            )

    def _get_hybrid_config(self) -> HybridRetrievalConfig:
        retrieval_cfg = settings.rag.retrieval
        return HybridRetrievalConfig(
            mode=retrieval_cfg.mode,
            dense_k=retrieval_cfg.dense_k,
            sparse_k=retrieval_cfg.sparse_k,
            candidate_k=retrieval_cfg.candidate_k,
            rrf_k=retrieval_cfg.rrf_k,
            weights=tuple(retrieval_cfg.weights),
        )

    def _success(self, **extra: Any) -> dict[str, Any]:
        payload = {"ok": True}
        payload.update(extra)
        return payload

    def _failure(self, code: str, message: str, **extra: Any) -> dict[str, Any]:
        payload = {
            "ok": False,
            "error_code": code,
            "error_message": message,
        }
        payload.update(extra)
        return payload

    def load_documents(self, file_path: str) -> list[Document]:
        """
        根据文件扩展名加载文档内容。
        支持 PDF/图片 (OCR), DOCX, XLSX, MD, TXT。

        Args:
            file_path: 文件绝对路径

        Returns:
            List[Document]: 加载的文档对象列表
        """
        return load_documents_from_path(file_path)

    def add_knowledge_base(self, file_path: str, user_id: str | None = None) -> dict[str, Any]:
        """
        将文件摄取到知识库中。

        流程：
        1. 加载文档（文本或 OCR）。
        2. 如果数据库可用，使用 Parent Retrieval 策略：
           - 将大块父文档存入 MySQL。
           - 将子切片存入 pgvector 向量库。
        3. 如果数据库不可用，返回错误。
        4. 持久化向量索引。

        Args:
            file_path: 文件路径
            user_id: 用户 ID (用于多租户隔离)

        Returns:
            dict[str, Any]: 结构化摄取结果
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"未找到文件: {file_path}")

        try:
            # 1. 加载文档
            try:
                docs = self.load_documents(file_path)
            except Exception as e:
                logger.error(f"加载文件 {file_path} 错误: {e}")
                return self._failure("document_load_failed", str(e), stage="load")

            if not docs:
                return self._failure("no_text_extracted", "未从文档中提取到文本", stage="load")

            use_parent_retrieval = ensure_schema_if_possible()
            splits: list[Document] = []
            if not use_parent_retrieval:
                logger.warning("未检测到可用数据库，无法写入 pgvector")
                return self._failure("database_not_ready", "数据库不可用，无法写入向量索引", stage="database")

            doc_store = MySQLDocStore()
            checksum = sha256_file(file_path)
            # 传入 user_id 写入 Document 表
            doc_id = doc_store.upsert_document(
                source_path=file_path, checksum=checksum, user_id=user_id
            )

            parent_chunks: list[dict[str, Any]] = []
            for d in docs:
                parent_parts = split_text_by_chars(
                    d.page_content, chunk_size=6000, overlap=400
                )
                for p in parent_parts:
                    parent_chunks.append(
                        {"content": p, "page_num": d.metadata.get("page")}
                    )

            doc_store.delete_parent_chunks(doc_id)
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

            if not splits:
                return self._failure("no_chunks_generated", "文档切片为空，无法建立索引", stage="chunk")

            # ... (rest of logic) ...

            PgDocEmbeddingStore().delete_by_doc_id(doc_id)

            # 批量处理向量嵌入，避免大量文档时内存溢出
            BATCH_SIZE = 100
            rows: list[dict[str, Any]] = []
            for i in range(0, len(splits), BATCH_SIZE):
                batch = splits[i:i + BATCH_SIZE]
                batch_contents = [d.page_content for d in batch]
                try:
                    batch_vectors = self.embeddings.embed_documents(batch_contents)
                except Exception as embedding_error:
                    logger.error(f"批量向量化失败：{embedding_error}")
                    return self._failure("embedding_failed", str(embedding_error), stage="embedding")

                for d, v in zip(batch, batch_vectors):
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
            try:
                PgDocEmbeddingStore().add_embeddings(rows)
            except Exception as store_error:
                logger.error(f"写入向量存储失败：{store_error}")
                return self._failure("vectorstore_write_failed", str(store_error), stage="vectorstore")

            if self._vectorstore is None:
                try:
                    self._vectorstore = PgVectorVectorStore(embeddings=self.embeddings)
                    self._hybrid_retriever = HybridRetrieverService(
                        vectorstore=self._vectorstore
                    )
                except Exception as vector_error:
                    logger.error(f"初始化向量存储失败：{vector_error}")
                    self._vectorstore = None
                    self._hybrid_retriever = None

            logger.info(f"成功添加了来自 {file_path} 的 {len(splits)} 个块")
            return self._success(stage="done", chunks=len(splits), doc_count=len(docs))
        except Exception as e:
            logger.error(f"添加到向量存储失败：{e}")
            return self._failure("ingest_failed", str(e), stage="ingest")

    def retrieve_candidates(
        self, query: str, *, fetch_k: int = 20, user_id: str = None
    ) -> list[Document]:
        if self._vectorstore is None:
            return []
        cfg = self._get_hybrid_config()
        filter_dict = {"user_id": user_id} if user_id else None

        if self._hybrid_retriever is None:
            self._hybrid_retriever = HybridRetrieverService(
                vectorstore=self._vectorstore
            )
        candidate_k = max(1, int(fetch_k or cfg.candidate_k))
        cfg = replace(
            cfg,
            candidate_k=candidate_k,
            dense_k=max(cfg.dense_k, candidate_k),
            sparse_k=max(cfg.sparse_k, candidate_k),
        )

        return self._hybrid_retriever.retrieve_candidates(
            query, config=cfg, filter=filter_dict
        )

    def retrieve_context(
        self, query: str, k: int = 3, fetch_k: int = 20, user_id: str = None
    ) -> list[Document]:
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
            return self.restore_parents(candidates, k=k)
        except Exception as e:
            logger.error(f"检索上下文错误: {e}")
            return []

    def restore_parents(self, docs: list[Document], *, k: int) -> list[Document]:
        if not docs:
            return []
        use_parent_retrieval = ensure_schema_if_possible()
        if not use_parent_retrieval:
            return docs[:k]

        parent_scores: dict[int, float] = {}
        parent_order: list[int] = []
        fallback_docs: list[Document] = []

        for doc in docs:
            meta = dict(getattr(doc, "metadata", {}) or {})
            score = float(
                meta.get("rerank_score")
                or meta.get("retrieval_rrf_score")
                or meta.get("bm25_score")
                or 0.0
            )
            parent_id = meta.get("parent_chunk_id")
            if parent_id is None:
                fallback_docs.append(doc)
                continue
            try:
                parent_id_int = int(parent_id)
            except (ValueError, TypeError) as e:
                logger.debug(f"Failed to parse parent_id {parent_id}: {e}")
                fallback_docs.append(doc)
                continue
            if parent_id_int not in parent_scores:
                parent_scores[parent_id_int] = score
                parent_order.append(parent_id_int)
            else:
                parent_scores[parent_id_int] = max(parent_scores[parent_id_int], score)

        out: list[Document] = list(fallback_docs)
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
                                "retrieval_rrf_score": parent_scores.get(parent_id),
                            },
                        )
                    )
            except Exception as e:
                logger.error(f"获取父文档失败，降级返回原切片: {e}")

        out.sort(
            key=lambda x: x.metadata.get("rerank_score")
            or x.metadata.get("retrieval_rrf_score")
            or x.metadata.get("bm25_score")
            or 0,
            reverse=True,
        )
        return out[:k]



    def clear(self) -> None:
        """
        清除向量存储（危险操作！）。
        删除 pgvector 中的向量记录并重置内存中的实例。
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
            logger.error(f"清空向量库失败：{e}")


_rag_engine: RAGEngine | None = None


def get_rag_engine() -> RAGEngine:
    """获取 RAGEngine 单例"""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine
