import os
import shutil

from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import (
    TextLoader, 
    Docx2txtLoader, 
    UnstructuredExcelLoader
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 自定义本地模型
from app.core.llm.embeddings import ModelEmbeddings
from app.core.llm.reranker import ModelReranker
from app.core.services.ocr_engine import ocr_engine
from app.core.database.schema import ensure_schema_if_possible
from app.core.database.stores import MySQLDocStore
from app.core.config.config_manager import config_manager
from app.core.services.hybrid_retriever_service import HybridRetrieverService, HybridRetrievalConfig
from app.core.utils.faiss_store import load_faiss, save_faiss
from app.core.utils.files import sha256_file
from app.core.utils.text_split import split_text_by_chars

DOC_VECTOR_STORE_PATH = os.path.join("data", "vector_store_docs_child")

class RAGEngine:
    """
    RAG (Retrieval-Augmented Generation) 引擎核心类。
    负责管理文档的摄取、切片、向量化存储以及检索增强。
    支持多种文件格式，并集成了 OCR 能力。
    """
    def __init__(self, persist_directory: str = DOC_VECTOR_STORE_PATH):
        self.persist_directory = persist_directory
        
        # 初始化 Embeddings（本地模型）
        print("正在初始化 RAG 引擎（本地向量模型）...")
        self.embeddings = ModelEmbeddings()
        
        # 初始化重排器（Reranker）
        self.reranker = ModelReranker()
        
        # 初始化向量库（FAISS）
        self._vectorstore = None
        self._hybrid_retriever: Optional[HybridRetrieverService] = None
        if os.path.exists(self.persist_directory) and os.path.exists(
            os.path.join(self.persist_directory, "index.faiss")
        ):
            print(f"正在从 {self.persist_directory} 加载 FAISS 索引")
            flags = (config_manager.get_config() or {}).get("feature_flags", {}) or {}
            self._vectorstore = load_faiss(
                self.persist_directory,
                self.embeddings,
                allow_dangerous_deserialization=bool(flags.get("allow_dangerous_deserialization", False)),
            )
            if self._vectorstore is None:
                print("加载 FAISS 索引失败")
            else:
                self._hybrid_retriever = HybridRetrieverService(vectorstore=self._vectorstore)
        else:
            print("未找到已有的 FAISS 索引，将在首次添加文档时初始化新索引。")
            self._vectorstore = None
            self._hybrid_retriever = None

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

        if ext in [".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"]:
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

    def _persist_vectorstore(self) -> bool:
        """持久化保存 FAISS 索引到磁盘"""
        ok = save_faiss(self.persist_directory, self._vectorstore)
        if not ok:
            print("保存向量存储失败")
        return ok

    def add_knowledge_base(self, file_path: str):
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
            if use_parent_retrieval:
                try:
                    doc_store = MySQLDocStore()
                    checksum = sha256_file(file_path)
                    doc_id = doc_store.upsert_document(source_path=file_path, checksum=checksum)

                    parent_chunks: List[Dict[str, Any]] = []
                    for d in docs:
                        # 父文档切分：较大粒度，保留更多上下文
                        parent_parts = split_text_by_chars(d.page_content, chunk_size=6000, overlap=400)
                        for p in parent_parts:
                            parent_chunks.append({"content": p, "page_num": d.metadata.get("page")})

                    parent_ids = doc_store.insert_parent_chunks(doc_id, parent_chunks)
                    for parent_id, parent in zip(parent_ids, parent_chunks):
                        # 子文档切分：较小粒度，用于精确向量检索
                        child_parts = split_text_by_chars(parent["content"], chunk_size=1400, overlap=120)
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
                                    },
                                )
                            )
                except Exception as e:
                    print(f"数据库操作失败，降级到纯向量模式: {e}")
                    use_parent_retrieval = False
            
            # 如果数据库不可用或操作失败，降级处理
            if not use_parent_retrieval:
                print("未检测到可用数据库或数据库操作失败，将使用纯向量存储模式（无 Parent Retrieval）...")
                print("正在使用语义切分（Semantic Chunking）...")
                text_splitter = SemanticChunker(
                    self.embeddings,
                    breakpoint_threshold_type="percentile",
                )
                splits = text_splitter.split_documents(docs)

            # 3. 添加到向量库
            if self._vectorstore is None:
                self._vectorstore = FAISS.from_documents(splits, self.embeddings)
                self._hybrid_retriever = HybridRetrieverService(vectorstore=self._vectorstore)
            else:
                self._vectorstore.add_documents(documents=splits)

            if not self._persist_vectorstore():
                return False
            print(f"成功添加了来自 {file_path} 的 {len(splits)} 个块")
            return True
        except Exception as e:
            print(f"添加到向量存储失败：{e}")
            return False

    def _get_hybrid_config(self) -> HybridRetrievalConfig:
        cfg = config_manager.get_config() or {}
        rag_cfg = (cfg.get("rag") or {}).get("retrieval") or {}
        mode = str(rag_cfg.get("mode") or "hybrid")
        dense_k = int(rag_cfg.get("dense_k") or 20)
        sparse_k = int(rag_cfg.get("sparse_k") or 20)
        candidate_k = int(rag_cfg.get("candidate_k") or 20)
        rrf_k = int(rag_cfg.get("rrf_k") or 60)
        weights = rag_cfg.get("weights") or [0.5, 0.5]
        try:
            w_sparse = float(weights[0])
            w_dense = float(weights[1])
        except Exception:
            w_sparse, w_dense = 0.5, 0.5
        return HybridRetrievalConfig(
            mode=mode,
            dense_k=dense_k,
            sparse_k=sparse_k,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            weights=(w_sparse, w_dense),
        )

    def retrieve_candidates(self, query: str, *, fetch_k: int = 20) -> List[Document]:
        if self._vectorstore is None:
            return []
        cfg = self._get_hybrid_config()
        cfg = HybridRetrievalConfig(
            mode=cfg.mode,
            dense_k=cfg.dense_k,
            sparse_k=cfg.sparse_k,
            candidate_k=int(fetch_k),
            rrf_k=cfg.rrf_k,
            weights=cfg.weights,
        )
        if self._hybrid_retriever is None:
            self._hybrid_retriever = HybridRetrieverService(vectorstore=self._vectorstore)
        return self._hybrid_retriever.retrieve_candidates(query, config=cfg)

    def rerank_candidates(self, query: str, candidates: List[Document], *, k: int) -> List[Document]:
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

    def retrieve_context(self, query: str, k: int = 3, fetch_k: int = 20) -> List[Document]:
        """
        检索查询的前 k 个相关文档。
        
        流程:
        1. 召回 (Recall): 使用 FAISS 进行相似度搜索，获取 fetch_k 个候选文档。
        2. 重排 (Rerank): 使用 Reranker 模型对候选文档打分，选出前 k 个最相关的。
        3. 还原 (Restore): 如果使用了 Parent Retrieval，返回对应的父文档块以提供更完整的上下文。
        
        Args:
            query: 用户查询
            k: 最终返回的文档数量
            fetch_k: 初步召回的文档数量
            
        Returns:
            List[Document]: 检索到的相关文档列表
        """
        try:
            candidates = self.retrieve_candidates(query, fetch_k=fetch_k)
            if not candidates:
                return []
            print(f"正在对 {len(candidates)} 条候选文档进行重排...")
            reranked = self.rerank_candidates(query, candidates, k=k)
            return self.restore_parents(reranked, k=k)
        except Exception as e:
            print(f"检索上下文错误: {e}")
            return []

    def clear(self):
        """
        清除向量存储（危险操作！）。
        删除磁盘上的 FAISS 索引文件并重置内存中的实例。
        """
        try:
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
            self._vectorstore = None
            print("向量库已清空。")
        except Exception as e:
            print(f"清空向量库失败：{e}")

_rag_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    """获取 RAGEngine 单例"""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine
