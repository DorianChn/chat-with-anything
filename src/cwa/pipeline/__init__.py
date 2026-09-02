"""RAG 流水线编排：切分 → 向量化 → 建索引。"""

from __future__ import annotations

from typing import List, Optional

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.storage_context import StorageContext

from ..config import Config
from ..store import build_store


def _configure_embeddings(cfg: Config) -> None:
    """配置 embedding：默认本地模型，可离线；也可切在线。"""
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        Settings.embed_model = HuggingFaceEmbedding(model_name=cfg.embed_model)
    except Exception:
        # 本地模型下载失败时，不在此处崩溃，交由上层提示
        raise


def build_index(
    documents: List[Document],
    cfg: Config,
    index_id: Optional[str] = None,
) -> str:
    """把文档列表构建成向量索引并持久化，返回 index_id。"""
    if not documents:
        raise ValueError("没有加载到任何文档内容")

    _configure_embeddings(cfg)

    Settings.chunk_size = 512
    Settings.chunk_overlap = 64
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)

    vector_store = build_store(cfg, index_id)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    return index_id or vector_store.collection_name
