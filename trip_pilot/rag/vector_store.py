from pathlib import Path

from langchain_chroma import Chroma

from trip_pilot.config import BGE_MODEL_PATH, CHROMA_DIR, COLLECTION_NAME
from trip_pilot.rag.embeddings import LocalBGEEmbeddings


def get_embeddings() -> LocalBGEEmbeddings:
    """创建本地 BGE embedding 对象。"""
    return LocalBGEEmbeddings(model_path=BGE_MODEL_PATH, use_fp16=False)


def get_vector_store() -> Chroma:
    """读取或创建 Chroma 向量库。"""
    embeddings = get_embeddings()
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
