from trip_pilot.config import RAW_DATA_DIR
from trip_pilot.rag.loaders import load_markdown_documents, split_documents
from trip_pilot.rag.vector_store import get_vector_store


def build_index(reset: bool = True) -> None:
    """构建旅行知识库向量索引。"""
    print("开始构建 Agent-TripPilot RAG 向量库")
    docs = load_markdown_documents(RAW_DATA_DIR)
    chunks = split_documents(docs)

    if not chunks:
        print("没有读取到文档块，构建终止")
        return

    vector_store = get_vector_store()
    if reset:
        print("重建向量集合，避免重复写入旧文档")
        vector_store.reset_collection()

    ids = [
        f"{chunk.metadata.get('file_name', 'doc')}-{chunk.metadata.get('chunk_index', i)}"
        for i, chunk in enumerate(chunks)
    ]
    vector_store.add_documents(chunks, ids=ids)
    print("向量库构建完成")


if __name__ == "__main__":
    build_index()
