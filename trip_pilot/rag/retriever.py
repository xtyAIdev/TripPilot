from typing import List

from langchain_core.documents import Document

from trip_pilot.rag.vector_store import get_vector_store


def search_travel_docs(query: str, k: int = 4) -> List[Document]:
    """从本地旅行知识库中检索相关内容。"""
    vector_store = get_vector_store()
    return vector_store.similarity_search(query, k=k)


def print_search_results(query: str, k: int = 4) -> None:
    """命令行调试用：打印检索结果。"""
    print(f"检索问题：{query}")
    docs = search_travel_docs(query, k=k)

    for i, doc in enumerate(docs, start=1):
        file_name = doc.metadata.get("file_name", "unknown")
        chunk_index = doc.metadata.get("chunk_index", "")
        print(f"\n--- 结果 {i} | {file_name} | chunk {chunk_index} ---")
        print(doc.page_content[:500])


if __name__ == "__main__":
    print_search_results("杭州文化路线，想去西湖、灵隐寺和博物馆")

