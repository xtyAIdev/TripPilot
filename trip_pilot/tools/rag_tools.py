from langchain.tools import tool

from trip_pilot.rag.retriever import search_travel_docs


@tool
def search_travel_knowledge(query: str) -> str:
    """检索本地旅行知识库，适合查询城市景点、经典路线、美食街区和注意事项。"""
    try:
        docs = search_travel_docs(query, k=4)
        if not docs:
            return "本地旅行知识库没有检索到相关内容。"

        results = []
        for i, doc in enumerate(docs, start=1):
            file_name = doc.metadata.get("file_name", "unknown")
            chunk_index = doc.metadata.get("chunk_index", "")
            content = doc.page_content[:500].replace("\n\n", "\n")
            results.append(f"[{i}] 来源：{file_name} chunk:{chunk_index}\n{content}")

        return "\n\n".join(results)
    except Exception as e:
        return f"检索本地旅行知识库失败：{e}"

