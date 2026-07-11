from typing import List
import re

from langchain_core.documents import Document

from trip_pilot.rag.vector_store import get_vector_store


def search_travel_docs(query: str, k: int = 4) -> List[Document]:
    """从本地旅行知识库中检索相关内容。"""
    vector_store = get_vector_store()
    candidates = vector_store.similarity_search(query, k=max(k * 4, 12))
    return _rerank_by_query_keywords(query, candidates)[:k]


def print_search_results(query: str, k: int = 4) -> None:
    """命令行调试用：打印检索结果。"""
    print(f"检索问题：{query}")
    docs = search_travel_docs(query, k=k)

    for i, doc in enumerate(docs, start=1):
        file_name = doc.metadata.get("file_name", "unknown")
        chunk_index = doc.metadata.get("chunk_index", "")
        print(f"\n--- 结果 {i} | {file_name} | chunk {chunk_index} ---")
        section_title = doc.metadata.get("section_title", "")
        if section_title:
            print(f"主题：{section_title}")
        print(doc.page_content[:500])


def _rerank_by_query_keywords(query: str, docs: List[Document]) -> List[Document]:
    keywords = _extract_query_keywords(query)
    if not keywords:
        return docs

    def score(doc: Document) -> int:
        text = doc.page_content
        title = str(doc.metadata.get("section_title", ""))
        value = 0
        for keyword in keywords:
            if keyword in title:
                value += 4
            if keyword in text:
                value += 2
        if any(word in text for word in ["旅行社", "旅游集散", "电话：", "地址：杭州市文三路"]):
            value -= 3
        if any(word in title for word in ["线路", "路线", "景点", "美食", "餐饮"]):
            value += 2
        return value

    return sorted(docs, key=score, reverse=True)


def _extract_query_keywords(query: str) -> List[str]:
    watched = [
        "杭州",
        "西湖",
        "灵隐寺",
        "飞来峰",
        "博物馆",
        "文化",
        "美食",
        "小吃",
        "餐饮",
        "路线",
        "行程",
        "三天",
        "两天",
        "一天",
        "骑行",
        "住宿",
        "交通",
        "赏花",
    ]
    keywords = [word for word in watched if word in query]
    if len(keywords) > 1 and "杭州" in keywords:
        keywords.remove("杭州")
    if not keywords:
        keywords.extend(re.findall(r"[\u4e00-\u9fff]{2,5}", query))
    return list(dict.fromkeys(keywords))


if __name__ == "__main__":
    for query in [
        "杭州三天文化路线怎么安排？",
        "杭州有哪些美食街和本地小吃？",
        "西湖、灵隐寺、博物馆怎么顺路安排？",
    ]:
        print_search_results(query)
