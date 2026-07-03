from pathlib import Path
from typing import List

from langchain_core.documents import Document


def load_markdown_documents(raw_dir: str) -> List[Document]:
    """读取 data/raw 下的 Markdown 文件。"""
    raw_path = Path(raw_dir)
    docs: List[Document] = []

    for file_path in sorted(raw_path.glob("*.md","*.pdf", "*.docx")):
        if file_path.name == "source_manifest.md":
            continue

        text = file_path.read_text(encoding="utf-8")
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(file_path),
                    "file_name": file_path.name,
                },
            )
        )

    print(f"已读取文档数量：{len(docs)}")

    return docs


def split_documents(
    docs: List[Document],
    chunk_size: int = 600,
    chunk_overlap: int = 80,
) -> List[Document]:
    """简单文本切分，先保证 RAG 流程清晰可控。"""
    chunks: List[Document] = []

    for doc in docs:
        text = doc.page_content.strip()
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                metadata = dict(doc.metadata)
                metadata["chunk_index"] = chunk_index
                chunks.append(Document(page_content=chunk_text, metadata=metadata))

            chunk_index += 1
            if end >= len(text):
                break
            start = end - chunk_overlap

    print(f"切分后的文档块数量：{len(chunks)}")
    return chunks
