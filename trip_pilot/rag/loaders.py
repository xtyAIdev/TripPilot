import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document


SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


def load_markdown_documents(raw_dir: str) -> List[Document]:
    """读取 data/raw 下的旅行资料。

    保留旧函数名，避免影响 build_index 调用；实际支持 Markdown、TXT 和 PDF。
    """
    raw_path = Path(raw_dir)
    docs: List[Document] = []

    for file_path in sorted(raw_path.iterdir()):
        if file_path.name == "source_manifest.md" or file_path.is_dir():
            continue
        if file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        text = _load_file_text(file_path)
        if not text.strip():
            print(f"跳过空文档：{file_path.name}")
            continue

        docs.append(
            Document(
                page_content=_normalize_text(text),
                metadata={
                    "source": str(file_path),
                    "file_name": file_path.name,
                    "file_type": file_path.suffix.lower().lstrip("."),
                },
            )
        )

    print(f"已读取文档数量：{len(docs)}")
    return docs


def split_documents(
    docs: List[Document],
    chunk_size: int = 900,
    chunk_overlap: int = 120,
) -> List[Document]:
    """按标题主题优先切分，超长主题再按段落/句子切分。

    旅行攻略通常围绕“交通/住宿/景点/美食/路线/注意事项”等小标题组织。
    固定字数硬切会把同一主题拆散，检索时容易返回半截信息；这里先识别标题，
    再把 section_title 写入 metadata 和 chunk 内容，提升召回可解释性。
    """
    chunks: List[Document] = []

    for doc in docs:
        sections = _split_into_sections(doc.page_content)
        chunk_index = 0

        for section_index, section in enumerate(sections):
            title = section["title"]
            content = section["content"].strip()
            if not content:
                continue

            section_text = f"{title}\n{content}" if title else content
            for part in _split_long_text(section_text, chunk_size, chunk_overlap):
                metadata = dict(doc.metadata)
                metadata.update(
                    {
                        "chunk_index": chunk_index,
                        "section_index": section_index,
                        "section_title": title or "未命名主题",
                    }
                )
                chunks.append(Document(page_content=part, metadata=metadata))
                chunk_index += 1

    print(f"切分后的文档块数量：{len(chunks)}")
    return chunks


def _load_file_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return file_path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        return _load_pdf_text(file_path)
    return ""


def _load_pdf_text(file_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "当前 raw 目录包含 PDF，但环境未安装 pypdf。"
            "请先执行：pip install pypdf，或把 PDF 转成 Markdown/TXT 后再构建索引。"
        ) from exc

    reader = PdfReader(str(file_path))
    pages = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"第{page_index}页\n{text}")
    return "\n\n".join(pages)


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_into_sections(text: str) -> List[dict]:
    lines = [line.strip() for line in text.splitlines()]
    sections: List[dict] = []
    current_title = "全文概览"
    current_lines: List[str] = []

    for line in lines:
        if _is_noise_line(line):
            continue
        if not line:
            current_lines.append("")
            continue

        if _looks_like_heading(line):
            _append_section(sections, current_title, current_lines)
            current_title = _clean_heading(line)
            current_lines = []
        else:
            current_lines.append(line)

    _append_section(sections, current_title, current_lines)
    return sections or [{"title": "全文概览", "content": text.strip()}]


def _append_section(sections: List[dict], title: str, lines: List[str]) -> None:
    content = "\n".join(lines).strip()
    if content:
        sections.append({"title": title, "content": content})


def _looks_like_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 48:
        return False

    if re.match(r"^#{1,6}\s+.+", line):
        return True
    if re.match(r"^第[一二三四五六七八九十\d]+[章节天日].*", line):
        return True
    if re.match(r"^[一二三四五六七八九十]+[、.．]\s*.+", line):
        return len(line) <= 28 and "地址" not in line
    if re.match(r"^\d{1,2}[、.．]\s*.+", line):
        return len(line) <= 28 and "地址" not in line
    if re.match(r"^\d{1,2}\.\d{1,2}\s*.+", line):
        return len(line) <= 28 and "地址" not in line

    if re.search(r"[，,、：:（）()]|的", line):
        return False

    title_keywords = [
        "交通",
        "住宿",
        "酒店",
        "景点",
        "路线",
        "行程",
        "美食",
        "餐饮",
        "门票",
        "预约",
        "预算",
        "注意事项",
        "推荐",
        "攻略",
        "必去",
        "避坑",
    ]
    no_sentence_end = not re.search(r"[。！？；;]$", line)
    return no_sentence_end and any(keyword in line for keyword in title_keywords)


def _is_noise_line(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    noise_patterns = [
        r"^第\d+页$",
        r"^\d{1,3}$",
        r"^No\.\d+$",
        r"^[A-Za-z\s·]+$",
        r"^[-—_]{2,}$",
        r"^.+\s+\d{1,3}$",
    ]
    return any(re.match(pattern, line) for pattern in noise_patterns)


def _clean_heading(line: str) -> str:
    line = re.sub(r"^#{1,6}\s*", "", line.strip())
    return line.strip(" -—")


def _split_long_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_split_by_sentence(paragraph, chunk_size, chunk_overlap))
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            chunks.append(current.strip())
            overlap_text = current[-chunk_overlap:].strip() if chunk_overlap > 0 else ""
            current = f"{overlap_text}\n\n{paragraph}".strip() if overlap_text else paragraph

    if current:
        chunks.append(current.strip())

    return [chunk for chunk in chunks if chunk]


def _split_by_sentence(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    sentences = [s for s in re.split(r"(?<=[。！？；;])", text) if s.strip()]
    if len(sentences) <= 1:
        return _split_by_window(text, chunk_size, chunk_overlap)

    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current}{sentence}".strip()
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            overlap_text = current[-chunk_overlap:].strip() if chunk_overlap > 0 else ""
            current = f"{overlap_text}{sentence}".strip() if overlap_text else sentence.strip()
    if current:
        chunks.append(current.strip())
    return chunks


def _split_by_window(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)
    return [chunk for chunk in chunks if chunk]
