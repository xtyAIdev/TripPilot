from pathlib import Path

from langchain.tools import tool

from trip_pilot.config import OUTPUT_DIR


@tool
def export_markdown(file_name: str, content: str) -> str:
    """把行程内容导出为 Markdown 文件。"""
    try:
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not file_name.endswith(".md"):
            file_name = f"{file_name}.md"

        file_path = output_dir / file_name
        file_path.write_text(content, encoding="utf-8")
        return f"已导出 Markdown 文件：{file_path}"
    except Exception as e:
        return f"导出 Markdown 失败：{e}"

