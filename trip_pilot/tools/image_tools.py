import base64
from pathlib import Path

import httpx
from langchain.tools import tool
from openai import OpenAIError

from trip_pilot.config import ARK_IMAGE_SIZE, OUTPUT_DIR
from trip_pilot.models import get_image_client, get_image_model_name


def _safe_file_name(file_name: str) -> str:
    stem = Path(file_name).stem or "trip_cover"
    clean = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)
    return f"{clean}.png"


@tool
def generate_trip_cover_image(prompt: str, file_name: str = "trip_cover.png") -> str:
    """使用 Doubao-Seedream 为行程生成封面图，返回本地文件路径或图片 URL。"""
    try:
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / _safe_file_name(file_name)

        client = get_image_client()
        response = client.images.generate(
            model=get_image_model_name(),
            prompt=prompt,
            size=ARK_IMAGE_SIZE,
        )

        image_data = response.data[0]
        b64_json = getattr(image_data, "b64_json", None)
        url = getattr(image_data, "url", None)

        if b64_json:
            output_path.write_bytes(base64.b64decode(b64_json))
            return f"已生成图片：{output_path}"

        if url:
            try:
                image_bytes = httpx.get(url, timeout=60).content
                output_path.write_bytes(image_bytes)
                return f"已生成图片：{output_path}"
            except Exception:
                url_path = output_path.with_suffix(".txt")
                url_path.write_text(url, encoding="utf-8")
                return f"图片接口返回 URL，下载失败，已保存 URL：{url_path}"

        return "图片生成接口没有返回 b64_json 或 url，请检查 Ark 图片模型接口返回格式。"
    except OpenAIError as e:
        message = str(e)
        model_name = get_image_model_name()
        if "ModelNotOpen" in message:
            return (
                "图片生成失败：当前火山 Ark 账号尚未开通图片模型服务。\n"
                f"当前配置的 ARK_IMAGE_MODEL={model_name}。\n"
                "请在 Ark 控制台开通该模型，或把 .env 里的 ARK_IMAGE_MODEL 改成你已经开通的图片模型 ID。"
            )
        if "model" in message.lower() and "not" in message.lower():
            return (
                f"图片生成失败：Ark 图片模型不可用。当前 ARK_IMAGE_MODEL={model_name}。\n"
                "请确认模型 ID 是否正确、是否已开通、是否在当前地域可用。"
            )
        if "image size" in message.lower() or "size" in message.lower():
            return (
                f"图片生成失败：图片尺寸参数不符合 Ark 要求。当前 ARK_IMAGE_SIZE={ARK_IMAGE_SIZE}。\n"
                "Seedream 4.5 要求像素数至少 3686400，建议使用 1920x1920 或更高。"
            )
        return f"图片生成失败：Ark 接口返回错误：{e}"
    except Exception as e:
        return f"图片生成失败：{e}"
