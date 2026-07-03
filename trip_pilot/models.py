from openai import OpenAI

from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

from trip_pilot.config import (
    API_KEY,
    ARK_API_KEY,
    ARK_BASE_URL,
    ARK_IMAGE_MODEL,
    ARK_MULTIMODAL_MODEL,
    BASE_URL,
    MODEL_ID,
)


def get_chat_model(temperature: float = 0.5, max_tokens: int = 2048):
    """主推理模型：用于正式行程生成、Refine 等复杂文本任务。"""
    if not API_KEY or not BASE_URL:
        raise ValueError("请先在 .env 中配置 api_key 和 base_url")

    return init_chat_model(
        model=MODEL_ID,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_ark_model(temperature: float = 0.19, max_tokens: int = 1200):
    """火山 Ark 模型：适合意图识别、参数抽取、轻量聊天和多模态扩展。"""
    if not ARK_API_KEY:
        return None

    return ChatOpenAI(
        model=ARK_MULTIMODAL_MODEL,
        api_key=ARK_API_KEY,
        base_url=ARK_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_fast_model():
    """快速模型入口：优先用豆包 mini，缺少 Ark key 时自动回退 DeepSeek。"""
    ark_model = get_ark_model(temperature=0.2, max_tokens=1200)
    if ark_model:
        return ark_model
    return get_chat_model(temperature=0.25, max_tokens=1200)


def get_multimodal_model():
    """多模态模型入口：后续 UI 上传图片、票据、截图时使用。"""
    ark_model = get_ark_model(temperature=0.4, max_tokens=1600)
    if not ark_model:
        raise ValueError("请先在 .env 中配置 api_key_ark")
    return ark_model


def get_reflection_model():
    """质检模型入口：使用 DeepSeek 低温输出，保证评价稳定。"""
    return get_chat_model(temperature=0.16, max_tokens=1800)


def get_image_client() -> OpenAI:
    """Seedream 图片生成客户端，按 OpenAI-compatible 方式调用 Ark。"""
    if not ARK_API_KEY:
        raise ValueError("请先在 .env 中配置 api_key_ark")
    return OpenAI(api_key=ARK_API_KEY, base_url=ARK_BASE_URL)


def get_image_model_name() -> str:
    return ARK_IMAGE_MODEL
