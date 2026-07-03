import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")


def get_env(name: str, default: str = "") -> str:
    """读取环境变量，方便后续统一管理配置。"""
    return os.getenv(name, default)


BGE_MODEL_PATH = get_env("BGE_MODEL_PATH", r"D:\models\bge-small-zh-v1.5")
CHROMA_DIR = get_env("CHROMA_DIR", str(BASE_DIR / "data" / "chroma_db"))
RAW_DATA_DIR = str(BASE_DIR / "data" / "raw")
COLLECTION_NAME = get_env("COLLECTION_NAME", "trip_pilot_travel_docs")

API_KEY = get_env("api_key")
BASE_URL = get_env("base_url")
MODEL_ID = get_env("model_id", "deepseek-chat")

# 火山 Ark 采用 OpenAI-compatible 接口。api_key_ark 放在 .env 中，不写入代码和文档。
ARK_API_KEY = get_env("api_key_ark")
ARK_BASE_URL = get_env("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
ARK_MULTIMODAL_MODEL = get_env("ARK_MULTIMODAL_MODEL", "doubao-seed-2-0-mini-260428")
ARK_IMAGE_MODEL = get_env("ARK_IMAGE_MODEL", "doubao-seedream-4-5-251128")
ARK_IMAGE_SIZE = get_env("ARK_IMAGE_SIZE", "1920x1920")

GAODE_MCP_URL = get_env("GAODE_MCP_URL")
GAODE_MCP_TRANSPORT = get_env("GAODE_MCP_TRANSPORT", "streamable_http")
HOTEL_MCP_URL = get_env("HOTEL_MCP_URL")
HOTEL_MCP_TRANSPORT = get_env("HOTEL_MCP_TRANSPORT", "streamable_http")
HOTEL_MCP_AUTH_TOKEN = get_env("HOTEL_MCP_AUTH_TOKEN", get_env("ROLLINGGO_API_KEY"))
HOTEL_MCP_BACKEND = get_env("HOTEL_MCP_BACKEND", "auto")
OUTPUT_DIR = str(BASE_DIR / "outputs")
