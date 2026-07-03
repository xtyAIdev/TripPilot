import asyncio
from typing import Dict, List
import json

from trip_pilot.config import (
    GAODE_MCP_TRANSPORT,
    GAODE_MCP_URL,
    HOTEL_MCP_TRANSPORT,
    HOTEL_MCP_URL,
    HOTEL_MCP_AUTH_TOKEN,
)


def _normalize_transport(transport: str) -> str:
    """ModelScope 写 streamable_http，LangChain adapter 使用 http。"""
    if transport == "streamable_http":
        return "http"
    return transport


def get_mcp_server_config() -> Dict[str, Dict[str, str]]:
    """根据 .env 组装 MCP server 配置，不在日志里打印 URL。"""
    servers: Dict[str, Dict[str, str]] = {}

    if GAODE_MCP_URL:
        servers["amap-maps"] = {
            "url": GAODE_MCP_URL,
            "transport": _normalize_transport(GAODE_MCP_TRANSPORT),
        }

    if HOTEL_MCP_URL:
        hotel_config = {
            "url": HOTEL_MCP_URL,
            "transport": _normalize_transport(HOTEL_MCP_TRANSPORT),
        }
        if HOTEL_MCP_AUTH_TOKEN:
            hotel_config["headers"] = {"Authorization": f"Bearer {HOTEL_MCP_AUTH_TOKEN}"}
        servers["hotel"] = hotel_config

    return servers


async def load_mcp_tools_async():
    """加载远程 MCP tools。未安装依赖时给出清晰错误。"""
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as e:
        raise RuntimeError(
            "缺少 MCP 依赖，请先安装：pip install langchain-mcp-adapters mcp"
        ) from e

    servers = get_mcp_server_config()
    if not servers:
        return []

    client = MultiServerMCPClient(servers)
    return await client.get_tools()


def load_mcp_tools():
    """同步入口，方便普通脚本调用。"""
    return asyncio.run(load_mcp_tools_async())


async def list_mcp_tools_async() -> List[str]:
    """列出 MCP 工具名称，调试连接时使用。"""
    tools = await load_mcp_tools_async()
    return [tool.name for tool in tools]


def list_mcp_tools() -> List[str]:
    return asyncio.run(list_mcp_tools_async())


async def call_mcp_tool_async(tool_name: str, args: dict):
    """按名称调用 MCP tool。"""
    tools = await load_mcp_tools_async()
    for tool in tools:
        if tool.name == tool_name:
            return await tool.ainvoke(args)
    raise ValueError(f"没有找到 MCP 工具：{tool_name}")


def call_mcp_tool(tool_name: str, args: dict):
    return asyncio.run(call_mcp_tool_async(tool_name, args))


def extract_mcp_text(result) -> str:
    """把 MCP 返回结果转成文本，兼容 LangChain content block。"""
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict) and "text" in first:
            return first["text"]
    return str(result)


def extract_mcp_json(result) -> dict:
    """从 MCP 文本结果中解析 JSON。"""
    text = extract_mcp_text(result)
    return json.loads(text)
