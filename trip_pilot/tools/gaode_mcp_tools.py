from langchain.tools import tool

from trip_pilot.config import GAODE_MCP_URL
from trip_pilot.mcp_client import call_mcp_tool
from trip_pilot.mcp_client import get_mcp_server_config


@tool
def gaode_route_placeholder(origin: str, destination: str, mode: str = "步行") -> str:
    """高德 MCP 路线规划占位工具，后续替换为真实 MCP 调用。"""
    if not GAODE_MCP_URL:
        return (
            "高德 MCP Remote URL 尚未配置。"
            "当前为占位结果：请在真实接入后查询路线距离、耗时和交通方式。"
        )

    servers = get_mcp_server_config()
    if "amap-maps" not in servers:
        return "高德 MCP 配置读取失败，请检查 .env 中的 GAODE_MCP_URL。"

    return (
        "已检测到高德 MCP 配置。\n"
        f"待安装 MCP 依赖并完成工具发现后，可查询 {origin} 到 {destination} 的{mode}路线。\n"
        "调试命令：python -m trip_pilot.app.list_mcp_tools"
    )


@tool
def check_gaode_mcp_status() -> str:
    """检查高德 MCP 是否已配置。"""
    if not GAODE_MCP_URL:
        return "高德 MCP 未配置。"
    return "高德 MCP 已配置，下一步请运行工具发现脚本确认服务端工具列表。"


@tool
def search_poi_via_gaode(keyword: str, city: str = "") -> str:
    """通过高德 MCP 关键词搜索 POI。"""
    if not GAODE_MCP_URL:
        return "高德 MCP 未配置，无法查询 POI。"

    args = {"keywords": keyword}
    if city:
        args["city"] = city

    try:
        result = call_mcp_tool("maps_text_search", args)
        return str(result)
    except Exception as e:
        return f"高德 POI 查询失败：{e}"


@tool
def get_weather_via_gaode(city: str) -> str:
    """通过高德 MCP 查询城市天气。"""
    if not GAODE_MCP_URL:
        return "高德 MCP 未配置，无法查询天气。"

    try:
        result = call_mcp_tool("maps_weather", {"city": city})
        return str(result)
    except Exception as e:
        return f"高德天气查询失败：{e}"


@tool
def geocode_via_gaode(address: str, city: str = "") -> str:
    """通过高德 MCP 将地址或景点名称解析为经纬度。"""
    if not GAODE_MCP_URL:
        return "高德 MCP 未配置，无法解析经纬度。"

    args = {"address": address}
    if city:
        args["city"] = city

    try:
        result = call_mcp_tool("maps_geo", args)
        return str(result)
    except Exception as e:
        return f"高德地址解析失败：{e}"


@tool
def distance_via_gaode(origins: str, destination: str, distance_type: str = "1") -> str:
    """通过高德 MCP 测量距离。type: 1驾车，0直线，3步行。"""
    if not GAODE_MCP_URL:
        return "高德 MCP 未配置，无法测量距离。"

    try:
        result = call_mcp_tool(
            "maps_distance",
            {
                "origins": origins,
                "destination": destination,
                "type": distance_type,
            },
        )
        return str(result)
    except Exception as e:
        return f"高德距离测量失败：{e}"
