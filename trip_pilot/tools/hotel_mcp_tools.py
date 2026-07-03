import json
from typing import Any

import httpx
from langchain.tools import tool

from trip_pilot.config import HOTEL_MCP_AUTH_TOKEN, HOTEL_MCP_BACKEND, HOTEL_MCP_URL
from trip_pilot.mcp_client import call_mcp_tool
from trip_pilot.mcp_client import extract_mcp_json
from trip_pilot.mcp_client import get_mcp_server_config


def _is_rollinggo_backend() -> bool:
    if HOTEL_MCP_BACKEND.lower() == "rollinggo":
        return True
    return bool(HOTEL_MCP_URL and "rollinggo.cn" in HOTEL_MCP_URL.lower())


def _build_hotel_search_args(
    place: str,
    origin_query: str,
    checkin_date: str = "",
    stay_nights: int = 1,
    adult_count: int = 2,
    place_type: str = "城市",
    size: int = 5,
    max_price_per_night: float | None = None,
    min_price_per_night: float | None = None,
    star_ratings: list[float] | None = None,
) -> dict:
    args: dict[str, Any] = {
        "originQuery": origin_query,
        "place": place,
        "placeType": place_type,
        "countryCode": "CN",
        "size": min(max(int(size), 1), 20),
        "checkInParam": {
            "stayNights": max(int(stay_nights), 1),
            "adultCount": max(int(adult_count), 1),
        },
    }
    if checkin_date:
        args["checkInParam"]["checkInDate"] = checkin_date

    filter_options: dict[str, Any] = {}
    if star_ratings:
        filter_options["starRatings"] = star_ratings
    if max_price_per_night is not None or min_price_per_night is not None:
        price_range = {}
        if min_price_per_night is not None:
            price_range["min"] = float(min_price_per_night)
        if max_price_per_night is not None:
            price_range["max"] = float(max_price_per_night)
        filter_options["priceRange"] = price_range
    if filter_options:
        args["filterOptions"] = filter_options
    return args


def _call_rollinggo_tool(tool_name: str, args: dict) -> dict:
    if not HOTEL_MCP_AUTH_TOKEN:
        raise RuntimeError("RollingGo Hotel MCP 需要配置 HOTEL_MCP_AUTH_TOKEN 或 ROLLINGGO_API_KEY。")

    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": args,
        },
        "id": 1,
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "Authorization": f"Bearer {HOTEL_MCP_AUTH_TOKEN}",
    }
    response = httpx.post(HOTEL_MCP_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def _call_hotel_tool(tool_name: str, args: dict):
    if _is_rollinggo_backend():
        return _call_rollinggo_tool(tool_name, args)
    return call_mcp_tool(tool_name, args)


def _extract_hotel_payload(result) -> dict:
    if isinstance(result, dict):
        if "result" in result:
            content = result["result"].get("content")
            if isinstance(content, list) and content:
                text = content[0].get("text") if isinstance(content[0], dict) else None
                if text:
                    try:
                        return json.loads(text)
                    except Exception:
                        return {"raw_text": text}
            return result["result"]
        return result
    try:
        return extract_mcp_json(result)
    except Exception:
        return {"raw_text": str(result)}


def _iter_dicts(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_dicts(child)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_dicts(item)


def _extract_hotels(data: dict) -> list[dict]:
    candidates = []
    preferred_keys = [
        "hotelInformationList",
        "hotelList",
        "hotels",
        "list",
        "items",
    ]
    for key in preferred_keys:
        value = data.get(key)
        if isinstance(value, list):
            candidates.extend(item for item in value if isinstance(item, dict))
    if candidates:
        return candidates

    hotels = []
    for item in _iter_dicts(data):
        if any(key in item for key in ["hotelName", "name", "hotelId"]) and any(
            key in item for key in ["price", "lowestPrice", "address", "starRating"]
        ):
            hotels.append(item)
    return hotels


def _extract_price(hotel: dict) -> float | None:
    price_fields = [
        hotel.get("lowestPrice"),
        hotel.get("minPrice"),
        hotel.get("price"),
        hotel.get("salePrice"),
    ]
    if isinstance(hotel.get("price"), dict):
        price_fields.extend(
            [
                hotel["price"].get("lowestPrice"),
                hotel["price"].get("minPrice"),
                hotel["price"].get("amount"),
            ]
        )
    for value in price_fields:
        if value in (None, "", []):
            continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def _hotel_name(hotel: dict) -> str:
    return str(hotel.get("name") or hotel.get("hotelName") or hotel.get("title") or "未知酒店")


def _format_hotel_summary(data: dict, max_items: int = 5) -> tuple[list[float], list[str]]:
    hotels = _extract_hotels(data)
    prices = []
    lines = []
    for hotel in hotels[:max_items]:
        name = _hotel_name(hotel)
        price = _extract_price(hotel)
        if price is not None:
            prices.append(price)
        star = hotel.get("starRating") or hotel.get("star") or hotel.get("starName") or ""
        address = hotel.get("address") or hotel.get("addr") or ""
        price_text = f"{price} 元起" if price is not None else "价格待确认"
        meta = "，".join(str(item) for item in [star, address] if item)
        lines.append(f"- {name}：{price_text}" + (f"（{meta}）" if meta else ""))
    return prices, lines


@tool
def hotel_search_placeholder(city: str, checkin_date: str = "", nights: int = 1) -> str:
    """酒店MCP查询工具。"""
    if not HOTEL_MCP_URL:
        return (
            "酒店 MCP Remote URL 尚未配置。"
            "当前只能使用预算工具的城市消费系数粗估住宿价格。"
        )

    servers = get_mcp_server_config()
    if "hotel" not in servers:
        return "酒店 MCP 配置读取失败，请检查 .env 中的 HOTEL_MCP_URL。"

    return (
        "已检测到酒店 MCP 配置。\n"
        f"待安装 MCP 依赖并完成工具发现后，可查询 {city} 入住日期 {checkin_date or '未指定'}、"
        f"{nights} 晚的酒店价格。\n"
        "调试命令：python -m trip_pilot.app.list_mcp_tools"
    )


@tool
def check_hotel_mcp_status() -> str:
    """检查酒店 MCP 是否已配置。"""
    if not HOTEL_MCP_URL:
        return "酒店 MCP 未配置。"
    return "酒店 MCP 已配置，下一步请运行工具发现脚本确认服务端工具列表。"


@tool
def search_hotels_via_mcp(
    city: str,
    checkin_date: str = "",
    stay_nights: int = 1,
    adult_count: int = 2,
    max_price_per_night: float | None = None,
    min_price_per_night: float | None = None,
    place_type: str = "城市",
    star_ratings: list[float] | None = None,
    size: int = 5,
) -> str:
    """通过酒店 MCP 查询酒店候选与价格。"""
    if not HOTEL_MCP_URL:
        return "酒店 MCP 未配置，无法查询实时酒店价格。"

    args = _build_hotel_search_args(
        place=city,
        place_type=place_type,
        origin_query=f"查询{city}酒店，入住{stay_nights}晚，成人{adult_count}人",
        checkin_date=checkin_date,
        stay_nights=stay_nights,
        adult_count=adult_count,
        max_price_per_night=max_price_per_night,
        min_price_per_night=min_price_per_night,
        star_ratings=star_ratings,
        size=size,
    )

    try:
        result = _call_hotel_tool("searchHotels", args)
        data = _extract_hotel_payload(result)
        prices, lines = _format_hotel_summary(data, max_items=size)
        if lines:
            return "酒店 MCP 查询成功：\n" + "\n".join(lines)
        return "酒店 MCP 查询成功，但未解析到标准酒店列表，原始摘要：" + str(data)[:1500]
    except Exception as e:
        return (
            "酒店 MCP 查询失败，已降级为预算估算模式。"
            f"失败原因：{type(e).__name__}: {e}"
        )


def query_hotel_reference_price(
    city: str,
    checkin_date: str = "",
    stay_nights: int = 1,
    adult_count: int = 2,
    max_price_per_night: float | None = None,
    place_type: str = "城市",
    star_ratings: list[float] | None = None,
) -> tuple[float | None, str]:
    """查询酒店参考价，供预算模块内部调用。"""
    if not HOTEL_MCP_URL:
        return None, "酒店 MCP 未配置，使用城市消费系数估算住宿。"

    args = _build_hotel_search_args(
        place=city,
        place_type=place_type,
        origin_query=f"查询{city}酒店价格，用于旅行预算估算",
        checkin_date=checkin_date,
        stay_nights=stay_nights,
        adult_count=adult_count,
        max_price_per_night=max_price_per_night,
        star_ratings=star_ratings,
        size=5,
    )

    try:
        result = _call_hotel_tool("searchHotels", args)
        data = _extract_hotel_payload(result)
        prices, lines = _format_hotel_summary(data, max_items=5)

        if not prices:
            if lines:
                return None, "酒店 MCP 返回候选，但没有可用价格，继续使用城市消费系数估算。\n" + "\n".join(lines)
            return None, "酒店 MCP 返回成功，但没有解析到可用酒店价格，继续使用城市消费系数估算。"

        prices = sorted(prices)
        lowest = prices[0]
        median = prices[len(prices) // 2]
        avg = round(sum(prices) / len(prices), 2)
        note = (
            f"酒店 MCP 查到 {len(prices)} 个候选，最低价 {lowest} 元，"
            f"中位参考价 {median} 元，均价约 {avg} 元。"
            "预算使用中位参考价，避免被极低价误导。\n"
            + "\n".join(lines[:3])
        )
        return median, note
    except Exception as e:
        return None, (
            "酒店 MCP 查询失败，继续使用城市消费系数估算。"
            f"失败原因：{type(e).__name__}: {e}。"
            "若当前使用 ModelScope 代理出现 502，可切换 RollingGo 官方 endpoint 并配置 Bearer Token。"
        )


def query_hotel_lowest_price(*args, **kwargs) -> tuple[float | None, str]:
    """兼容旧函数名，实际返回更稳健的酒店参考价。"""
    return query_hotel_reference_price(*args, **kwargs)
