from langchain.tools import tool


CITY_COST_LEVEL = {
    "北京": 1.35,
    "上海": 1.4,
    "杭州": 1.18,
    "深圳": 1.35,
    "广州": 1.2,
    "成都": 1.0,
    "重庆": 0.95,
    "长沙": 0.95,
    "西安": 0.95,
    "南京": 1.08,
    "厦门": 1.15,
    "青岛": 1.1,
    "郑州": 0.9,
}


LEVEL_BASE = {
    "低": {"hotel": 180, "food": 70, "local_transport": 25, "tickets": 40},
    "中等": {"hotel": 320, "food": 120, "local_transport": 45, "tickets": 90},
    "高": {"hotel": 650, "food": 240, "local_transport": 100, "tickets": 180},
}


def _get_city_factor(city: str) -> float:
    for name, factor in CITY_COST_LEVEL.items():
        if name in city:
            return factor
    return 1.0


@tool
def estimate_trip_budget(
    days: int,
    people: int,
    city: str,
    budget_level: str = "中等",
    hotel_price_per_night: float | None = None,
) -> str:
    """估算旅行基础预算，可传入酒店 MCP 查询到的每晚住宿价格。"""
    try:
        days = max(int(days), 1)
        people = max(int(people), 1)

        if budget_level not in LEVEL_BASE:
            budget_level = "中等"

        city_factor = _get_city_factor(city)
        base = LEVEL_BASE[budget_level]
        hotel = hotel_price_per_night or round(base["hotel"] * city_factor)
        food = round(base["food"] * city_factor)
        local_transport = round(base["local_transport"] * city_factor)
        tickets = round(base["tickets"] * city_factor)

        # 两人同行通常住同一间房，住宿按房间计，不乘人数。
        hotel_total = hotel * max(days - 1, 1)
        food_total = food * days * people
        transport_total = local_transport * days * people
        ticket_total = tickets * days * people
        total = hotel_total + food_total + transport_total + ticket_total

        hotel_note = (
            "住宿价格来自外部酒店查询结果"
            if hotel_price_per_night
            else "住宿价格来自城市消费系数粗估，建议接入酒店 MCP 后替换"
        )

        return (
            f"{city}{days}天{people}人{budget_level}预算估算：\n"
            f"- 城市消费系数：{city_factor}\n"
            f"- 住宿：约 {hotel_total} 元（{hotel_note}）\n"
            f"- 餐饮：约 {food_total} 元（按每人每天 {food} 元）\n"
            f"- 市内交通：约 {transport_total} 元（按每人每天 {local_transport} 元）\n"
            f"- 门票/活动：约 {ticket_total} 元（按每人每天 {tickets} 元）\n"
            f"- 合计：约 {total} 元\n"
            "说明：这是粗略估算，不含往返大交通。酒店、门票和餐饮价格应优先用实时工具确认。"
        )
    except Exception as e:
        return f"预算估算失败：{e}"
