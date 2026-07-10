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
    user_budget: float | None = None,
    budget_scope: str | None = None,
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
        per_person_total = round(total / people)
        per_person_day = round(total / people / days)

        hotel_note = (
            "住宿价格来自外部酒店查询结果"
            if hotel_price_per_night
            else "住宿价格来自城市消费系数粗估，建议接入酒店 MCP 后替换"
        )
        scope_note = budget_scope or "未确认是否包含往返大交通"

        feasibility = "未提供用户预算，暂不判断是否超支。"
        if user_budget is not None:
            user_budget = float(user_budget)
            gap = round(user_budget - total)
            if gap >= 0:
                feasibility = f"用户预算约 {user_budget:.0f} 元，按当前估算还剩约 {gap} 元机动空间。"
            else:
                feasibility = f"用户预算约 {user_budget:.0f} 元，按当前估算缺口约 {abs(gap)} 元，需要压缩住宿、餐饮或活动。"
            if "含往返" in scope_note:
                feasibility += " 由于预算包含往返大交通，实际当地可用预算会更紧。"

        return (
            f"{city}{days}天{people}人{budget_level}预算估算：\n"
            f"- 城市消费系数：{city_factor}\n"
            f"- 住宿：约 {hotel_total} 元（{hotel_note}）\n"
            f"- 餐饮：约 {food_total} 元（按每人每天 {food} 元）\n"
            f"- 市内交通：约 {transport_total} 元（按每人每天 {local_transport} 元）\n"
            f"- 门票/活动：约 {ticket_total} 元（按每人每天 {tickets} 元）\n"
            f"- 合计：约 {total} 元\n"
            f"- 人均合计：约 {per_person_total} 元，人均每天：约 {per_person_day} 元\n"
            f"- 预算口径：{scope_note}\n"
            f"- 可行性判断：{feasibility}\n"
            "说明：这是当地基础消费粗估，酒店、门票、餐饮和往返大交通应优先以实时工具或平台价格为准。"
        )
    except Exception as e:
        return f"预算估算失败：{e}"
