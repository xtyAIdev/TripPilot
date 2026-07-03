from dataclasses import dataclass


@dataclass
class ScoreItem:
    dimension: str
    metric: str
    score: int
    max_score: int
    note: str


def evaluate_trip_response(user_input: str, response: str, debug_text: str = "") -> list[ScoreItem]:
    """轻量启发式评分，便于每次调试后快速发现短板。"""
    checks = [
        ScoreItem("需求理解", "参数抽取准确率", _has_any(response, ["目的地", "天数", "人数", "预算"], 20), 20, "检查是否覆盖关键参数"),
        ScoreItem("约束满足", "预算/偏好/节奏/天数满足率", _constraint_score(user_input, response), 20, "检查预算、偏好和节奏是否落地"),
        ScoreItem("路线规划", "可执行性与顺路性", _has_any(response, ["交通", "路线", "换乘", "步行", "距离"], 15), 15, "检查是否有交通或动线说明"),
        ScoreItem("工具使用", "工具选择与调用效率", _tool_score(debug_text), 15, "检查是否有 Plan/ReAct/Observation"),
        ScoreItem("输出质量", "结构化、表格化、可读性", _structure_score(response), 10, "检查表格和章节"),
        ScoreItem("流畅性", "自然度、承接感、对话体验", _fluency_score(response), 10, "检查是否先给结论且不过度生硬"),
        ScoreItem("记忆稳定", "多轮延续与修改处理", _has_any(response, ["上一版", "基于", "调整", "保留"], 5), 5, "检查是否体现上下文承接"),
        ScoreItem("可靠性", "幻觉控制与不确定性表达", _has_any(response, ["待确认", "以实际", "实时", "估算"], 5), 5, "检查是否标注不确定项"),
    ]
    return checks


def render_score_table(items: list[ScoreItem]) -> str:
    total = sum(item.score for item in items)
    lines = [
        "| 维度 | 指标 | 分值 | 说明 |",
        "| --- | --- | ---: | --- |",
    ]
    for item in items:
        lines.append(f"| {item.dimension} | {item.metric} | {item.score}/{item.max_score} | {item.note} |")
    lines.append(f"| 总分 |  | {total}/100 |  |")
    return "\n".join(lines)


def _has_any(text: str, keywords: list[str], max_score: int) -> int:
    hits = sum(1 for keyword in keywords if keyword in text)
    return round(max_score * min(hits / max(len(keywords), 1), 1))


def _constraint_score(user_input: str, response: str) -> int:
    watched = ["预算", "文化", "美食", "轻松", "少走路", "亲子", "夜游"]
    requested = [keyword for keyword in watched if keyword in user_input]
    if not requested:
        return 16 if "待确认" in response else 14
    hits = sum(1 for keyword in requested if keyword in response)
    return round(20 * hits / len(requested))


def _tool_score(debug_text: str) -> int:
    if not debug_text:
        return 5
    score = 0
    for keyword in ["Plan", "Thought", "Action", "Observation", "Constraint Check"]:
        if keyword in debug_text:
            score += 3
    return min(score, 15)


def _structure_score(response: str) -> int:
    score = 0
    if "##" in response:
        score += 3
    if "|" in response:
        score += 3
    if "待确认" in response:
        score += 2
    if "下一步" in response:
        score += 2
    return min(score, 10)


def _fluency_score(response: str) -> int:
    first_line = response.strip().splitlines()[0] if response.strip() else ""
    score = 6 if first_line and not first_line.startswith("|") and len(first_line) < 120 else 3
    if "建议" in response:
        score += 2
    if "可以" in response:
        score += 2
    return min(score, 10)
