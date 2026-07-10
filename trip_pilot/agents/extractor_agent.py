import json
import re

from langchain.messages import HumanMessage, SystemMessage

from trip_pilot.models import get_fast_model
from trip_pilot.schemas import IntentResult, TripRequest


EXTRACTOR_SYSTEM_PROMPT = """
你是一个出行需求参数抽取助手。
你的任务是把用户自然语言中的出行需求抽取为结构化字段，并结合已有状态做增量更新。
如果用户没有明确给出某个字段，就保留为空或使用默认值，不要编造。
如果用户是在回答上一轮追问，例如“包含”“不包含”“三天”“两个人”“高铁”，要结合已有状态理解它是在补充哪个字段。
例：
- 已有 budget 但 budget_scope 为空，用户说“包含”，应抽取 budget_scope="含往返大交通"。
- 已有 budget 但 budget_scope 为空，用户说“不包含/只算当地”，应抽取 budget_scope="仅当地花费"。
- 用户说“三天”，应抽取 days=3。
- 用户说“两个人”，应抽取 people=2。
start_date 可以是具体日期，也可以是“本周末”“下周六”等相对日期；如果用户只说“周末两天”，可以抽取为“本周末”。
budget_scope 用于记录预算是否包含往返大交通，例如“含往返大交通”“仅当地花费”“不确定”。
accommodation_preference 用于记录住宿偏好，例如青旅、经济型酒店、四星酒店、地铁沿线。
preferences 用于记录兴趣偏好，例如文化、美食、亲子、轻松、夜游。
constraints 用于记录限制条件，例如别太累、预算有限、必须当天往返。
need_report 表示用户是否明确要求生成正式行程、攻略、计划或报告。
need_export 表示用户是否明确要求保存、导出、下载文件。

请只输出 JSON，不要输出解释。JSON 字段如下：
{
  "origin": "始发地或 null",
  "destination": "目的地城市",
  "start_date": "出发日期或 null",
  "days": 旅行天数或 null,
  "people": 出行人数,
  "budget": 总预算数字或 null,
  "budget_scope": "含往返大交通/仅当地花费/不确定/null",
  "travel_mode": "城际交通偏好或 null",
  "city_mode": "市内交通偏好或 null",
  "accommodation_preference": "住宿偏好或 null",
  "preferences": ["兴趣偏好"],
  "constraints": ["限制条件"],
  "need_report": true 或 false,
  "need_export": true 或 false
}
"""

CONTEXTUAL_UPDATE_PROMPT = """
你是 Agent-TripPilot 的上下文补槽器。
你的任务不是重新规划，而是结合当前 TripState 和用户本轮短回答，判断用户在补充哪个旅行字段。

特别注意：
- 用户短答往往是在回答上一轮追问，不要按孤立句子理解。
- 如果当前已有 budget 且 budget_scope 为空，用户说“包含/含/包括/算上”，应理解为 budget_scope="含往返大交通"。
- 如果当前已有 budget 且 budget_scope 为空，用户说“不包含/不含/只算当地/仅当地”，应理解为 budget_scope="仅当地花费"。
- 如果用户说“三天/3天”，应理解为 days=3。
- 如果用户说“两个人/2人”，应理解为 people=2。
- 如果用户说“高铁/飞机/自驾”，应理解为 travel_mode。
- 不确定时不要编造，保留已有字段。

请只输出合并后的 TripRequest JSON，不要解释。
严禁因为示例词覆盖当前状态；例如用户说“三天”时，只能改 days，不能把 people 改成示例中的 2。
"""

INTENT_SYSTEM_PROMPT = """
你是 Agent-TripPilot 的意图识别器。
请判断用户这句话属于哪类意图：
- chat：普通聊天、问候、闲聊、一般问题，不要求规划旅行
- trip_plan：用户想规划旅行、行程、攻略、路线、预算、出游
- update：用户在补充或修改已有旅行需求
- export：用户要求保存、导出、生成文件
- cancel：用户取消、退出、重置需求

请只输出 JSON：
{
  "intent": "chat/trip_plan/update/export/cancel",
  "confidence": 0到1的小数,
  "reason": "简短理由"
}
"""


def classify_intent(user_input: str, history_summary: str = "") -> IntentResult:
    """判断用户当前输入的意图。"""
    model = get_fast_model()
    prompt = f"""
# 历史摘要
{history_summary or "无"}

# 用户输入
{user_input}
"""
    response = model.invoke(
        [
            SystemMessage(content=INTENT_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )
    try:
        data = _parse_json(response.content)
        return IntentResult.model_validate(data)
    except Exception:
        return _heuristic_intent(user_input, history_summary)


def extract_trip_request(user_input: str, current_request: TripRequest | None = None) -> TripRequest:
    """将用户自然语言需求抽取成 TripRequest。"""
    model = get_fast_model()
    current_json = (
        current_request.model_dump_json(indent=2, ensure_ascii=False)
        if current_request
        else "无"
    )
    prompt = f"""
# 已有出行需求状态
{current_json}

# 用户新输入
{user_input}

请抽取用户新输入中的信息。如果已有状态中有信息，而用户没有修改，不要重复编造；只输出合并后的 JSON。
"""
    response = model.invoke(
        [
            SystemMessage(content=EXTRACTOR_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )
    try:
        data = _parse_json(response.content)
        extracted = TripRequest.model_validate(data)
    except Exception:
        extracted = _heuristic_trip_request(user_input)
    if current_request:
        extracted = _guard_contextual_update(user_input, current_request, extracted)
        return current_request.merge(extracted)
    return extracted


def resolve_contextual_trip_update(
    user_input: str,
    current_request: TripRequest,
    state_summary: str = "",
    history_summary: str = "",
) -> TripRequest:
    """用 LLM 结合上下文理解短回答，规则抽取只作为兜底。"""
    model = get_fast_model()
    prompt = f"""
# 当前 TripRequest
{current_request.model_dump_json(indent=2, ensure_ascii=False)}

# 当前 TripState
{state_summary or "无"}

# 最近对话
{history_summary or "无"}

# 用户本轮输入
{user_input}

请结合上下文输出合并后的 TripRequest JSON。
"""
    try:
        response = model.invoke(
            [
                SystemMessage(content=CONTEXTUAL_UPDATE_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
        data = _parse_json(response.content)
        extracted = TripRequest.model_validate(data)
        extracted = _guard_contextual_update(user_input, current_request, extracted)
    except Exception:
        extracted = _heuristic_trip_request(user_input)
    return current_request.merge(extracted)


def _guard_contextual_update(
    user_input: str,
    current_request: TripRequest,
    extracted: TripRequest,
) -> TripRequest:
    """短回答补槽时只允许更新用户本轮明确提到的字段，防止 LLM 被示例污染。"""
    text = user_input.strip()
    if len(text) > 18:
        return extracted

    allowed = set()
    if re.search(r"[一二两三四五六七八九十\d]+\s*(天|日)", text):
        allowed.add("days")
    if re.search(r"[一二两三四五六七八九十\d]+\s*(个人|人)", text):
        allowed.add("people")
    if any(keyword in text for keyword in ["包含", "包括", "含", "不包含", "不含", "当地", "往返"]):
        allowed.add("budget_scope")
    if any(keyword in text for keyword in ["高铁", "火车", "飞机", "自驾", "大巴"]):
        allowed.add("travel_mode")
    if any(keyword in text for keyword in ["本周末", "周末", "明天", "后天", "下周"]) or re.search(r"\d{4}-\d{1,2}-\d{1,2}", text):
        allowed.add("start_date")
    if "出发" in text:
        allowed.add("origin")
    if any(keyword in text for keyword in ["预算", "元", "块"]):
        allowed.add("budget")

    if not allowed:
        return _heuristic_trip_request(user_input)

    base = current_request.model_dump()
    new_data = extracted.model_dump()
    for field in allowed:
        value = new_data.get(field)
        if value not in (None, "", []):
            base[field] = value
    return TripRequest.model_validate(base)


def _parse_json(text: str) -> dict:
    """从模型输出中解析 JSON，兼容 ```json 代码块。"""
    cleaned = (text or "").strip()
    if not cleaned:
        raise json.JSONDecodeError("empty model output", cleaned, 0)
    match = re.search(r"```json\s*(.*?)```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _heuristic_intent(user_input: str, history_summary: str = "") -> IntentResult:
    text = user_input.strip()
    if any(keyword in text for keyword in ["重置", "取消", "退出", "重新开始"]):
        return IntentResult(intent="cancel", confidence=0.8, reason="规则兜底识别为取消或重置")
    if any(keyword in text for keyword in ["导出", "保存", "下载", "Markdown", "文件"]):
        return IntentResult(intent="export", confidence=0.8, reason="规则兜底识别为导出")
    if _looks_like_followup_value(text) or any(keyword in text for keyword in ["便宜", "预算", "轻松", "美食", "酒店", "住宿", "改", "加"]):
        return IntentResult(intent="update", confidence=0.75, reason="规则兜底识别为补充或修改")
    if any(keyword in text for keyword in ["去", "旅行", "旅游", "行程", "攻略", "玩", "出发"]):
        return IntentResult(intent="trip_plan", confidence=0.75, reason="规则兜底识别为旅行规划")
    if "缺少" in history_summary or "补充" in history_summary:
        return IntentResult(intent="update", confidence=0.65, reason="结合历史，当前输入可能是在补充缺失信息")
    return IntentResult(intent="chat", confidence=0.55, reason="规则兜底识别为普通聊天")


def _heuristic_trip_request(user_input: str) -> TripRequest:
    text = user_input.strip()
    data = {}

    origin_match = re.search(r"(?:从)?([^，。,\s]+)出发", text)
    if origin_match:
        data["origin"] = origin_match.group(1)

    dest_match = re.search(r"(?:去|到)([^，。,\s]+)", text)
    if dest_match:
        data["destination"] = _clean_place_name(dest_match.group(1))

    days = _extract_number_before_keywords(text, ["天", "日"])
    if days:
        data["days"] = days

    people = _extract_number_before_keywords(text, ["个人", "人"])
    if people:
        data["people"] = people

    budget_match = re.search(r"(?:预算|花费|总共|一共)\D*(\d+(?:\.\d+)?)", text)
    if budget_match:
        data["budget"] = float(budget_match.group(1))

    if any(keyword in text for keyword in ["本周末", "周末", "明天", "后天", "下周"]):
        data["start_date"] = next(keyword for keyword in ["本周末", "周末", "明天", "后天", "下周"] if keyword in text)
    date_match = re.search(r"\d{4}-\d{1,2}-\d{1,2}", text)
    if date_match:
        data["start_date"] = date_match.group(0)

    preferences = []
    for keyword in ["文化", "美食", "亲子", "夜游", "自然", "购物"]:
        if keyword in text:
            preferences.append(keyword)

    constraints = []
    for keyword in ["别太累", "轻松", "少走路", "便宜", "预算有限"]:
        if keyword in text:
            constraints.append(keyword)

    if preferences:
        data["preferences"] = preferences
    if constraints:
        data["constraints"] = constraints
    if any(keyword in text for keyword in ["生成", "正式", "攻略", "行程", "计划"]):
        data["need_report"] = True
    if any(keyword in text for keyword in ["导出", "保存", "下载"]):
        data["need_export"] = True

    return TripRequest.model_validate(data)


def _clean_place_name(value: str) -> str:
    value = re.sub(r"[一二两三四五六七八九十\d]+(?:天|日|个人|人).*", "", value)
    value = re.sub(r"(预算|喜欢|偏好|出发).*", "", value)
    return value.strip()


def _looks_like_followup_value(text: str) -> bool:
    return bool(
        re.fullmatch(r"[一二两三四五六七八九十\d]+天", text)
        or re.fullmatch(r"[一二两三四五六七八九十\d]+个人?", text)
        or re.fullmatch(r"\d+(?:\.\d+)?", text)
    )


def _extract_number_before_keywords(text: str, keywords: list[str]) -> int | None:
    pattern = r"([一二两三四五六七八九十\d]+)\s*(?:" + "|".join(map(re.escape, keywords)) + r")"
    match = re.search(pattern, text)
    if not match:
        return None
    return _parse_cn_number(match.group(1))


def _parse_cn_number(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    mapping = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    if value in mapping:
        return mapping[value]
    if value.startswith("十") and len(value) == 2:
        return 10 + mapping.get(value[1], 0)
    if value.endswith("十") and len(value) == 2:
        return mapping.get(value[0], 0) * 10
    if "十" in value and len(value) == 3:
        return mapping.get(value[0], 0) * 10 + mapping.get(value[2], 0)
    return None
