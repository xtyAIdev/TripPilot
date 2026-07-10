import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from langchain.messages import HumanMessage, SystemMessage

from trip_pilot.agents.constraint_checker import format_constraint_check, validate_constraints
from trip_pilot.agents.extractor_agent import (
    classify_intent,
    extract_trip_request,
    resolve_contextual_trip_update,
)
from trip_pilot.config import OUTPUT_DIR
from trip_pilot.memory import TripPilotMemory
from trip_pilot.models import get_chat_model, get_fast_model, get_reflection_model
from trip_pilot.rag.retriever import search_travel_docs
from trip_pilot.schemas import TripRequest
from trip_pilot.tools.budget_tools import estimate_trip_budget
from trip_pilot.tools.gaode_mcp_tools import (
    distance_via_gaode,
    geocode_via_gaode,
    get_weather_via_gaode,
    search_poi_via_gaode,
)
from trip_pilot.tools.hotel_mcp_tools import query_hotel_reference_price
from trip_pilot.tools.runtime_tools import calculate, current_time_text, get_current_time


REQUIRED_FIELDS = {
    "origin": "始发地",
    "destination": "目的地城市",
    "start_date": "出发日期",
    "days": "旅行天数",
    "people": "出行人数",
}


GENERAL_CHAT_PROMPT = """
你是 Agent-TripPilot，一个懂旅行规划的智能助手。
当前用户不是在要求正式生成行程时，请正常聊天或简洁回答。
如果用户只是其它意图，不要强行追问旅行参数。
如果用户表达了旅行意图，再引导他们提供目的地、天数、人数、预算、偏好等信息。
不要使用 emoji 或特殊装饰符号。
"""


STYLE_GUIDE = """
回复风格：
- 先给一句自然结论，再给结构化内容。
- 不要机械复述用户原话，不要模板腔。
- 对短问题少说，对正式报告再展开。
- 不确定信息说清楚“待确认”，不要编造票价、营业时间、实时库存。
- 结尾给 2 到 3 个下一步建议。
"""


PLANNER_PROMPT = """
你是 Planner。只决定要做哪些步骤，不写行程。

可用 tool：
- time：当前时间、相对日期
- rag：本地旅行知识
- weather：高德天气
- poi：高德 POI 搜索
- geocode：地点转经纬度
- distance：高德距离测量
- hotel：酒店参考价
- budget：预算估算
- calculate：基础数学计算
- draft：生成草案或正式行程
- reflection：质检和优化

输出 JSON 数组：
[
  {
    "step": 1,
    "name": "步骤名",
    "tool": "tool名称",
    "reason": "为什么需要",
    "params": {"poi_keywords": ["可选"], "route_points": ["可选"], "budget_focus": "可选"}
  }
]

规则：
- 只使用上面列出的 tool。
- 数组里的每一项必须是对象，不能输出字符串步骤。
- 每个对象必须包含 step、name、tool、reason 四个字段。
- params 可省略；如果用户有明确偏好，请把 POI 关键词、路线点或预算关注点写进 params，方便 Executor 真正按计划执行。
- 草案模式可少用工具，正式报告应优先使用 weather/poi/hotel/budget。
- 用户关心路线顺序或少走路时，加入 geocode 和 distance。
- 用户提到预算拆分或算账时，加入 calculate。
"""


DRAFT_PROMPT = """
你是 Agent-TripPilot 的行程生成 Agent。
请基于用户需求、执行计划和工具 Observation 生成一版 Markdown 行程初稿。

要求：
- 开头先用 1 到 2 句话说明结论和适用前提。
- 明确说明哪些信息来自 MCP，哪些来自 RAG，哪些是估算。
- 不要假装知道实时票价、开放时间和预约状态。
- 若预算明显不足，要直接指出并给替代方案。
- 如果用户要求粗略规划或仍有字段缺失，请输出“草案”，不要装作最终确定方案。
- 必须包含：需求摘要表、每日安排表、预算拆分表、待确认项、下一步建议。
- 不要使用 emoji 或特殊装饰符号。
"""


OFFICIAL_REPORT_PROMPT = """
你是 Agent-TripPilot 的正式行程报告生成 Agent。

目标：输出可执行的 Markdown 行程报告。
要求：
- 开头先用 1 到 2 句话说明结论和整体策略。
- 必须严格使用 TripRequest 中的人数、出发地、目的地、天数和交通方式；不要把示例、默认值或“同伴”写进报告。
- 明确区分 RAG、MCP、预算工具、估算信息。
- 必须包含：需求摘要表、天气提醒、每日安排表、交通建议、预算拆分表、风险与待确认事项、下一步建议。
- 本生成阶段不要输出“质检结论”章节；Reflection Agent 会在最终阶段统一追加，避免重复。
- 如果预算或时间不可行，先说明风险，再给替代方案。
- 没有工具证据时，禁止编造具体车次、酒店名称、精确票价、营业时间、预约余量；只能写“建议到 12306/官方平台确认”或“以实时查询为准”。
- 不要使用 emoji 或特殊装饰符号。
"""


REFLECT_JSON_PROMPT = """
你是 Agent-TripPilot 的 Reflection Agent。
请审查行程初稿是否需要优化。

检查维度：
1. 必要参数是否缺失
2. 时间是否过紧
3. 路线是否绕路
4. 预算是否明显不合理
5. 是否满足用户偏好
6. 是否充分利用了 RAG、高德 MCP、酒店 MCP 和预算工具
7. 是否对实时信息过度确定

请只输出 JSON：
{
  "need_refine": true 或 false,
  "reason": "是否需要优化的核心理由",
  "issues": ["问题1", "问题2"],
  "suggestions": ["建议1", "建议2"]
}

终止条件：
- 如果没有严重问题，need_refine=false。
- 如果只剩需要用户确认的实时信息，不要无限优化，need_refine=false，并把待确认项写入 reason。
"""


REFINE_PROMPT = """
你是 Agent-TripPilot 的 Refine Agent。
请根据 Reflection 反馈修改行程。

要求：
- 保留原行程中合理部分。
- 修正预算、路线、时间和提醒问题。
- 不要引入没有依据的新事实。
"""


REVISION_PROMPT = """
你是 Agent-TripPilot 的行程修订 Agent。
用户已经有一版行程，现在补充了新信息或提出修改要求。

请基于“上一版行程”和“用户新输入”直接修订，不要重新展开完整工具调用过程。
如果用户补充的信息会影响实时价格、天气、交通或预约，请在修订稿中标注“需要重新查询确认”。
如果用户只是补充偏好，请局部修改行程，不要重写无关部分。

"""


class TripPilotConversationAgent:
    """多轮智能 Agent：意图识别、记忆、确认参数、Plan/ReAct/Reflection。"""

    def __init__(self, thread_id: str = "default", max_reflection_iters: int = 2):
        self.memory = TripPilotMemory(thread_id=thread_id)
        self.max_reflection_iters = max_reflection_iters
        self.current_plan_mode = "official"
        self.current_reply_mode = "standard"
        self.progress_callback = None

    def chat(self, user_input: str, debug: bool = True, progress_callback: Callable[[str], None] | None = None) -> str:
        self.progress_callback = progress_callback
        self.memory.add_user_message(user_input)
        self.current_reply_mode = self._detect_reply_mode(user_input)
        self.memory.reply_mode = self.current_reply_mode
        self.memory.phase = "collecting"

        self._progress("正在解析出行需求", stage="parse")
        intent = classify_intent(user_input, self.memory.history_summary())
        self._debug(f"[Intent] {intent.intent} | confidence={intent.confidence} | {intent.reason}", debug)
        contextual_updated = self._resolve_contextual_update(user_input, debug=debug)
        direct_updated = self._apply_direct_slot_updates(user_input)

        if intent.intent == "cancel":
            self.memory.reset_trip_request()
            self.memory.phase = "idle"
            answer = "已重置当前出行需求。你可以重新告诉我目的地、天数、人数和偏好。"
            self.memory.add_ai_message(answer)
            return answer

        if intent.intent == "chat" and not self._looks_like_trip_context():
            self.memory.phase = "idle"
            answer = self._general_chat(user_input)
            self.memory.add_ai_message(answer)
            return answer

        if intent.intent == "export" and self.memory.last_plan:
            path = self._export_plan(self.memory.last_plan)
            self.memory.phase = "done"
            answer = f"已导出上一版行程：{path}"
            self.memory.add_ai_message(answer)
            return answer

        if intent.intent in {"trip_plan", "update", "export"} or contextual_updated or direct_updated:
            if not direct_updated or intent.intent in {"trip_plan", "update", "export"}:
                self.memory.trip_request = extract_trip_request(user_input, self.memory.trip_request)
                self._apply_direct_slot_updates(user_input)
            self._update_profile_from_request()
            self._debug(
                "[Memory] 当前结构化需求：\n"
                + self.memory.trip_request.model_dump_json(indent=2, ensure_ascii=False),
                debug,
            )
            self._debug("[State]\n" + self.memory.state_summary(), debug)

        if self.memory.last_plan and self._should_answer_as_followup(user_input, intent.intent):
            self.memory.phase = "follow_up"
            answer = self._answer_followup(user_input, debug=debug)
            if not self._needs_hotel_lookup(user_input):
                self.memory.last_plan = answer
            self.memory.add_ai_message(answer)
            return answer

        rough_mode = self._allows_rough_plan(user_input)

        if not self.memory.trip_request.need_report and intent.intent != "export" and not rough_mode:
            missing = self._missing_basic_fields(self.memory.trip_request)
            if missing:
                answer = self._ask_for_missing_fields(missing, formal=False)
                self.memory.add_ai_message(answer)
                return answer
            answer = (
                "我已经记录了这些出行信息。"
                "如果你想正式生成行程，请告诉我“生成行程”或继续补充预算、日期、交通方式、住宿偏好。"
            )
            self.memory.add_ai_message(answer)
            return answer

        missing = self._missing_formal_fields(self.memory.trip_request)
        self.current_plan_mode = "draft" if rough_mode else "official"
        self.memory.plan_mode = self.current_plan_mode

        if missing and not rough_mode:
            self.memory.phase = "collecting"
            answer = self._ask_for_missing_fields(missing, formal=True)
            self.memory.add_ai_message(answer)
            return answer
        if missing:
            self._debug(
                "[Mode] 用户允许先做粗略规划，以下字段暂缺："
                + "、".join(missing),
                debug,
            )

        self.memory.phase = "planning"
        final_plan = self._run_plan_solve_react_reflection(debug=debug)

        if self.memory.trip_request.need_export or intent.intent == "export":
            path = self._export_plan(final_plan)
            final_plan = f"{final_plan}\n\n已导出文件：{path}"

        self.memory.last_plan = final_plan
        self.memory.phase = "done"
        self.memory.add_ai_message(final_plan)
        return final_plan

    def _wants_full_regenerate(self, user_input: str) -> bool:
        keywords = ["重新生成", "重新规划", "从头", "完整重做", "全部重来"]
        return any(keyword in user_input for keyword in keywords)

    def _should_answer_as_followup(self, user_input: str, intent: str) -> bool:
        if self._wants_full_regenerate(user_input):
            return False
        followup_keywords = [
            "住",
            "住宿",
            "酒店",
            "民宿",
            "附近",
            "推荐",
            "预算",
            "改",
            "换",
            "补充",
            "加上",
            "删掉",
            "轻松",
            "少走路",
        ]
        return intent == "update" or any(keyword in user_input for keyword in followup_keywords)

    def _allows_rough_plan(self, user_input: str) -> bool:
        keywords = ["先粗略", "粗略", "大概", "先看看", "简单规划", "初步", "不用很详细"]
        return any(keyword in user_input for keyword in keywords)

    def _detect_reply_mode(self, user_input: str) -> str:
        if any(keyword in user_input for keyword in ["简版", "简单说", "简短", "概括", "先给结论"]):
            return "brief"
        if any(keyword in user_input for keyword in ["详细", "完整", "展开", "细一点", "正式报告"]):
            return "detailed"
        return "standard"

    def _progress(self, message: str, stage: str = "run", detail: str = "") -> None:
        self.memory.remember_step(stage=stage, title=message, detail=detail)
        if self.progress_callback:
            self.progress_callback(message)

    def _resolve_contextual_update(self, user_input: str, debug: bool = True) -> bool:
        """让 LLM 结合当前 TripState 理解短回答，返回状态是否变化。"""
        if not self._looks_like_trip_context():
            return False
        before = self.memory.trip_request.model_dump_json(ensure_ascii=False)
        self._progress("正在结合上下文理解补充信息", stage="memory")
        updated = resolve_contextual_trip_update(
            user_input=user_input,
            current_request=self.memory.trip_request,
            state_summary=self.memory.state_summary(),
            history_summary=self.memory.history_summary(),
        )
        self.memory.trip_request = updated
        after = self.memory.trip_request.model_dump_json(ensure_ascii=False)
        changed = before != after
        if changed:
            self._debug("[Contextual Slot Update]\n" + self.memory.trip_request.model_dump_json(indent=2, ensure_ascii=False), debug)
        return changed

    def _apply_direct_slot_updates(self, user_input: str) -> bool:
        """处理“包含”“三天”“高铁”这类短回答，避免完全依赖 LLM JSON 抽取。"""
        text = user_input.strip()
        request = self.memory.trip_request
        updated = False

        negative_scope = ["不包含", "不含", "不包括", "仅当地", "当地花费", "不算"]
        positive_scope = ["包含", "包括", "含", "算上", "往返"]
        if request.budget is not None and any(keyword in text for keyword in negative_scope):
            request.budget_scope = "仅当地花费"
            updated = True
        elif request.budget is not None and any(keyword in text for keyword in positive_scope):
            request.budget_scope = "含往返大交通"
            updated = True

        days = self._extract_short_number(text, ["天", "日"])
        if days:
            request.days = days
            updated = True

        people = self._extract_short_number(text, ["个人", "人"])
        if people:
            request.people = people
            updated = True

        if any(keyword in text for keyword in ["高铁", "火车", "飞机", "自驾", "大巴"]):
            request.travel_mode = text
            updated = True

        if any(keyword in text for keyword in ["本周末", "周末", "明天", "后天", "下周"]):
            request.start_date = next(keyword for keyword in ["本周末", "周末", "明天", "后天", "下周"] if keyword in text)
            updated = True
        date_match = re.search(r"\d{4}-\d{1,2}-\d{1,2}", text)
        if date_match:
            request.start_date = date_match.group(0)
            updated = True

        if updated:
            self.memory.trip_request = request
        return updated

    def _extract_short_number(self, text: str, keywords: List[str]) -> int | None:
        pattern = r"([一二两三四五六七八九十\d]+)\s*(?:" + "|".join(map(re.escape, keywords)) + r")"
        match = re.search(pattern, text)
        if not match:
            return None
        value = match.group(1)
        if value.isdigit():
            return int(value)
        mapping = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
        if value in mapping:
            return mapping[value]
        return None

    def _revise_existing_plan(self, user_input: str, debug: bool = True) -> str:
        self._debug("\n=== Revision | 基于上一版行程局部修订 ===", debug)
        model = get_chat_model()
        prompt = f"""
# 当前结构化需求
{self.memory.trip_request.model_dump_json(indent=2, ensure_ascii=False)}

# 用户新输入
{user_input}

# 上一版行程
{self.memory.last_plan}

请输出修订后的完整行程。
"""
        response = model.invoke(
            [
                SystemMessage(content=REVISION_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
        return response.content

    def _answer_followup(self, user_input: str, debug: bool = True) -> str:
        self._debug("\n=== Follow-up | 基于上一版行程回答补充问题 ===", debug)
        observations = {}

        if self._needs_hotel_lookup(user_input):
            request = self.memory.trip_request
            city = request.destination or ""
            stay_nights = max((request.days or 2) - 1, 1)
            people = request.people or 1
            self._debug("Thought: 用户追问住宿/酒店，需要查询酒店 MCP 作为预算参考。", debug)
            self._debug(
                f'Action: query_hotel_reference_price(city="{city}", stay_nights={stay_nights}, adult_count={people})',
                debug,
            )
            hotel_price, hotel_note = query_hotel_reference_price(
                city=city,
                checkin_date=self._mcp_date(request.start_date),
                stay_nights=stay_nights,
                adult_count=people,
            )
            observations["hotel"] = hotel_note
            observations["hotel_reference_price"] = str(hotel_price)
            self._debug(f"Observation: {hotel_note}", debug)

        model = get_chat_model()
        prompt = f"""
# 当前结构化需求
{self.memory.trip_request.model_dump_json(indent=2, ensure_ascii=False)}

# 会话状态快照
{self.memory.state_summary()}

# 用户追问
{user_input}

# 工具观察结果
{json.dumps(observations, ensure_ascii=False, indent=2)}

# 上一版行程
{self.memory.last_plan}

请直接回答用户本轮追问。
回复粒度：{self.current_reply_mode}
要求：
- 先给一句自然结论。
- 如果用户问酒店或住宿，结合预算、位置偏好和酒店 MCP 结果给出建议。
- 不要重新生成完整行程，除非用户明确要求从头重做。
- 如果酒店 MCP 没有稳定返回，就说明需要以平台实时价格为准，并给出筛选方法。
- 输出包含“推荐区域、预算判断、筛选条件、待确认事项、下一步建议”。
"""
        response = model.invoke(
            [
                SystemMessage(content=REVISION_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
        return response.content

    def _needs_hotel_lookup(self, user_input: str) -> bool:
        keywords = ["住", "住宿", "酒店", "民宿", "西湖附近", "客栈"]
        return any(keyword in user_input for keyword in keywords)

    def _update_profile_from_request(self) -> None:
        request = self.memory.trip_request
        joined = " ".join(request.preferences + request.constraints)
        if any(keyword in joined for keyword in ["轻松", "别太累", "少走路"]):
            self.memory.user_profile.preferred_pace = "轻松"
        if request.accommodation_preference:
            self.memory.user_profile.hotel_preference = request.accommodation_preference
        for preference in request.preferences:
            if "美食" in preference and "美食" not in self.memory.user_profile.food_preference:
                self.memory.user_profile.food_preference.append("美食")
            if preference and preference not in self.memory.user_profile.travel_style:
                self.memory.user_profile.travel_style.append(preference)

    def _looks_like_trip_context(self) -> bool:
        request = self.memory.trip_request
        return any([request.destination, request.origin, request.days, request.budget, request.preferences])

    def _general_chat(self, user_input: str) -> str:
        model = get_fast_model()
        response = model.invoke(
            [
                SystemMessage(content=GENERAL_CHAT_PROMPT),
                HumanMessage(content=f"用户输入：{user_input}\n历史：{self.memory.history_summary()}"),
            ]
        )
        return response.content

    def _missing_basic_fields(self, request: TripRequest) -> List[str]:
        missing = []
        for field in ["destination", "days", "people"]:
            if getattr(request, field) in (None, "", []):
                missing.append(REQUIRED_FIELDS[field])
        return missing

    def _missing_formal_fields(self, request: TripRequest) -> List[str]:
        missing = []
        for field, label in REQUIRED_FIELDS.items():
            if getattr(request, field) in (None, "", []):
                missing.append(label)
        if request.origin and not request.travel_mode:
            missing.append("往返交通方式")
        if request.budget is not None and not request.budget_scope:
            missing.append("预算是否包含往返大交通")
        return missing

    def _ask_for_missing_fields(self, missing: List[str], formal: bool = False) -> str:
        fields = "、".join(missing)
        self.memory.open_questions = [
            self._open_question_for_label(label)
            for label in missing
        ]
        if formal:
            return (
                f"可以继续，但正式生成前还差：{fields}。\n\n"
                "这些信息会影响天气、酒店价格、往返交通和预算判断。"
                "你直接用一句话补充即可，例如“下周五出发，高铁，预算包含往返交通”。"
            )
        return (
            f"我已经先记下这次旅行意图了，还差：{fields}。\n\n"
            "你可以继续补一句，比如“三天，两个人，预算 1500，想轻松一点”。"
        )

    def _open_question_for_label(self, label: str):
        from trip_pilot.schemas import OpenQuestion

        question_map = {
            "始发地": "你从哪个城市出发？",
            "目的地城市": "这次想去哪个城市？",
            "出发日期": "大概什么时候出发？可以说本周末、下周六或具体日期。",
            "旅行天数": "计划玩几天？",
            "出行人数": "一共几个人出行？",
            "往返交通方式": "往返大交通偏好高铁、飞机、自驾还是暂不确定？",
            "预算是否包含往返大交通": "预算是否包含往返大交通？",
        }
        return OpenQuestion(
            field=label,
            question=question_map.get(label, f"请确认{label}。"),
            reason="补齐后行程、预算和工具查询会更准确。",
        )

    def _run_plan_solve_react_reflection(self, debug: bool = True) -> str:
        request = self.memory.trip_request

        self._progress("正在制定执行计划", stage="plan")
        self._debug("\n=== Plan-and-Solve | 规划阶段 ===", debug)
        plan = self._make_plan(request)
        for item in plan:
            self._debug(
                f"[Plan] Step {item.get('step')}: {item.get('name')} | "
                f"tool={item.get('tool')} | {item.get('reason')}",
                debug,
            )

        self._debug("\n=== ReAct | 执行阶段 ===", debug)
        observations = self._execute_react_steps(request, plan, debug=debug)

        self._progress("正在检查约束条件", stage="check")
        constraint_check = validate_constraints(request)
        self.memory.open_questions = constraint_check.open_questions
        observations["constraint_check"] = format_constraint_check(constraint_check)
        self._debug("\n=== Constraint Check | 约束校验 ===", debug)
        self._debug(observations["constraint_check"], debug)

        self._progress("正在生成行程方案", stage="solve")
        self._debug("\n=== Solve | 生成初稿 ===", debug)
        draft = self._generate_draft(request, plan, observations)
        self._debug(draft[:1200], debug)

        if self.current_plan_mode == "draft":
            self._debug("\n=== Draft Mode | 草案模式跳过 Reflection ===", debug)
            return draft + "\n\n## 草案说明\n\n这是初步草案，未进入 Reflection 正式质检。补充日期、交通方式、预算范围后可生成正式行程。"

        self._progress("正在进行行程质检", stage="reflect")
        self._debug("\n=== Reflection | 反思优化阶段 ===", debug)
        final_plan = self._reflect_and_refine(request, draft, observations, debug=debug)
        return final_plan

    def _make_plan(self, request: TripRequest) -> List[Dict]:
        model = get_fast_model()
        prompt = f"""
# 出行需求
{request.model_dump_json(indent=2, ensure_ascii=False)}

# 运行时上下文
{current_time_text()}

请生成执行计划。
"""
        response = model.invoke(
            [
                SystemMessage(content=PLANNER_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
        try:
            return _normalize_plan(json.loads(_extract_json_text(response.content)))
        except Exception:
            return _default_plan()

    def _execute_react_steps(self, request: TripRequest, plan: List[Dict], debug: bool = True) -> Dict[str, str]:
        observations: Dict[str, str] = {}
        destination = request.destination or ""
        days = request.days or 1
        people = request.people or 1
        tools = self._tools_from_plan(plan)
        plan_hints = self._collect_plan_hints(plan)
        if not tools:
            tools = {"time", "rag", "weather", "poi", "hotel", "budget"}

        query = f"{destination} {' '.join(request.preferences)} {' '.join(request.constraints)}"

        if "time" in tools:
            self._progress("正在解析出行时间", stage="time")
            self._debug("Thought: 用户可能使用相对日期，需要注入当前运行时间。", debug)
            self._debug("Action: get_current_time()", debug)
            observations["time"] = self._cached_tool_invoke("time", {}, lambda: get_current_time.invoke({}))
            self._debug(f"Observation: {observations['time']}", debug)

        if "rag" in tools:
            self._progress("正在检索目的地知识", stage="rag")
            self._debug("Thought: 需要检索本地知识库，获得稳定的城市旅行背景。", debug)
            self._debug(f'Action: search_travel_docs(query="{query}")', debug)
            observations["rag"] = self._cached_tool_invoke(
                "rag",
                {"query": query, "destination": destination},
                lambda: self._format_docs(query, destination),
            )
            self._debug(f"Observation: {observations['rag'][:1000]}", debug)

        if "weather" in tools:
            self._progress("正在查询天气信息", stage="weather")
            self._debug("Thought: 天气会影响户外景点和行程强度，需要查询高德天气。", debug)
            self._debug(f'Action: get_weather_via_gaode(city="{destination}")', debug)
            observations["weather"] = self._cached_tool_invoke(
                "weather",
                {"city": destination},
                lambda: get_weather_via_gaode.invoke({"city": destination}),
            )
            self._debug(f"Observation: {observations['weather'][:1000]}", debug)

        if "poi" in tools:
            self._progress("正在查询景点与周边点位", stage="poi")
            poi_keywords = plan_hints.get("poi_keywords") or self._select_poi_keywords(request)
            poi_results = []
            for keyword in poi_keywords:
                self._debug("Thought: 需要用高德 POI 验证候选地点，并获取地址线索。", debug)
                self._debug(f'Action: search_poi_via_gaode(keyword="{keyword}", city="{destination}")', debug)
                result = self._cached_tool_invoke(
                    "poi",
                    {"keyword": keyword, "city": destination},
                    lambda keyword=keyword: search_poi_via_gaode.invoke({"keyword": keyword, "city": destination}),
                )
                poi_results.append(f"关键词：{keyword}\n{result[:1200]}")
                self._debug(f"Observation: {result[:800]}", debug)
            observations["poi"] = "\n\n".join(poi_results)

        if "geocode" in tools:
            self._progress("正在解析路线点位", stage="route")
            geo_keywords = plan_hints.get("route_points") or self._select_poi_keywords(request)[:2]
            geo_results = []
            for keyword in geo_keywords:
                self._debug("Thought: 路线和距离计算需要经纬度，先解析地点。", debug)
                self._debug(f'Action: geocode_via_gaode(address="{keyword}", city="{destination}")', debug)
                result = self._cached_tool_invoke(
                    "geocode",
                    {"address": keyword, "city": destination},
                    lambda keyword=keyword: geocode_via_gaode.invoke({"address": keyword, "city": destination}),
                )
                geo_results.append(f"地点：{keyword}\n{result[:1000]}")
                self._debug(f"Observation: {result[:800]}", debug)
            observations["geocode"] = "\n\n".join(geo_results)

        if "distance" in tools:
            self._progress("正在查询路线与交通", stage="route")
            observations["distance"] = (
                "Planner 要求距离测量，但当前缺少可靠的起终点经纬度。"
                "已在草案中标记路线距离需要进一步确认。"
            )
            self._debug("Thought: 缺少可直接用于距离测量的经纬度，暂不硬算。", debug)
            self._debug(f"Observation: {observations['distance']}", debug)

        hotel_price = None
        if "hotel" in tools:
            self._progress("正在核对住宿参考", stage="hotel")
            stay_nights = max(days - 1, 1)
            self._debug("Thought: 住宿价格波动大，查询酒店 MCP 参考价。", debug)
            self._debug(
                f'Action: query_hotel_reference_price(city="{destination}", stay_nights={stay_nights}, adult_count={people})',
                debug,
            )
            hotel_price, hotel_note = self._cached_hotel_reference(
                city=destination,
                checkin_date=self._mcp_date(request.start_date),
                stay_nights=stay_nights,
                adult_count=people,
            )
            observations["hotel"] = hotel_note
            self._debug(f"Observation: {hotel_note}", debug)

        if "budget" in tools:
            self._progress("正在估算预算", stage="budget")
            budget_level = self._guess_budget_level(request)
            self._debug("Thought: 需要根据城市消费系数和酒店参考价估算当地预算。", debug)
            self._debug(
                f'Action: estimate_trip_budget(days={days}, people={people}, city="{destination}", budget_level="{budget_level}", hotel_price_per_night={hotel_price})',
                debug,
            )
            observations["budget"] = self._cached_tool_invoke(
                "budget",
                {
                    "days": days,
                    "people": people,
                    "city": destination,
                    "budget_level": budget_level,
                    "hotel_price_per_night": hotel_price,
                    "user_budget": request.budget,
                    "budget_scope": request.budget_scope,
                },
                lambda: estimate_trip_budget.invoke(
                    {
                        "days": days,
                        "people": people,
                        "city": destination,
                        "budget_level": budget_level,
                        "hotel_price_per_night": hotel_price,
                        "user_budget": request.budget,
                        "budget_scope": request.budget_scope,
                    }
                ),
            )
            self._debug(f"Observation: {observations['budget']}", debug)

        if "calculate" in tools and request.budget and request.people:
            self._progress("正在计算预算拆分", stage="calculate")
            expression = f"{request.budget} / {request.people}"
            self._debug("Thought: 用户预算需要拆分，调用计算工具。", debug)
            self._debug(f'Action: calculate(expression="{expression}")', debug)
            observations["calculation"] = self._cached_tool_invoke(
                "calculate",
                {"expression": expression},
                lambda: calculate.invoke({"expression": expression}),
            )
            self._debug(f"Observation: {observations['calculation']}", debug)

        missing = self._missing_formal_fields(request)
        if missing:
            observations["missing_fields"] = "仍需确认：" + "、".join(missing)

        self.memory.last_tool_summary = {
            key: value[:300] if isinstance(value, str) else str(value)[:300]
            for key, value in observations.items()
        }
        return observations

    def _collect_plan_hints(self, plan: List[Dict]) -> Dict[str, List[str] | str]:
        hints: Dict[str, List[str] | str] = {
            "poi_keywords": [],
            "route_points": [],
            "budget_focus": "",
        }
        for item in plan:
            params = item.get("params") if isinstance(item, dict) else None
            if not isinstance(params, dict):
                continue
            for key in ["poi_keywords", "route_points"]:
                value = params.get(key)
                if isinstance(value, str) and value.strip():
                    hints[key].append(value.strip())
                elif isinstance(value, list):
                    hints[key].extend(str(v).strip() for v in value if str(v).strip())
            if params.get("budget_focus"):
                hints["budget_focus"] = str(params.get("budget_focus"))

        hints["poi_keywords"] = list(dict.fromkeys(hints["poi_keywords"]))[:4]
        hints["route_points"] = list(dict.fromkeys(hints["route_points"]))[:4]
        return hints

    def _cached_tool_invoke(self, tool_name: str, args: Dict, fn: Callable[[], str]) -> str:
        cache_key = json.dumps({"tool": tool_name, "args": args}, ensure_ascii=False, sort_keys=True)
        if cache_key in self.memory.tool_cache:
            return "[cache]\n" + self.memory.tool_cache[cache_key]
        try:
            result = fn()
        except Exception as e:
            result = f"{tool_name} 工具调用失败，已降级为保守模式：{e}"
        text = str(result)
        self.memory.tool_cache[cache_key] = text
        return text

    def _cached_hotel_reference(
        self,
        city: str,
        checkin_date: str,
        stay_nights: int,
        adult_count: int,
    ) -> Tuple[float | None, str]:
        cache_key = json.dumps(
            {
                "tool": "hotel_reference",
                "city": city,
                "checkin_date": checkin_date,
                "stay_nights": stay_nights,
                "adult_count": adult_count,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if cache_key in self.memory.tool_cache:
            cached = json.loads(self.memory.tool_cache[cache_key])
            return cached.get("price"), "[cache]\n" + cached.get("note", "")
        price, note = query_hotel_reference_price(
            city=city,
            checkin_date=checkin_date,
            stay_nights=stay_nights,
            adult_count=adult_count,
        )
        self.memory.tool_cache[cache_key] = json.dumps(
            {"price": price, "note": note},
            ensure_ascii=False,
        )
        return price, note

    def _tools_from_plan(self, plan: List[Dict]) -> set[str]:
        allowed = {
            "time",
            "rag",
            "weather",
            "poi",
            "geocode",
            "distance",
            "hotel",
            "budget",
            "calculate",
        }
        return {
            str(item.get("tool", "")).strip()
            for item in plan
            if isinstance(item, dict) and str(item.get("tool", "")).strip() in allowed
        }

    def _format_docs(self, query: str, destination: str = "") -> str:
        docs = search_travel_docs(query, k=8)
        if destination:
            city_docs = [doc for doc in docs if destination in doc.page_content]
            if city_docs:
                docs = city_docs
        lines = []
        for i, doc in enumerate(docs[:4], start=1):
            file_name = doc.metadata.get("file_name", "unknown")
            chunk_index = doc.metadata.get("chunk_index", "")
            lines.append(f"[{i}] {file_name} chunk:{chunk_index}\n{doc.page_content[:700]}")
        return "\n\n".join(lines) if lines else "没有检索到本地旅行知识。"

    def _select_poi_keywords(self, request: TripRequest) -> List[str]:
        base = []
        for preference in request.preferences:
            if "文化" in preference:
                base.extend(["博物馆", "历史文化街区"])
            if "美食" in preference:
                base.extend(["美食街", "小吃"])
            if "亲子" in preference:
                base.extend(["亲子", "公园"])
            if "自然" in preference or "风景" in preference:
                base.extend(["风景区", "公园"])

        for constraint in request.constraints:
            if "别太累" in constraint or "轻松" in constraint:
                base.append("地铁站附近景点")

        if not base:
            base = ["景点", "美食街"]

        return [item for item in list(dict.fromkeys(base)) if item][:3]

    def _generate_draft(self, request: TripRequest, plan: List[Dict], observations: Dict[str, str]) -> str:
        model = get_chat_model()
        system_prompt = DRAFT_PROMPT if self.current_plan_mode == "draft" else OFFICIAL_REPORT_PROMPT
        prompt = f"""
# 结构化需求
{request.model_dump_json(indent=2, ensure_ascii=False)}

# 会话状态快照
{self.memory.state_summary()}

# 生成模式
{self.current_plan_mode}

# 执行计划
{json.dumps(plan, ensure_ascii=False, indent=2)}

# 工具观察结果
{json.dumps(observations, ensure_ascii=False, indent=2)}

{STYLE_GUIDE}

回复粒度：{self.current_reply_mode}
- brief：只给结论、关键表格和待确认项。
- standard：给自然总结、必要表格、风险和下一步建议。
- detailed：给完整报告，细化预算、交通、证据来源和质检结论。

请生成行程初稿。
"""
        response = model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt),
            ]
        )
        return response.content

    def _reflect_and_refine(
        self,
        request: TripRequest,
        draft: str,
        observations: Dict[str, str],
        debug: bool = True,
    ) -> str:
        current = draft
        model = get_reflection_model()

        for i in range(self.max_reflection_iters):
            self._debug(f"[Reflection] 第 {i + 1}/{self.max_reflection_iters} 轮", debug)
            feedback_prompt = f"""
# 用户需求
{request.model_dump_json(indent=2, ensure_ascii=False)}

# 会话状态快照
{self.memory.state_summary()}

# 工具观察结果
{json.dumps(observations, ensure_ascii=False, indent=2)}

# 当前行程
{current}
"""
            feedback_response = model.invoke(
                [
                    SystemMessage(content=REFLECT_JSON_PROMPT),
                    HumanMessage(content=feedback_prompt),
                ]
            )
            feedback = _parse_json_dict(feedback_response.content)
            self._debug(
                "[Reflection Feedback]\n"
                + json.dumps(feedback, ensure_ascii=False, indent=2),
                debug,
            )

            if not feedback.get("need_refine"):
                self._debug("Action: Finish[Reflection 判断无需继续优化]", debug)
                return _append_quality_conclusion(current, feedback.get("reason", "无需继续优化。"))

            self._debug("Thought: Reflection 发现重要问题，需要根据反馈优化行程。", debug)
            self._debug("Action: refine_trip_plan(feedback=Reflection反馈)", debug)
            current = self._refine_plan(request, current, observations, feedback)

        self._debug("Action: Finish[达到最大 Reflection 迭代次数]", debug)
        return _append_quality_conclusion(current, "已达到最大反思迭代次数，建议人工确认剩余实时信息。")

    def _refine_plan(
        self,
        request: TripRequest,
        current: str,
        observations: Dict[str, str],
        feedback: Dict,
    ) -> str:
        model = get_chat_model()
        prompt = f"""
# 用户需求
{request.model_dump_json(indent=2, ensure_ascii=False)}

# 工具观察结果
{json.dumps(observations, ensure_ascii=False, indent=2)}

# 当前行程
{current}

# Reflection 反馈
{json.dumps(feedback, ensure_ascii=False, indent=2)}

请输出优化后的完整行程。
"""
        response = model.invoke(
            [
                SystemMessage(content=REFINE_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
        return response.content

    def _guess_budget_level(self, request: TripRequest) -> str:
        if request.budget is None or request.days is None or request.people is None:
            return "中等"
        per_person_day = request.budget / max(request.people, 1) / max(request.days, 1)
        if per_person_day < 350:
            return "低"
        if per_person_day > 900:
            return "高"
        return "中等"

    def _mcp_date(self, date_text: str | None) -> str:
        """MCP 酒店工具只接收 YYYY-MM-DD，相对日期不直接传。"""
        if not date_text:
            return ""
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_text.strip()):
            return date_text.strip()
        return ""

    def _export_plan(self, content: str) -> str:
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "trip_plan.md"
        output_path.write_text(content, encoding="utf-8")
        return str(output_path)

    def _debug(self, content: str, debug: bool = True) -> None:
        self.memory.remember_debug(content)
        if debug:
            print(content)


def _extract_json_text(text: str) -> str:
    cleaned = text.strip()
    match = re.search(r"```json\s*(.*?)```", cleaned, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        return match.group(0)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return match.group(0)
    return cleaned


def _parse_json_dict(text: str) -> Dict:
    cleaned = text.strip()
    match = re.search(r"```json\s*(.*?)```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()
    else:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {
        "need_refine": False,
        "reason": "Reflection 输出不是严格 JSON，停止自动优化，交由人工确认。",
        "issues": [],
        "suggestions": [],
    }


def _append_quality_conclusion(plan_text: str, conclusion: str) -> str:
    """统一追加最终质检结论，避免生成阶段和 Reflection 阶段重复输出。"""
    cleaned = re.sub(r"\n## 质检结论\s*\n[\s\S]*?(?=\n## |\Z)", "\n", plan_text).strip()
    cleaned = re.sub(
        r"\n\*\*质检结论\*\*\s*\n[\s\S]*?(?=\n\*\*下一步建议\*\*|\n## 下一步建议|\Z)",
        "\n",
        cleaned,
    ).strip()
    return f"{cleaned}\n\n## 质检结论\n\n{conclusion}"


def _default_plan() -> List[Dict]:
    return [
        {"step": 1, "name": "获取当前时间", "tool": "time", "reason": "解析相对日期"},
        {"step": 2, "name": "检索目的地旅行知识", "tool": "rag", "reason": "获取稳定城市知识"},
        {"step": 3, "name": "查询天气", "tool": "weather", "reason": "补充实时环境"},
        {"step": 4, "name": "查询核心 POI", "tool": "poi", "reason": "验证地点"},
        {"step": 5, "name": "查询酒店参考价", "tool": "hotel", "reason": "判断住宿预算"},
        {"step": 6, "name": "估算预算", "tool": "budget", "reason": "判断预算可行性"},
        {"step": 7, "name": "生成行程", "tool": "draft", "reason": "汇总工具结果"},
        {"step": 8, "name": "质检优化", "tool": "reflection", "reason": "修正问题"},
    ]


def _normalize_plan(raw_plan) -> List[Dict]:
    allowed_tools = {
        "time",
        "rag",
        "weather",
        "poi",
        "geocode",
        "distance",
        "hotel",
        "budget",
        "calculate",
        "draft",
        "reflection",
    }

    if isinstance(raw_plan, dict):
        raw_items = raw_plan.get("steps") or raw_plan.get("plan") or [raw_plan]
    elif isinstance(raw_plan, list):
        raw_items = raw_plan
    else:
        return _default_plan()

    normalized = []
    for index, item in enumerate(raw_items, start=1):
        if isinstance(item, dict):
            tool = str(item.get("tool", "")).strip()
            if tool not in allowed_tools:
                tool = _guess_tool_from_text(
                    f"{item.get('name', '')} {item.get('reason', '')} {item.get('step', '')}"
                )
            normalized.append(
                {
                    "step": item.get("step") or index,
                    "name": item.get("name") or f"步骤{index}",
                    "tool": tool,
                    "reason": item.get("reason") or "Planner 未给出明确理由",
                    "params": item.get("params") if isinstance(item.get("params"), dict) else {},
                }
            )
            continue

        if isinstance(item, str):
            normalized.append(
                {
                    "step": index,
                    "name": item[:40],
                    "tool": _guess_tool_from_text(item),
                    "reason": "Planner 返回了字符串步骤，已自动规范化。",
                }
            )

    normalized = [item for item in normalized if item.get("tool") in allowed_tools]
    return normalized or _default_plan()


def _guess_tool_from_text(text: str) -> str:
    tool_keywords = [
        ("酒店", "hotel"),
        ("住宿", "hotel"),
        ("天气", "weather"),
        ("POI", "poi"),
        ("景点", "poi"),
        ("检索", "rag"),
        ("知识", "rag"),
        ("预算", "budget"),
        ("费用", "budget"),
        ("计算", "calculate"),
        ("距离", "distance"),
        ("经纬度", "geocode"),
        ("路线", "distance"),
        ("时间", "time"),
        ("日期", "time"),
        ("质检", "reflection"),
        ("反思", "reflection"),
        ("生成", "draft"),
    ]
    for keyword, tool in tool_keywords:
        if keyword in text:
            return tool
    return "draft"
