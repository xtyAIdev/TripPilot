from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TripRequest(BaseModel):
    """用户出行需求的结构化结果。"""

    origin: Optional[str] = Field(default=None, description="始发地")
    destination: Optional[str] = Field(default=None, description="目的地城市")
    start_date: Optional[str] = Field(default=None, description="出发日期")
    days: Optional[int] = Field(default=None, description="旅行天数")
    people: Optional[int] = Field(default=None, description="出行人数")
    budget: Optional[float] = Field(default=None, description="总预算，单位元")
    budget_scope: Optional[str] = Field(default=None, description="预算范围，例如含往返大交通/仅当地花费/不确定")
    travel_mode: Optional[str] = Field(default=None, description="城际交通偏好")
    city_mode: Optional[str] = Field(default=None, description="市内交通偏好")
    accommodation_preference: Optional[str] = Field(default=None, description="住宿偏好")
    preferences: List[str] = Field(default_factory=list, description="兴趣偏好")
    constraints: List[str] = Field(default_factory=list, description="限制条件")
    need_report: bool = Field(default=False, description="用户是否明确要求生成正式行程")
    need_export: bool = Field(default=False, description="用户是否明确要求导出文件")

    def merge(self, other: "TripRequest") -> "TripRequest":
        """把新抽取的信息合并到旧状态，空值不覆盖已有值。"""
        data = self.model_dump()
        other_data = other.model_dump()

        for key, value in other_data.items():
            if key in {"preferences", "constraints"}:
                merged = list(dict.fromkeys(data.get(key, []) + (value or [])))
                data[key] = merged
            elif key in {"need_report", "need_export"}:
                data[key] = bool(data.get(key)) or bool(value)
            elif value not in (None, "", []):
                data[key] = value

        return TripRequest.model_validate(data)


class UserProfile(BaseModel):
    """稳定用户偏好，后续可以升级为长期记忆。"""

    preferred_pace: Optional[str] = Field(default=None, description="节奏偏好，例如轻松/紧凑")
    hotel_preference: Optional[str] = Field(default=None, description="长期住宿偏好")
    food_preference: List[str] = Field(default_factory=list, description="长期饮食偏好")
    travel_style: List[str] = Field(default_factory=list, description="长期旅行风格")


class OpenQuestion(BaseModel):
    """仍需要用户确认的问题。"""

    field: str
    question: str
    reason: str = ""


class ConstraintCheck(BaseModel):
    """约束校验结果，用于生成前和反思前的硬规则检查。"""

    passed: bool
    score: int = Field(default=100, ge=0, le=100)
    issues: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    open_questions: List[OpenQuestion] = Field(default_factory=list)


class TripState(BaseModel):
    """当前旅行任务的显式状态，后续可直接迁移为 LangGraph state。"""

    user_profile: UserProfile = Field(default_factory=UserProfile)
    current_trip_requirements: TripRequest = Field(default_factory=TripRequest)
    current_trip_plan: str = ""
    open_questions: List[OpenQuestion] = Field(default_factory=list)
    latest_constraints: List[str] = Field(default_factory=list)
    last_tool_summary: Dict[str, str] = Field(default_factory=dict)
    phase: str = Field(default="idle", description="当前阶段：idle/collecting/planning/follow_up/done")
    reply_mode: str = Field(default="standard", description="brief/standard/detailed")
    plan_mode: str = Field(default="official", description="draft/official")


class ConversationStateSnapshot(TripState):
    """兼容旧名称，实际等同于 TripState。"""


class IntentResult(BaseModel):
    """用户当前输入的意图识别结果。"""

    intent: str = Field(description="chat/trip_plan/update/export/cancel")
    confidence: float = Field(default=0.5, description="置信度，0到1")
    reason: str = Field(default="", description="判断理由")


class BudgetItem(BaseModel):
    """预算明细。"""

    name: str
    amount: float
    note: str = ""


class DayPlan(BaseModel):
    """单日行程。"""

    day: int
    theme: str
    morning: str
    afternoon: str
    evening: str
    transport_note: str = ""
    food_note: str = ""


class QualityReport(BaseModel):
    """行程质检结果。"""

    passed: bool
    problems: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


class TripPlan(BaseModel):
    """最终行程结果。"""

    request: TripRequest
    summary: str
    days: List[DayPlan]
    budget_items: List[BudgetItem] = Field(default_factory=list)
    rag_references: List[str] = Field(default_factory=list)
    quality_report: Optional[QualityReport] = None
