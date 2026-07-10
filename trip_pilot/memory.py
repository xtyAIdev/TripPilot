from dataclasses import dataclass, field
from typing import Dict, List

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage

from trip_pilot.schemas import AgentStepEvent, OpenQuestion, TripRequest, TripState, UserProfile


@dataclass
class TripPilotMemory:
    """会话记忆：消息历史 + 结构化出行需求。"""

    thread_id: str = "default"
    chat_history: InMemoryChatMessageHistory = field(default_factory=InMemoryChatMessageHistory)
    trip_request: TripRequest = field(default_factory=TripRequest)
    user_profile: UserProfile = field(default_factory=UserProfile)
    last_plan: str = ""
    open_questions: List[OpenQuestion] = field(default_factory=list)
    tool_cache: Dict[str, str] = field(default_factory=dict)
    last_tool_summary: Dict[str, str] = field(default_factory=dict)
    step_events: List[AgentStepEvent] = field(default_factory=list)
    debug_events: List[str] = field(default_factory=list)
    phase: str = "idle"
    reply_mode: str = "standard"
    plan_mode: str = "official"

    def add_user_message(self, content: str) -> None:
        self.chat_history.add_message(HumanMessage(content=content))

    def add_ai_message(self, content: str) -> None:
        self.chat_history.add_message(AIMessage(content=content))

    def remember_debug(self, content: str) -> None:
        self.debug_events.append(content)
        if len(self.debug_events) > 120:
            self.debug_events = self.debug_events[-120:]

    def remember_step(
        self,
        stage: str,
        title: str,
        detail: str = "",
        status: str = "running",
    ) -> None:
        if self.step_events and self.step_events[-1].status == "running":
            self.step_events[-1].status = "done"
        self.step_events.append(
            AgentStepEvent(stage=stage, title=title, detail=detail, status=status)
        )
        if len(self.step_events) > 80:
            self.step_events = self.step_events[-80:]

    def history_summary(self, limit: int = 6) -> str:
        messages = self.chat_history.messages[-limit:]
        lines = []
        for msg in messages:
            role = "用户" if msg.type == "human" else "助手"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def state_snapshot(self) -> TripState:
        return TripState(
            user_profile=self.user_profile,
            current_trip_requirements=self.trip_request,
            current_trip_plan=self.last_plan,
            open_questions=self.open_questions,
            latest_constraints=self.trip_request.constraints,
            last_tool_summary=self.last_tool_summary,
            phase=self.phase,
            reply_mode=self.reply_mode,
            plan_mode=self.plan_mode,
        )

    def state_summary(self) -> str:
        snapshot = self.state_snapshot()
        return snapshot.model_dump_json(indent=2, ensure_ascii=False)

    def reset_trip_request(self) -> None:
        self.trip_request = TripRequest()
        self.last_plan = ""
        self.open_questions = []
        self.tool_cache = {}
        self.last_tool_summary = {}
        self.step_events = []
        self.debug_events = []
        self.phase = "idle"
        self.reply_mode = "standard"
        self.plan_mode = "official"
