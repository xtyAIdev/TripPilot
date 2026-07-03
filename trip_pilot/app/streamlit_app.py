import sys
import traceback
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trip_pilot.agents.conversation_agent import TripPilotConversationAgent
from trip_pilot.evaluation.evaluator import evaluate_trip_response, render_score_table
from trip_pilot.tools.image_tools import generate_trip_cover_image


st.set_page_config(
    page_title="Agent-TripPilot",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --trip-border: rgba(31, 41, 55, 0.12);
        --trip-soft: #f7f4ee;
        --trip-ink: #111827;
        --trip-muted: #6b7280;
        --trip-accent: #0f766e;
        --trip-accent-2: #b45309;
    }
    .stApp {
        background:
            radial-gradient(circle at 20% 0%, rgba(15, 118, 110, 0.12), transparent 28%),
            radial-gradient(circle at 85% 12%, rgba(180, 83, 9, 0.12), transparent 28%),
            linear-gradient(180deg, #fbfaf7 0%, #f4efe7 46%, #ffffff 100%);
        color: var(--trip-ink);
    }
    .main .block-container {
        max-width: 1360px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.72);
        backdrop-filter: blur(18px);
        border-right: 1px solid var(--trip-border);
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-size: 20px;
        font-weight: 760;
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    .trip-title {
        font-size: 38px;
        font-weight: 820;
        line-height: 1.15;
        margin-bottom: 6px;
    }
    .trip-subtitle {
        color: var(--trip-muted);
        font-size: 14px;
        margin-bottom: 20px;
    }
    .status-row {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin-bottom: 16px;
    }
    .status-item {
        border: 1px solid var(--trip-border);
        border-radius: 12px;
        padding: 14px 16px;
        background: rgba(255,255,255,0.78);
        box-shadow: 0 14px 36px rgba(17, 24, 39, 0.06);
    }
    .status-label {
        color: var(--trip-muted);
        font-size: 12px;
        margin-bottom: 4px;
    }
    .status-value {
        font-size: 15px;
        font-weight: 650;
        overflow-wrap: anywhere;
    }
    .progress-panel {
        border: 1px solid rgba(15, 118, 110, 0.18);
        border-radius: 14px;
        padding: 14px 16px;
        margin: 8px 0 16px 0;
        background: rgba(255, 255, 255, 0.82);
        box-shadow: 0 18px 46px rgba(15, 23, 42, 0.08);
    }
    .progress-title {
        font-size: 14px;
        font-weight: 720;
        margin-bottom: 6px;
    }
    .progress-current {
        color: #334155;
        font-size: 14px;
        font-weight: 620;
        padding: 9px 10px;
        border-radius: 8px;
        background: rgba(15, 118, 110, 0.07);
    }
    div[data-testid="stTabs"] button {
        border-radius: 10px;
        padding: 8px 12px;
    }
    .stButton > button {
        border-radius: 10px;
        border: 1px solid var(--trip-border);
        background: rgba(255, 255, 255, 0.84);
        box-shadow: 0 8px 22px rgba(17, 24, 39, 0.05);
        min-height: 38px;
    }
    .stButton > button:hover {
        border-color: rgba(15, 118, 110, 0.45);
        color: #0f766e;
    }
    .stTextInput input, .stTextArea textarea, .stNumberInput input {
        border-radius: 10px;
    }
    [data-testid="stChatMessage"] {
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.68);
        border: 1px solid rgba(31, 41, 55, 0.08);
        box-shadow: 0 12px 34px rgba(17, 24, 39, 0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_agent() -> TripPilotConversationAgent:
    if "trip_agent" not in st.session_state:
        st.session_state.trip_agent = TripPilotConversationAgent(thread_id="streamlit")
    return st.session_state.trip_agent


def add_message(role: str, content: str) -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.messages.append({"role": role, "content": content})


def run_agent(user_input: str) -> None:
    agent = get_agent()
    add_message("user", user_input)
    progress_slot = st.empty()

    def show_progress(message: str) -> None:
        progress_slot.markdown(
            f"""
            <div class="progress-panel">
              <div class="progress-title">正在处理这次旅行请求</div>
              <div class="progress-current">{message}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    show_progress("正在解析出行需求")
    with st.spinner("正在生成..."):
        try:
            answer = agent.chat(user_input, debug=False, progress_callback=show_progress)
            scores = evaluate_trip_response(
                user_input=user_input,
                response=answer,
                debug_text="\n".join(agent.memory.debug_events[-80:]),
            )
            st.session_state.last_score_table = render_score_table(scores)
            st.session_state.last_error = ""
        except Exception as e:
            error_detail = traceback.format_exc()
            st.session_state.last_error = error_detail
            agent.memory.remember_debug("[UI Error]\n" + error_detail)
            answer = (
                f"运行失败：{type(e).__name__}: {e}\n\n"
                "右侧“错误详情”里可以查看完整 traceback。"
            )
    progress_slot.empty()
    add_message("assistant", answer)


def run_followup_action(user_input: str) -> None:
    agent = get_agent()
    if not agent.memory.last_plan:
        add_message("assistant", "还没有可调整的行程。请先补齐目的地、天数、人数等信息，并生成一版草案或正式行程。")
        return
    run_agent(user_input)


agent = get_agent()
st.session_state.setdefault("messages", [])
st.session_state.setdefault("last_error", "")
st.session_state.setdefault("last_score_table", "")

with st.sidebar:
    st.header("TripPilot")

    origin = st.text_input("始发地", placeholder="郑州")
    destination = st.text_input("目的地", placeholder="杭州")
    start_date = st.text_input("出发日期", placeholder="本周末 / 2026-07-10")
    days = st.number_input("天数", min_value=1, max_value=30, value=2, step=1)
    people = st.number_input("人数", min_value=1, max_value=20, value=2, step=1)
    budget = st.text_input("预算", placeholder="1500，含往返大交通")
    travel_mode = st.selectbox("城际交通", ["暂不确定", "高铁", "普速火车", "飞机", "自驾", "市内为主"])
    plan_mode = st.radio("规划深度", ["草案", "正式"], horizontal=True)
    need_export = st.checkbox("生成后导出 Markdown", value=False)
    preferences = st.text_area("偏好和限制", placeholder="文化、美食、别太累、少走路")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("发送需求", use_container_width=True):
            parts = []
            if origin:
                parts.append(f"我从{origin}出发")
            if destination:
                parts.append(f"去{destination}")
            if start_date:
                parts.append(f"{start_date}出发")
            parts.append(f"{int(days)}天，{int(people)}个人")
            if budget:
                parts.append(f"预算{budget}")
            if travel_mode != "暂不确定":
                parts.append(f"城际交通偏好{travel_mode}")
            if preferences:
                parts.append(preferences)
            if plan_mode == "草案":
                parts.append("先做粗略草案")
            else:
                parts.append("生成正式行程")
            if need_export:
                parts.append("并导出 Markdown")
            run_agent("，".join(parts))
            st.rerun()
    with col_b:
        if st.button("重置", use_container_width=True):
            agent.memory.reset_trip_request()
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.caption("快捷调整")
    quick_a, quick_b = st.columns(2)
    with quick_a:
        if st.button("压缩预算", use_container_width=True):
            run_followup_action("在保留当前行程核心体验的基础上，帮我压缩预算，只说明改了哪里。")
            st.rerun()
        if st.button("增加美食", use_container_width=True):
            run_followup_action("在当前方案里增加美食体验，不要重写全文，只调整相关部分。")
            st.rerun()
    with quick_b:
        if st.button("降低强度", use_container_width=True):
            run_followup_action("把当前行程强度降低一点，第一天和第二天都别太累，只说明改动。")
            st.rerun()
        if st.button("导出 Markdown", use_container_width=True):
            run_agent("导出 Markdown")
            st.rerun()

st.markdown('<div class="trip-title">Agent-TripPilot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="trip-subtitle">智能出行策划与执行工作台</div>',
    unsafe_allow_html=True,
)

request = agent.memory.trip_request
st.markdown(
    f"""
    <div class="status-row">
      <div class="status-item">
        <div class="status-label">目的地</div>
        <div class="status-value">{request.destination or "未填写"}</div>
      </div>
      <div class="status-item">
        <div class="status-label">日期与人数</div>
        <div class="status-value">{request.start_date or "未确认"} / {request.people or "未确认"} 人</div>
      </div>
      <div class="status-item">
        <div class="status-label">预算</div>
        <div class="status-value">{request.budget or "未填写"}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([0.66, 0.34], gap="large")

with left:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("输入一句话，例如：我想周末去杭州，两个人，喜欢文化和美食")
    if user_input:
        run_agent(user_input)
        st.rerun()

with right:
    tab_plan, tab_state, tab_trace, tab_score, tab_cover = st.tabs(
        ["结果", "状态", "过程", "评分", "封面"]
    )

    with tab_plan:
        if agent.memory.last_plan:
            st.markdown(agent.memory.last_plan)
            st.download_button(
                "下载当前 Markdown",
                data=agent.memory.last_plan,
                file_name="trip_plan.md",
                mime="text/markdown",
                use_container_width=True,
            )
        else:
            st.info("生成行程后，这里会展示当前方案。")

    with tab_state:
        snapshot = agent.memory.state_snapshot()
        st.json(snapshot.model_dump())
        if snapshot.open_questions:
            st.markdown("**待确认项**")
            for item in snapshot.open_questions:
                st.write(f"- {item.question}：{item.reason}")

    with tab_trace:
        if st.session_state.last_error:
            st.markdown("**错误详情**")
            st.code(st.session_state.last_error, language="python")
        debug_text = "\n\n".join(agent.memory.debug_events[-80:])
        st.code(debug_text or "暂无调试信息", language="text")

    with tab_score:
        if st.session_state.last_score_table:
            st.markdown(st.session_state.last_score_table)
        else:
            st.info("完成一次对话后，这里会显示本轮评分。")

    with tab_cover:
        cover_prompt = st.text_area(
            "图片提示词",
            value="高级旅行杂志封面，杭州城市旅行，西湖、江南建筑、清爽自然光，写实摄影风格，无文字",
            height=110,
        )
        if st.button("生成封面图", use_container_width=True):
            with st.spinner("Seedream 正在生成图片..."):
                result = generate_trip_cover_image.invoke(
                    {"prompt": cover_prompt, "file_name": "trip_cover.png"}
                )
            st.info(result)
