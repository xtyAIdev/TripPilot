from langchain.messages import HumanMessage, SystemMessage

from trip_pilot.models import get_reflection_model


REFLECTION_SYSTEM_PROMPT = """
你是一个严格的旅行行程质检员。
请检查行程是否存在以下问题：
1. 时间过紧或路线绕路
2. 预算明显不合理
3. 没有满足用户偏好
4. 对实时信息过度确定
5. 缺少必要提醒，比如预约、开放时间、交通拥堵

请输出简洁的中文质检报告。
不要使用 emoji 或特殊装饰符号，避免 Windows 终端编码问题。
"""


def reflect_trip_plan(user_request: str, draft_plan: str) -> str:
    """对行程初稿做质检。"""
    model = get_reflection_model()
    prompt = f"""
# 用户需求
{user_request}

# 行程初稿
{draft_plan}

请给出质检结果和修改建议。
"""
    response = model.invoke(
        [
            SystemMessage(content=REFLECTION_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )
    return response.content
