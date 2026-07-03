from trip_pilot.agents.conversation_agent import TripPilotConversationAgent


def generate_trip_plan(user_input: str, save_output: bool = True, debug: bool = True) -> str:
    """内部使用多轮 Agent。"""
    agent = TripPilotConversationAgent()
    request = user_input

    if "生成" not in request and "规划" not in request and "行程" not in request:
        request = request + " 请生成正式行程。"
    if save_output and "导出" not in request and "保存" not in request:
        request = request + " 并导出 Markdown 文件。"

    return agent.chat(request, debug=debug)
