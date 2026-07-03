from trip_pilot.agents.conversation_agent import TripPilotConversationAgent


def main():
    print("--------------- Agent-TripPilot 智能出行策划 Agent ---------------")
    print("你可以先聊天，也可以逐步补充出行需求。")
    print("示例：")
    print("我从郑州出发，周末两天去杭州，两个人，预算1500，喜欢文化和美食，别太累。")
    print("当你想正式生成时，输入：生成行程。")
    print("当你想保存文件时，输入：导出 Markdown。")
    print("输入 reset 重置当前出行需求，输入 exit 退出。")

    agent = TripPilotConversationAgent()

    while True:
        user_input = input("\n你：").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            break
        if user_input.lower() in {"reset", "重置"}:
            agent.memory.reset_trip_request()
            print("Agent：已重置当前出行需求。")
            continue
        if not user_input:
            continue

        try:
            answer = agent.chat(user_input, debug=True)
            print("\nAgent：")
            print(answer)
        except Exception as e:
            print(f"运行失败：{e}")


if __name__ == "__main__":
    main()
