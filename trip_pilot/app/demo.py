from trip_pilot.agents.trip_agent import generate_trip_plan


def main():
    """固定样例，一键验证 MVP 主流程。"""
    user_input = "我从郑州出发，周末两天去杭州，两个人，预算1500，喜欢文化和美食，别太累。"
    generate_trip_plan(user_input)


if __name__ == "__main__":
    main()

