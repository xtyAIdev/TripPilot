from trip_pilot.schemas import ConstraintCheck, OpenQuestion, TripRequest


def validate_constraints(request: TripRequest, plan_text: str = "") -> ConstraintCheck:
    """用规则做一层轻量约束检查，避免只依赖 LLM 自我评价。"""
    issues: list[str] = []
    suggestions: list[str] = []
    open_questions: list[OpenQuestion] = []

    required_fields = {
        "origin": "始发地",
        "destination": "目的地城市",
        "start_date": "出发日期",
        "days": "旅行天数",
        "people": "出行人数",
    }
    for field, label in required_fields.items():
        if getattr(request, field) in (None, "", []):
            open_questions.append(
                OpenQuestion(
                    field=field,
                    question=f"请确认{label}。",
                    reason="正式规划需要该信息，否则天气、预算或节奏判断不稳定。",
                )
            )

    if request.budget is not None and not request.budget_scope:
        open_questions.append(
            OpenQuestion(
                field="budget_scope",
                question="请确认预算是否包含往返大交通。",
                reason="这会直接影响酒店和当地消费的可用空间。",
            )
        )

    if request.days is not None and request.days <= 0:
        issues.append("旅行天数必须大于 0。")
        suggestions.append("请把天数修正为 1 天或以上。")

    if request.people is not None and request.people <= 0:
        issues.append("出行人数必须大于 0。")
        suggestions.append("请把人数修正为 1 人或以上。")

    if request.budget is not None and request.people and request.days:
        per_person_day = request.budget / max(request.people, 1) / max(request.days, 1)
        if per_person_day < 250:
            issues.append("人均每日预算偏紧，住宿、餐饮和门票需要明显压缩。")
            suggestions.append("优先选择经济型住宿、公共交通和免费/低价景点。")

    joined_constraints = " ".join(request.constraints + request.preferences)
    if any(keyword in joined_constraints for keyword in ["别太累", "轻松", "少走路"]):
        if plan_text and not any(keyword in plan_text for keyword in ["轻松", "少走路", "午休", "减少换乘"]):
            issues.append("用户要求轻松或少走路，但方案文本中缺少明确节奏安排。")
            suggestions.append("减少每日景点数量，加入午休或就近动线说明。")

    if request.preferences and plan_text:
        missing_preferences = [item for item in request.preferences if item and item not in plan_text]
        if missing_preferences:
            issues.append("部分用户偏好没有在方案中明确体现：" + "、".join(missing_preferences[:3]))
            suggestions.append("在每日主题或景点选择中补足这些偏好。")

    score = 100 - len(issues) * 12 - len(open_questions) * 6
    score = max(min(score, 100), 0)
    return ConstraintCheck(
        passed=not issues and not open_questions,
        score=score,
        issues=issues,
        suggestions=suggestions,
        open_questions=open_questions,
    )


def format_constraint_check(check: ConstraintCheck) -> str:
    lines = [f"约束校验分：{check.score}/100"]
    if check.issues:
        lines.append("问题：" + "；".join(check.issues))
    if check.suggestions:
        lines.append("建议：" + "；".join(check.suggestions))
    if check.open_questions:
        lines.append("待确认：" + "；".join(item.question for item in check.open_questions))
    if len(lines) == 1:
        lines.append("没有发现明显硬性约束问题。")
    return "\n".join(lines)
