from datetime import datetime, timedelta
import ast
import operator

from langchain.tools import tool


SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def current_time_text() -> str:
    """返回当前运行时间，供提示词注入。"""
    now = datetime.now()
    return (
        f"当前系统时间：{now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"今天日期：{now.strftime('%Y-%m-%d')}\n"
        f"明天日期：{(now + timedelta(days=1)).strftime('%Y-%m-%d')}\n"
        f"本周末参考：周六 {(now + timedelta(days=(5 - now.weekday()) % 7)).strftime('%Y-%m-%d')}，"
        f"周日 {(now + timedelta(days=(6 - now.weekday()) % 7)).strftime('%Y-%m-%d')}"
    )


@tool
def get_current_time() -> str:
    """获取当前系统时间，用于解析今天、明天、本周末等相对日期。"""
    return current_time_text()


@tool
def calculate(expression: str) -> str:
    """安全计算纯数学表达式，例如 '(1500 - 520) / 2'。"""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
        return str(result)
    except Exception as e:
        return f"计算失败：{e}"


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in SAFE_OPERATORS:
        return SAFE_OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_OPERATORS:
        return SAFE_OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError("只允许数字和基础数学运算符")

