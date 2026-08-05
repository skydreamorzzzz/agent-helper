from __future__ import annotations

import ast
import operator

from pydantic import BaseModel, Field

from src.planner.models import RiskLevel
from src.tools.base import Tool


class CalculatorArguments(BaseModel):
    expression: str = Field(min_length=1, max_length=500)


class CalculatorTool(Tool):
    name = "calculator"
    description = "Safely evaluate a basic arithmetic expression. Supports +, -, *, /, //, %, **, parentheses, and numeric literals."
    argument_schema = CalculatorArguments
    risk_level = RiskLevel.READ_ONLY

    def execute(self, arguments: CalculatorArguments) -> str:
        value = _safe_eval(arguments.expression)
        return str(value)


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(expression: str) -> int | float:
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BINARY_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent too large")
        return _BINARY_OPERATORS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return _UNARY_OPERATORS[op_type](_eval_node(node.operand))

    raise ValueError(f"Unsupported expression node: {type(node).__name__}")
