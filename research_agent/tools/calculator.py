"""Safe calculator: evaluates arithmetic by walking the AST (no eval())."""
from __future__ import annotations

import ast
import math
import operator
import re

from .base import Param, Tool, ToolContext, ToolError, ToolOutput

_BIN = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "log2": math.log2,
    "exp": math.exp, "abs": abs, "round": round, "min": min, "max": max,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "ceil": math.ceil, "floor": math.floor,
}
_CONSTS = {"pi": math.pi, "e": math.e}
_MAX_EXPONENT = 1000
_MAX_RESULT_BITS = 4096  # 10 ** 1000 fits; (2 ** 1000) ** 1000 would tie up a thread for minutes


def evaluate(expression: str) -> float:
    # "1,250,000" -> "1250000" (thousands separators), "15%" -> "(15/100)"
    expr = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expression.strip())
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%(?!\s*[\d.(])", r"(\1/100)", expr).replace("^", "**")
    if len(expr) > 300:
        raise ToolError("expression too long")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        raise ToolError(f"could not parse expression {expression!r}")
    return _eval(tree.body)


def _eval(node: ast.AST):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > _MAX_EXPONENT:
                raise ToolError("exponent too large")
            if abs(left) > 1 and math.log2(abs(left)) * right > _MAX_RESULT_BITS:
                raise ToolError("result too large")
        try:
            return _BIN[type(node.op)](left, right)
        except ZeroDivisionError:
            raise ToolError("division by zero")
        except OverflowError:
            raise ToolError("result too large")
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        try:
            return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
        except (ValueError, TypeError, OverflowError) as e:
            raise ToolError(f"math error in {node.func.id}(): {e}")
    raise ToolError(f"unsupported syntax: {ast.dump(node)[:60]}. Allowed: numbers, + - * / // % **, "
                    f"{', '.join(_FUNCS)}, pi, e")


def _format(value) -> str:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ToolError("result is not a finite number")
        return f"{value:,.6f}".rstrip("0").rstrip(".")
    return f"{value:,}"


def _run(args: dict, ctx: ToolContext) -> ToolOutput:
    value = evaluate(args["expression"])
    return ToolOutput(text=f"{args['expression']} = {_format(value)}", data=value)


calculator = Tool(
    name="calculator",
    description="Exact arithmetic. Use for ANY numeric computation (growth rates, differences, "
                "percentages, averages). Supports + - * / // % ** and sqrt, log, exp, round, min, max.",
    params={"expression": Param("string", "e.g. '(1250 - 980) / 980 * 100'", required=True)},
    func=_run,
    timeout=5,
)
