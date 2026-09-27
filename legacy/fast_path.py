"""
Intent matching for requests that need no model.

Most of MyAgent's tools are deterministic local Python: run some code, convert
a unit, count words, do arithmetic, read the clock, list tasks. Sending those
through an LLM buys nothing. For "Run this Python code: print(...)" the model's
entire contribution is reading the words "run this python code" and emitting a
run_code call -- which a regex does for free, offline, in microseconds, and
without a bill. The subprocess that actually runs the code is the same either
way.

This module decides ONE thing: which tool, and with which arguments. It
executes nothing and implements no tool. execute_tool() in app.py stays the
single implementation, so the fast path and the model path cannot drift apart
about what a tool actually does -- they call the same function.

Returning None means "this needs a model", and that is the default. Anything
conversational, anything needing judgement, anything ambiguous, anything
needing live data (a currency rate, the news) falls through to react(). A
wrong fast-path match is worse than an API call: it answers the wrong question
instantly and confidently, and the user has no reason to doubt it.
"""
from __future__ import annotations

import datetime
import re

# ── units ─────────────────────────────────────────────────────────────────────
# Maps what people type to the keys execute_tool()'s conversion table uses.
# If a spelling is not here the match fails and the model handles it, which is
# the right outcome -- guessing at a unit is how you convert to the wrong one.
UNIT_ALIASES = {
    "km": "km", "kilometer": "km", "kilometers": "km", "kilometre": "km", "kilometres": "km",
    "m": "m", "meter": "m", "meters": "m", "metre": "m", "metres": "m",
    "cm": "cm", "centimeter": "cm", "centimeters": "cm", "centimetre": "cm", "centimetres": "cm",
    "mile": "miles", "miles": "miles", "mi": "miles",
    "ft": "feet", "foot": "feet", "feet": "feet",
    "in": "inch", "inch": "inch", "inches": "inch",
    "kg": "kg", "kilo": "kg", "kilos": "kg", "kilogram": "kg", "kilograms": "kg",
    "g": "g", "gram": "g", "grams": "g",
    "lb": "lb", "lbs": "lb", "pound": "lb", "pounds": "lb",
    "oz": "oz", "ounce": "oz", "ounces": "oz",
    "gb": "gb", "mb": "mb", "kb": "kb", "tb": "tb",
    "gigabyte": "gb", "gigabytes": "gb", "megabyte": "mb", "megabytes": "mb",
    "kilobyte": "kb", "kilobytes": "kb", "terabyte": "tb", "terabytes": "tb",
    "kmh": "kmh", "km/h": "kmh", "kph": "kmh", "kmph": "kmh",
    "ms": "ms", "m/s": "ms",
    "mph": "mph",
    "sqm": "sqm", "sqft": "sqft", "hectare": "hectare", "hectares": "hectare",
    "c": "celsius", "celsius": "celsius", "celcius": "celsius", "centigrade": "celsius",
    "f": "fahrenheit", "fahrenheit": "fahrenheit", "farenheit": "fahrenheit",
    "k": "kelvin", "kelvin": "kelvin",
}

# Currency codes look exactly like units in "convert 100 USD to INR", and they
# are the one conversion that needs a live rate. Name them so they fall through
# rather than failing inside the converter with a confusing message.
CURRENCIES = {
    "usd", "inr", "eur", "gbp", "jpy", "aud", "cad", "chf", "cny", "sgd",
    "aed", "rs", "rupee", "rupees", "dollar", "dollars", "euro", "euros", "pound_sterling",
}

MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

# What a calculator expression may contain once the words are stripped. Note
# this is a gate on the fast path, not a security boundary -- execute_tool()
# does its own eval in a restricted namespace and is what actually has to hold.
MATH_CHARS = re.compile(r"^[0-9\s+\-*/%().,^]+$")
MATH_FUNC = re.compile(r"^[0-9a-z_\s+\-*/%().,^]+$")
MATH_NAMES = re.compile(r"[a-z_]+")
MATH_ALLOWED_NAMES = {
    "math", "sqrt", "log", "log10", "log2", "exp", "sin", "cos", "tan", "asin",
    "acos", "atan", "pi", "e", "tau", "floor", "ceil", "fabs", "factorial",
    "degrees", "radians", "pow", "abs", "round", "min", "max", "sum", "hypot",
}


def _num(text: str) -> float:
    """Parse a number that may carry thousands separators."""
    return float(text.replace(",", ""))


def _tidy(value: float) -> float | int:
    return int(value) if value == int(value) else value


def _parse_date(text: str) -> str | None:
    """A date in any of the forms people actually type -> ISO, or None.

    Deliberately narrow. dateutil would accept far more, but it is a
    dependency and its guesses on ambiguous input are exactly the kind of
    silent wrongness this module must not produce.
    """
    t = text.strip().strip("?.,").lower()
    t = re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", t)   # 31st -> 31
    t = re.sub(r"\s+", " ", t).replace(" of ", " ")

    if t in ("today", "now"):
        return datetime.date.today().isoformat()
    if t == "tomorrow":
        return (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
    if t == "yesterday":
        return (datetime.date.today() - datetime.timedelta(days=1)).isoformat()

    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", t)
    if m:
        y, mo, d = (int(g) for g in m.groups())
    else:
        # 31 December 2026 / 31 dec 2026
        m = re.fullmatch(r"(\d{1,2})[ ,]+([a-z]+)[ ,]+(\d{4})", t)
        if m and m.group(2) in MONTHS:
            d, mo, y = int(m.group(1)), MONTHS[m.group(2)], int(m.group(3))
        else:
            # December 31 2026 / dec 31, 2026
            m = re.fullmatch(r"([a-z]+)[ ,]+(\d{1,2})[ ,]+(\d{4})", t)
            if m and m.group(1) in MONTHS:
                mo, d, y = MONTHS[m.group(1)], int(m.group(2)), int(m.group(3))
            else:
                # 31/12/2026 -- day first, because the rest of this file is
                # written for a user who writes "31 December". A bare
                # ambiguous 05/06/2026 is a coin flip either way, so it is
                # read the same way as the unambiguous forms rather than
                # switching convention halfway.
                m = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", t)
                if not m:
                    return None
                d, mo, y = (int(g) for g in m.groups())

    try:
        return datetime.date(y, mo, d).isoformat()   # rejects 31 February
    except ValueError:
        return None


def _looks_like_math(expr: str) -> bool:
    """True if this is arithmetic and not a sentence that happens to hold digits."""
    expr = expr.strip()
    if not expr or not re.search(r"\d", expr):
        return False
    if not re.search(r"[+\-*/%^]|math\.", expr):
        return False                                  # a bare number is not a sum
    if MATH_CHARS.fullmatch(expr):
        return True
    if not MATH_FUNC.fullmatch(expr.lower()):
        return False
    return all(name in MATH_ALLOWED_NAMES for name in MATH_NAMES.findall(expr.lower()))


def match(message: str):
    """(tool_name, args) for a request that needs no model, else None."""
    raw = (message or "").strip()
    if not raw:
        return None
    low = raw.lower()

    # ── run code ──────────────────────────────────────────────────────────────
    fence = re.search(r"```(?:python|py)?\s*\n?(.*?)```", raw, re.S)
    if fence and fence.group(1).strip():
        return "run_code", {"code": fence.group(1).strip()}

    m = re.match(
        r"^(?:please\s+)?(?:run|execute|eval(?:uate)?)\s+"
        r"(?:this\s+|the\s+following\s+|some\s+)?(?:python\s*)?(?:code|script|snippet)?\s*:?\s*(.+)$",
        raw, re.S | re.I,
    )
    if m and m.group(1).strip():
        return "run_code", {"code": m.group(1).strip()}

    # ── unit conversion ───────────────────────────────────────────────────────
    m = re.match(
        r"^(?:convert\s+|what(?:'s| is)\s+)?(-?[\d,]+(?:\.\d+)?)\s*"
        r"(°?[a-z/]+)\s+(?:to|in|into|=)\s+(°?[a-z/]+)\s*\??$",
        low,
    )
    if m:
        src, dst = m.group(2).lstrip("°"), m.group(3).lstrip("°")
        if src in CURRENCIES or dst in CURRENCIES:
            return None                               # needs a live rate
        if src in UNIT_ALIASES and dst in UNIT_ALIASES:
            return "unit_convert", {
                "value": _tidy(_num(m.group(1))),
                "from_unit": UNIT_ALIASES[src],
                "to_unit": UNIT_ALIASES[dst],
            }
        return None

    # ── word count ────────────────────────────────────────────────────────────
    m = re.match(
        r"^(?:count\s+(?:the\s+)?words?\s+(?:in|of)|word\s*count(?:\s+for)?|"
        r"how\s+many\s+words?\s+(?:are\s+)?(?:in|is)?)\s*:?\s*(.+)$",
        raw, re.S | re.I,
    )
    if m and m.group(1).strip():
        return "word_count", {"text": m.group(1).strip()}

    # ── dates ─────────────────────────────────────────────────────────────────
    m = re.match(r"^(?:how\s+many\s+)?days?\s+(?:until|till|to|since|from)\s+(.+?)\s*\??$", low)
    if m:
        iso = _parse_date(m.group(1))
        if iso:
            return "date_diff", {"date1": "today", "date2": iso}
        return None

    m = re.match(r"^(?:how\s+many\s+)?days?\s+between\s+(.+?)\s+and\s+(.+?)\s*\??$", low)
    if m:
        a, b = _parse_date(m.group(1)), _parse_date(m.group(2))
        if a and b:
            return "date_diff", {"date1": a, "date2": b}
        return None

    if re.fullmatch(
        r"(?:what(?:'s|s| is)?\s+)?(?:the\s+)?(?:current\s+|today'?s?\s+)?"
        r"(?:date\s+and\s+time|time\s+and\s+date|date|time|day)"
        r"(?:\s+(?:is\s+it|today|now|right\s+now))*\s*\??",
        low,
    ):
        return "get_datetime", {}

    # ── lists ─────────────────────────────────────────────────────────────────
    m = re.fullmatch(
        r"(?:what\s+are\s+|show\s+|list\s+|view\s+)?(?:my\s+|all\s+)?"
        r"(pending\s+|completed\s+|done\s+|high\s+priority\s+|medium\s+priority\s+|low\s+priority\s+)?"
        r"tasks?\s*(?:list)?\s*\??",
        low,
    )
    if m:
        word = (m.group(1) or "pending").strip()
        kind = {"completed": "done", "done": "done", "high priority": "high",
                "medium priority": "medium", "low priority": "low"}.get(word, "pending")
        return "list_tasks", {"filter": kind}

    # "pending high priority tasks" -- two qualifiers, which the pattern above
    # takes one of. Priority wins: it is the narrower ask, and list_tasks
    # already excludes completed tasks from every priority filter.
    m = re.fullmatch(
        r"(?:what\s+are\s+|show\s+|list\s+)?(?:my\s+)?pending\s+"
        r"(high|medium|low)(?:\s+priority)?\s+tasks?\s*\??",
        low,
    )
    if m:
        return "list_tasks", {"filter": m.group(1)}

    if re.fullmatch(r"(?:show\s+|list\s+|view\s+)?(?:my\s+|all\s+)?notes?\s*\??", low):
        return "list_notes", {}

    if re.fullmatch(r"(?:show\s+|list\s+|view\s+)?(?:my\s+|all\s+)?files?\s*\??", low):
        return "list_files", {}

    if re.fullmatch(r"(?:what\s+do\s+you\s+)?(?:remember|recall)(?:\s+about\s+me)?\s*\??", low):
        return "recall_memory", {}

    # ── password ──────────────────────────────────────────────────────────────
    m = re.match(
        r"^(?:generate|create|make|give\s+me)\s+(?:a|an)?\s*(?:new\s+)?(?:secure\s+|strong\s+|random\s+)*"
        r"(?:(\d{1,3})[- ]?(?:char(?:acter)?s?)?\s+)?password\s*(?:of\s+(\d{1,3})\s*(?:char(?:acter)?s?)?)?"
        r"\s*(?:\(?\s*(no|without)\s+symbols?\s*\)?)?\s*\??$",
        low,
    )
    if m:
        length = m.group(1) or m.group(2)
        args = {"length": max(8, min(128, int(length)))} if length else {}
        if m.group(3):
            args["include_symbols"] = False
        return "generate_password", args

    # ── arithmetic ────────────────────────────────────────────────────────────
    # "15% of 85000" is the one phrasing worth rewriting, because it is the
    # single most common thing people type at an assistant and Python has no
    # percent-of operator.
    m = re.fullmatch(
        r"(?:what(?:'s| is)\s+|calculate\s+|compute\s+|how\s+much\s+is\s+)?"
        r"([\d,.]+)\s*(?:%|percent)\s+of\s+([\d,.]+)\s*\??",
        low,
    )
    if m:
        return "calculator", {"expression": f"{_num(m.group(2))}*{_num(m.group(1))}/100"}

    expr = re.sub(
        r"^(?:what(?:'s| is)|calculate|compute|how\s+much\s+is|solve|evaluate|eval)\s+",
        "", low,
    ).strip().rstrip("=?").strip()
    if _looks_like_math(expr):
        return "calculator", {"expression": expr.replace("^", "**")}

    return None
