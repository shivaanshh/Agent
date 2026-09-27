"""
What the fast path must and must not claim.

The must-nots matter more than the musts. A missed match costs an API call;
a wrong match answers a question the user did not ask, instantly, with no
signal that anything went wrong.
"""
import datetime
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "legacy"))

from fast_path import match  # noqa: E402


# ── the quick-action chips in templates/index.html ────────────────────────────
# These are the exact strings the buttons send. If a chip stops matching, it
# silently starts costing money again, so they are pinned here.
def test_run_code_chip():
    assert match("Run this Python code: print([x**2 for x in range(1,11)])") == (
        "run_code", {"code": "print([x**2 for x in range(1,11)])"})


def test_word_count_chip():
    assert match("Count the words in: The quick brown fox jumps over the lazy dog") == (
        "word_count", {"text": "The quick brown fox jumps over the lazy dog"})


def test_convert_units_chip():
    assert match("Convert 5 km to miles") == (
        "unit_convert", {"value": 5, "from_unit": "km", "to_unit": "miles"})


def test_percent_chip():
    assert match("What is 15% of 85000?") == ("calculator", {"expression": "85000.0*15.0/100"})


def test_days_until_chip():
    assert match("How many days until 31 December 2026?") == (
        "date_diff", {"date1": "today", "date2": "2026-12-31"})


def test_high_priority_chip():
    assert match("What are my pending high priority tasks?") == (
        "list_tasks", {"filter": "high"})


def test_currency_chip_falls_through():
    # Needs a live rate. Matching it locally would invent an exchange rate.
    assert match("Convert 100 USD to INR") is None


def test_summarize_chip_falls_through():
    assert match("Summarize the text I will paste below in brief") is None


# ── code ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("text", [
    "run this code: print(1)",
    "Execute this Python: print(1)",
    "please run python code print(1)",
])
def test_code_phrasings(text):
    assert match(text) == ("run_code", {"code": "print(1)"})


def test_fenced_block_keeps_newlines():
    tool, args = match("```python\nfor i in range(3):\n    print(i)\n```")
    assert tool == "run_code"
    assert args["code"] == "for i in range(3):\n    print(i)"


def test_fence_wins_over_prose():
    # The prose prefix would otherwise swallow the backticks into the code.
    tool, args = match("Run this for me:\n```\nprint(2)\n```")
    assert (tool, args) == ("run_code", {"code": "print(2)"})


# ── units ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("text,expected", [
    ("convert 10 kilometers to miles", {"value": 10, "from_unit": "km", "to_unit": "miles"}),
    ("2.5 kg in lbs", {"value": 2.5, "from_unit": "kg", "to_unit": "lb"}),
    ("100 celsius to fahrenheit", {"value": 100, "from_unit": "celsius", "to_unit": "fahrenheit"}),
    ("convert 60 km/h to m/s", {"value": 60, "from_unit": "kmh", "to_unit": "ms"}),
    ("1,500 mb to gb", {"value": 1500, "from_unit": "mb", "to_unit": "gb"}),
])
def test_units(text, expected):
    assert match(text) == ("unit_convert", expected)


def test_unknown_unit_falls_through():
    # Better an API call than a confident conversion between made-up units.
    assert match("convert 5 furlongs to smoots") is None


# ── dates ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("text,iso", [
    ("days until 2026-12-25", "2026-12-25"),
    ("How many days until December 31, 2026?", "2026-12-31"),
    ("days until 31st December 2026", "2026-12-31"),
    ("days until 25/12/2026", "2026-12-25"),
])
def test_days_until(text, iso):
    assert match(text) == ("date_diff", {"date1": "today", "date2": iso})


def test_days_between():
    assert match("days between 2026-01-01 and 2026-03-01") == (
        "date_diff", {"date1": "2026-01-01", "date2": "2026-03-01"})


def test_impossible_date_falls_through():
    # 31 February parses by shape and is not a date. Handing it to the model
    # gets the user a sentence about it; handling it here would get them a
    # number silently computed from 3 March.
    assert match("days until 2026-02-31") is None


def test_tomorrow_resolves():
    tool, args = match("days until tomorrow")
    assert tool == "date_diff"
    assert args["date2"] == (datetime.date.today() + datetime.timedelta(days=1)).isoformat()


@pytest.mark.parametrize("text", ["what is the date", "what time is it", "today's date"])
def test_clock(text):
    assert match(text) == ("get_datetime", {})


# ── lists ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("text,kind", [
    ("list tasks", "pending"),
    ("show my tasks", "pending"),
    ("my completed tasks", "done"),
    ("list high priority tasks", "high"),
])
def test_task_lists(text, kind):
    assert match(text) == ("list_tasks", {"filter": kind})


def test_notes_and_files():
    assert match("list notes") == ("list_notes", {})
    assert match("show my files") == ("list_files", {})


# ── passwords ─────────────────────────────────────────────────────────────────
def test_password_default():
    assert match("generate a secure password") == ("generate_password", {})


def test_password_length():
    assert match("generate a 24 character password") == ("generate_password", {"length": 24})
    assert match("create a password of 32 characters") == ("generate_password", {"length": 32})


def test_password_no_symbols():
    tool, args = match("generate a password without symbols")
    assert (tool, args["include_symbols"]) == ("generate_password", False)


# ── arithmetic ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("text,expr", [
    ("2+2", "2+2"),
    ("what is 100000*(1.085**20)", "100000*(1.085**20)"),
    ("calculate 2^10", "2**10"),
    ("math.sqrt(144)*2", "math.sqrt(144)*2"),
])
def test_math(text, expr):
    assert match(text) == ("calculator", {"expression": expr})


@pytest.mark.parametrize("text", [
    "42",                                   # a number is not a sum
    "what is the meaning of life",          # no digits
    "I have 3 apples and 2 oranges, what should I cook?",
    "remind me to call mom at 5",
    "open('secrets.txt').read()",           # a name the calculator cannot take
])
def test_not_math(text):
    assert match(text) is None


# ── the default ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("text", [
    "",
    "   ",
    "hello",
    "what's the weather in Delhi",
    "search the web for the latest on the RBI repo rate",
    "email dana@example.com about tomorrow's meeting",
    "summarize the file I uploaded",
    "add a task to call the plumber on Friday",
])
def test_falls_through_to_the_model(text):
    assert match(text) is None
