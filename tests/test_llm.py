"""LLM wrapper: JSON parsing/repair, retries, pacing and budget."""
import time
from types import SimpleNamespace

import httpx
import openai
import pytest

from conftest import Cut, ScriptedClient
from research_agent.config import Settings
from research_agent.llm import (CUT_OFF, LLM, BudgetExceeded, Lane, LLMError, LLMOutputError, TokenBucket,
                                _lanes_from, parse_json_object)
from research_agent.models import AgentAction, Verdict
from research_agent.reporter import normalize_url, outside_window, requested_window_days

REVIEW_SYS = [{"role": "system", "content": "You are the REVIEWER"}, {"role": "user", "content": "x"}]


def make_llm(replies, **kw):
    events = []
    client = ScriptedClient({"review": replies})
    kw.setdefault("tpm_limit", 0)
    llm = LLM(Settings(api_key="t", backoff_base=0, llm_retries=2, **kw), emit=lambda k, **d: events.append(k),
              client=client, sleep=lambda s: sleeps.append(s))
    llm.sleeps = sleeps = []
    return llm, client, events


@pytest.mark.parametrize("raw", ['{"a": 1}', '```json\n{"a": 1}\n```', 'Sure! {"a": 1} hope it helps'])
def test_parse_json_object_is_lenient(raw):
    assert parse_json_object(raw) == {"a": 1}


@pytest.mark.parametrize("raw", ["", "no json", "[1, 2]", '{"a": '])
def test_parse_json_object_rejects_garbage(raw):
    with pytest.raises(ValueError):
        parse_json_object(raw)


def test_complete_json_repairs_invalid_reply():
    llm, client, events = make_llm(['{"reason": "no passed field"}', '{"passed": true, "reason": "ok"}'])
    v = llm.complete_json(REVIEW_SYS, Verdict)
    assert v.passed and events == ["json_repair"]
    repair_msg = client.calls[1][1][-1]["content"]
    assert "passed" in repair_msg and "invalid" in repair_msg


def test_complete_json_gives_up_after_max_repairs():
    llm, _, _ = make_llm(["bad", "bad", "bad"])
    with pytest.raises(LLMOutputError):
        llm.complete_json(REVIEW_SYS, Verdict, max_repairs=2)


def _status_error(cls, code, headers=None, message="err"):
    req = httpx.Request("POST", "https://api.test/v1/chat/completions")
    return cls(message, response=httpx.Response(code, headers=headers or {}, request=req), body=None)


def test_rate_limit_is_retried():
    llm, _, events = make_llm([_status_error(openai.RateLimitError, 429, {"retry-after": "1"}),
                               '{"passed": true}'])
    assert llm.complete_json(REVIEW_SYS, Verdict).passed
    assert events == ["llm_rate_limit"]


def test_long_rate_limit_gives_up_with_clear_error():
    llm, _, _ = make_llm([_status_error(openai.RateLimitError, 429, {"retry-after": "3600"})])
    with pytest.raises(LLMError, match="rate limited"):
        llm.chat(REVIEW_SYS)


def test_auth_error_is_not_retried():
    llm, client, _ = make_llm([_status_error(openai.AuthenticationError, 401)])
    with pytest.raises(LLMError, match="API key"):
        llm.chat(REVIEW_SYS)
    assert len(client.calls) == 1


def test_server_errors_exhaust_retries():
    llm, client, events = make_llm([_status_error(openai.InternalServerError, 500)] * 3)
    with pytest.raises(LLMError, match="failed after 3 attempts"):
        llm.chat(REVIEW_SYS)
    assert events == ["llm_retry", "llm_retry"]


def test_budget_blocks_non_critical_calls_only():
    llm, _, _ = make_llm(['{"passed": true}'] * 3, token_budget=100)
    llm.chat(REVIEW_SYS)  # uses 120 tokens
    with pytest.raises(BudgetExceeded):
        llm.chat(REVIEW_SYS)
    llm.chat(REVIEW_SYS, critical=True)


def test_normalize_url_for_grounding():
    assert normalize_url("https://www.Example.com/a/?utm_source=x#frag") == normalize_url("https://example.com/a")


def test_rate_limits_do_not_use_up_error_retries():
    # llm_retries=2 would allow only 3 attempts, but 429 waits have their own allowance
    llm, _, events = make_llm([_status_error(openai.RateLimitError, 429, {"retry-after": "1"})] * 5
                              + ['{"passed": true}'])
    assert llm.complete_json(REVIEW_SYS, Verdict).passed
    assert events == ["llm_rate_limit"] * 5


def test_endless_rate_limiting_gives_up():
    llm, _, _ = make_llm([_status_error(openai.RateLimitError, 429, {"retry-after": "1"})] * 20,
                         llm_rate_limit_waits=3)
    with pytest.raises(LLMError, match="still rate limited"):
        llm.chat(REVIEW_SYS)


def test_token_bucket_refills_continuously():
    b = TokenBucket(6000)                                 # refills 100 tokens/s
    b.charge(3000, now=100.0)
    assert (b.used(100.0), b.used(115.0), b.used(200.0)) == (3000, 1500, 0)
    assert b.wait_for(1000, now=100.0) == 0               # 3000 + 1000 fits under 95% of 6000
    assert b.wait_for(4200, now=100.0) == pytest.approx(15.0)  # must refill 1500 first


def test_pacing_waits_only_as_long_as_the_request_needs():
    llm, _, events = make_llm(['{"passed": true}'] * 2, tpm_limit=6000)
    llm.chat(REVIEW_SYS, max_tokens=300)
    llm.bucket().charge(5000, time.monotonic())           # pretend a big call just finished
    llm.chat(REVIEW_SYS, max_tokens=1000)                 # ~5120 used + ~1010 asked > 5700
    assert events == ["llm_throttle"] and 4.0 < llm.sleeps[0] < 4.5  # seconds, not a whole minute


def test_pacing_estimate_calibrates_to_the_providers_token_count():
    llm, _, _ = make_llm(['{"passed": true}'], tpm_limit=8000)
    msgs = [REVIEW_SYS[0], {"role": "user", "content": "word " * 80}]  # 400+ chars
    chars = sum(len(m["content"]) for m in msgs)
    assert llm.estimate_tokens(msgs) == int(chars / 3)  # conservative before any response
    llm.chat(msgs)                                       # scripted usage: 100 prompt tokens
    assert llm.estimate_tokens(msgs, max_tokens=300) == 100 + 300


def test_json_is_salvaged_from_reasoning_when_content_is_empty():
    from types import SimpleNamespace
    llm, client, events = make_llm([])
    msg = SimpleNamespace(content="", reasoning='Args look like {query}. So: {"passed": true, "reason": "ok"} done')
    client.chat.completions.create = lambda **kw: SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)
    assert llm.complete_json(REVIEW_SYS, Verdict).passed
    assert events == ["reasoning_salvage"]


def test_provider_json_mode_failure_switches_to_plain_mode_for_the_run():
    req = httpx.Request("POST", "https://api.test/v1/chat/completions")
    bad = openai.BadRequestError("Failed to validate JSON", response=httpx.Response(400, request=req),
                                 body={"error": {"code": "json_validate_failed", "failed_generation": ""}})
    llm, client, events = make_llm([bad, '{"passed": true}', '{"passed": false}'])
    assert llm.complete_json(REVIEW_SYS, Verdict).passed
    assert not llm.complete_json(REVIEW_SYS, Verdict).passed
    assert events == ["json_mode_off"]
    assert client.response_formats == [True, False, False]


@pytest.mark.parametrize("goal,days", [
    ("Top 3 developments in AI agents from the last week", 7), ("AI news this week", 7),
    ("What happened in the past 14 days?", 14), ("Market moves today", 1),
    ("Chip news from the last month", 31), ("Tell me the latest about Python", None),
])
def test_requested_window_is_read_from_the_goal(goal, days):
    assert requested_window_days(goal) == days


def test_outside_window_only_flags_parseable_old_dates():
    import datetime as dt
    today = dt.date(2026, 9, 23)
    assert outside_window("2026-09-11", 7, today)
    assert not outside_window("2026-09-16", 7, today)          # inside, incl. one day of slack
    assert not outside_window("Sept 11, 2026", 7, today)       # unparseable -> not flagged
    assert not outside_window("2026-09-11", None, today)       # goal set no window


@pytest.mark.parametrize("raw, action, tool", [
    ({"thought": "t", "tool": "web_search", "args": {"query": "x"}}, "call_tool", "web_search"),
    ({"thought": "t", "output": "done", "sources": []}, "finish", None),
    ({"thought": "t", "action": "web_search", "args": {"query": "x"}}, "call_tool", "web_search"),
    ({"thought": "t", "action": "call_tool", "tool": "fetch_page", "args": {}}, "call_tool", "fetch_page"),
])
def test_missing_or_misplaced_action_is_inferred_without_a_repair(raw, action, tool):
    a = AgentAction.model_validate(raw)
    assert (a.action, a.tool) == (action, tool)


def test_action_that_cannot_be_inferred_still_fails_validation():
    with pytest.raises(ValueError):
        AgentAction.model_validate({"thought": "hmm"})


def test_pacing_charges_what_was_used_not_the_max_tokens_reservation():
    llm, _, _ = make_llm(['{"passed": true}'], tpm_limit=8000)
    llm.chat(REVIEW_SYS, max_tokens=900)          # scripted usage: 100 prompt, 20 completion
    # unused max_tokens is released (abs=5: Windows' ~16 ms clock tick refills ~2 tokens meanwhile)
    assert llm.bucket().used(time.monotonic()) == pytest.approx(120, abs=5)


def test_pacing_follows_the_providers_rate_limit_headers():
    b = TokenBucket(8000)
    b.sync({"x-ratelimit-limit-tokens": "250000", "x-ratelimit-remaining-tokens": "249000"},
           reserved=900, completion=100, now=50.0)
    assert b.limit == 250_000 and b.used(50.0) == 200     # 1000, less the 800 of max_tokens still held
    b.sync({"x-ratelimit-remaining-tokens": "240000"}, reserved=100, completion=100, now=50.0)
    assert b.used(50.0) == 10_000                         # someone else on the key: adopt their usage
    b.sync({"x-ratelimit-remaining-tokens": "249990"}, reserved=100, completion=100, now=50.0)
    assert b.used(50.0) == 10_000                         # the provider knows less than we do: keep ours


def test_llm_reads_rate_limit_headers_from_the_raw_response():
    client = ScriptedClient({"review": ['{"passed": true}']})
    headers = {"x-ratelimit-limit-tokens": "30000", "x-ratelimit-remaining-tokens": "29880"}
    client.chat.completions.with_raw_response = SimpleNamespace(
        create=lambda **kw: SimpleNamespace(parse=lambda: client._create(**kw), headers=headers))
    llm = LLM(Settings(api_key="t", tpm_limit=8000), client=client, sleep=lambda s: None)
    llm.chat(REVIEW_SYS)
    assert llm.bucket().limit == 30_000


def test_calls_can_go_to_a_second_model_with_its_own_limit():
    llm, client, _ = make_llm(['{"passed": true}'] * 2, tpm_limit=8000)
    llm.chat(REVIEW_SYS, model="small-model")
    llm.chat(REVIEW_SYS)
    assert client.models == ["small-model", Settings.model]
    assert set(llm.buckets) == {"small-model", Settings.model}


def test_unavailable_second_model_falls_back_to_the_main_one():
    llm, client, events = make_llm([_status_error(openai.NotFoundError, 404), '{"passed": true}',
                                    '{"passed": true}'])
    llm.chat(REVIEW_SYS, model="retired-model")
    llm.chat(REVIEW_SYS, model="retired-model")          # remembered: no second 404
    assert events == ["model_fallback"]
    assert client.models == ["retired-model", Settings.model, Settings.model]


def test_main_model_not_found_is_fatal():
    llm, _, _ = make_llm([_status_error(openai.NotFoundError, 404)])
    with pytest.raises(LLMError, match="not found"):
        llm.chat(REVIEW_SYS)


def test_reply_cut_off_at_max_tokens_is_asked_again_with_more_room():
    llm, client, events = make_llm([Cut('{"passed": true, "reason": "the list goes on and'), '{"passed": true}'])
    assert llm.chat(REVIEW_SYS, max_tokens=300) == '{"passed": true}'
    assert events == ["llm_truncated"] and client.max_tokens == [300, 600]


def test_cut_off_json_is_repaired_by_asking_for_a_shorter_reply_without_the_fragment():
    llm, client, events = make_llm(['{"passed": true, "reason": "went on and on', '{"passed": true}'])
    assert llm.complete_json(REVIEW_SYS, Verdict).passed
    assert events == ["json_repair"]
    retry = client.calls[-1][1]
    assert "SHORTER" in retry[-1]["content"] and all(m["role"] != "assistant" for m in retry)
    with pytest.raises(ValueError, match=CUT_OFF):
        parse_json_object('{"thought": "x", "output": "1. a {b} 2.')


def test_json_repair_event_keeps_a_preview_of_the_raw_reply():
    events = []
    llm = LLM(Settings(api_key="t", backoff_base=0, tpm_limit=0), emit=lambda k, **d: events.append((k, d)),
              client=ScriptedClient({"review": ['{"reason": "no passed field"}' + " " * 500,
                                                '{"passed": true}']}), sleep=lambda s: None)
    llm.complete_json(REVIEW_SYS, Verdict)
    kind, data = events[0]
    assert kind == "json_repair" and data["raw"].startswith('{"reason": "no passed field"}')
    assert len(data["raw"]) == 300


# An optional second provider ("lane") for the main model, e.g. Cerebras' free tier next to Groq's.

def make_lane_llm(primary_replies, lane_replies, rpm=5, **kw):
    events = []
    lane = Lane("cerebras", ScriptedClient({"review": lane_replies}), "gpt-oss-120b", TokenBucket(30_000), rpm=rpm)
    kw.setdefault("tpm_limit", 8000)
    llm = LLM(Settings(api_key="t", backoff_base=0, llm_retries=2, **kw), emit=lambda k, **d: events.append(k),
              client=ScriptedClient({"review": primary_replies}), sleep=lambda s: None, lanes=[lane])
    return llm, lane, events


OK = '{"passed": true}'


def test_main_model_calls_go_to_whichever_provider_can_take_them_first():
    llm, lane, _ = make_lane_llm([OK] * 2, [OK] * 2, rpm=2, tpm_limit=0)
    for _ in range(3):
        llm.chat(REVIEW_SYS)                      # the third finds the lane's 2-a-minute window full
    llm.chat(REVIEW_SYS, model="small-model")     # the reviewer's model is only on the primary
    assert lane.client.models == ["gpt-oss-120b"] * 2
    assert llm.client.models == [Settings.model, "small-model"]
    assert llm.usage["by_provider"] == {"cerebras": 2, "groq": 2} and llm.usage["calls"] == 4


def test_the_lane_takes_calls_while_the_primary_bucket_refills_and_not_while_it_rests():
    llm, lane, _ = make_lane_llm([OK], [OK])
    llm.bucket().charge(8000, time.monotonic())   # the primary needs ~12 s before it can take a call
    llm.chat(REVIEW_SYS)
    lane.rest_until = time.monotonic() + 30       # now the lane is the slower one
    llm.chat(REVIEW_SYS)
    assert lane.client.models == ["gpt-oss-120b"] and llm.client.models == [Settings.model]


def test_a_failing_lane_hands_the_call_to_the_primary_and_is_switched_off_after_repeated_failures():
    llm, lane, events = make_lane_llm([OK] * 3, [_status_error(openai.InternalServerError, 500)] * 2)
    for _ in range(3):
        assert llm.chat(REVIEW_SYS) == OK
    assert events == ["lane_fallback", "lane_disabled"]  # no error retries were spent on the lane
    assert len(lane.client.calls) == 2 and len(llm.client.calls) == 3 and lane.disabled


def test_a_rejected_lane_key_switches_the_lane_off_at_once():
    llm, lane, events = make_lane_llm([OK] * 2, [_status_error(openai.AuthenticationError, 401)])
    llm.chat(REVIEW_SYS)
    llm.chat(REVIEW_SYS)
    assert events == ["lane_disabled"] and len(lane.client.calls) == 1 and len(llm.client.calls) == 2


def test_a_rate_limited_lane_rests_while_the_primary_serves():
    llm, lane, events = make_lane_llm([OK] * 2, [_status_error(openai.RateLimitError, 429, {"retry-after": "30"}), OK])
    llm.chat(REVIEW_SYS)
    llm.chat(REVIEW_SYS)                          # the lane is resting: straight to the primary
    lane.rest_until = 0.0
    llm.chat(REVIEW_SYS)
    assert events == ["lane_fallback"] and len(lane.client.calls) == 2 and len(llm.client.calls) == 2


def test_a_lane_stops_sending_what_it_rejected():
    no_json = _status_error(openai.BadRequestError, 400, message="response_format json_object is not supported")
    llm, lane, events = make_lane_llm([OK], [no_json, OK])
    llm.chat(REVIEW_SYS)
    llm.chat(REVIEW_SYS)
    assert lane.client.response_formats == [True, False] and not lane.disabled
    assert llm.client.response_formats == [True]  # the primary keeps its JSON mode


def test_a_lane_with_a_smaller_context_window_only_gets_requests_that_fit():
    too_long = _status_error(openai.BadRequestError, 400, message="Please reduce the length of the messages: "
                                                                   "maximum context length is 8192")
    llm, lane, _ = make_lane_llm([OK] * 2, [too_long, OK])
    llm.chat(REVIEW_SYS, max_tokens=4000)         # rejected: the primary serves it
    llm.chat(REVIEW_SYS, max_tokens=4000)         # as big: not tried on the lane again
    llm.chat(REVIEW_SYS, max_tokens=500)          # smaller: the lane takes it
    assert lane.client.max_tokens == [4000, 500] and llm.client.max_tokens == [4000, 4000]


def test_per_minute_rate_limit_header_names_are_read_too():
    b = TokenBucket(30_000)                       # Cerebras: x-ratelimit-*-tokens-minute
    b.sync({"x-ratelimit-limit-tokens-minute": "60000", "x-ratelimit-remaining-tokens-minute": "50000"},
           reserved=100, completion=100, now=50.0)
    assert b.limit == 60_000 and b.used(50.0) == 10_000


def test_the_lane_exists_only_with_a_key_the_same_model_and_a_real_client():
    assert _lanes_from(Settings(api_key="t")) == []
    lanes = _lanes_from(Settings(api_key="t", cerebras_api_key="csk-test"))
    assert [(lane.name, lane.model, lane.rpm) for lane in lanes] == [("cerebras", "gpt-oss-120b", 5)]
    assert _lanes_from(Settings(api_key="t", cerebras_api_key="csk-test", model="qwen/qwen3-32b")) == []
    # a scripted or demo client must never be mixed with a real provider
    assert LLM(Settings(api_key="t", cerebras_api_key="csk-test"), client=ScriptedClient({})).lanes == []
