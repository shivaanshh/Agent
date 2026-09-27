"""End-to-end control-flow tests with a scripted LLM and fake tools."""
import pytest

from conftest import (FAIL, NEWS_URL, PASS, WEB_URL, call, finish, plan_json, report_json)
from research_agent.executor import urls_in
from research_agent.verifier import ungrounded

LONG = "Found: Acme released an open-source agent framework on 2026-09-20, see example.com. " * 2


def kinds(tracer):
    return [r["kind"] for r in tracer.recoveries]


def test_happy_path_produces_grounded_report(make_agent):
    agent, client = make_agent({
        "plan": [plan_json(("Search news", "news_search"), ("Rank findings", "none"))],
        "exec": [call("news_search", query="AI agents"), finish(LONG, [NEWS_URL]),
                 finish("Ranked: 1) Acme agent framework launch, the most significant item this week.")],
        "review": [PASS, PASS],
        "report": [report_json([NEWS_URL, "https://made-up.example/fake"])],
    })
    run = agent.run("Top AI agent developments this week")

    assert [r.status for r in run.results] == ["passed", "passed"]
    report = run.report
    assert report["stats"]["steps_passed"] == 2
    srcs = report["findings"][0]["sources"]
    assert srcs == [{"url": NEWS_URL, "verified": True}, {"url": "https://made-up.example/fake", "verified": False}]
    assert report["grounding"]["rate"] == 0.5
    assert report["grounding"]["findings_outside_window"] == 0
    assert agent.tracer.recoveries == []


def test_invented_urls_cap_confidence_and_are_not_offered_to_the_reporter(make_agent):
    fake = "https://techcrunch.com/2099/01/01/made-up-story/"
    draft = report_json([fake, "https://venturebeat.com/also-made-up/"])
    draft["confidence"] = "high"
    agent, client = make_agent({
        "plan": [plan_json(("Search news", "news_search"), ("Rank findings", "none"))],
        "exec": [call("news_search", query="AI agents"), finish(LONG, [NEWS_URL, fake]),
                 finish("Ranked: 1) Acme agent framework launch, the most significant item this week.")],
        "review": [PASS],
        "report": [draft],
    }, max_step_retries=0, max_replans=0)  # so the step that cited an invented URL reaches the reporter as it is
    run = agent.run("Top AI agent developments this week")
    report = run.report

    assert run.results[0].status == "failed" and "may be invented" in run.results[0].verdict.reason
    assert report["grounding"]["verified_sources"] == 0
    assert report["confidence"] == "low"
    assert "may be invented" in report["limitations"][0] and fake in report["limitations"][0]
    reporter_prompt = [m for role, m in client.calls if role == "report"][0][1]["content"]
    assert NEWS_URL in reporter_prompt and fake not in reporter_prompt


def test_finding_older_than_the_requested_window_is_flagged(make_agent):
    import datetime as dt
    stale = report_json([NEWS_URL])
    stale["findings"][0]["date"] = (dt.date.today() - dt.timedelta(days=12)).isoformat()
    stale["confidence"] = "high"
    agent, _ = make_agent({
        "plan": [plan_json(("Search news", "news_search"), ("Rank findings", "none"))],
        "exec": [call("news_search", query="AI agents"), finish(LONG, [NEWS_URL]),
                 finish("Ranked: 1) Acme agent framework launch, the most significant item this week.")],
        "review": [PASS, PASS],
        "report": [stale],
    })
    report = agent.run("Top AI agent developments from the last week").report

    assert report["findings"][0]["outside_requested_window"] is True
    assert report["grounding"]["window_days"] == 7 and report["grounding"]["findings_outside_window"] == 1
    assert report["confidence"] == "medium"          # capped: the LLM said "high"
    assert "7-day window" in report["limitations"][0]


def test_tool_outage_recovers_by_switching_tool(make_agent):
    """news_search always fails (injected 503): registry retries, then the agent switches to web_search."""
    agent, client = make_agent({
        "plan": [plan_json(("Search news", "news_search"), ("Rank findings", "none"))],
        "exec": [call("news_search", query="AI agents"),
                 call("web_search", query="AI agents news"),
                 finish(LONG, [WEB_URL]),
                 finish("Ranked list of the findings from the web results above, with sources.")],
        "review": [PASS, PASS],
        "report": [report_json([WEB_URL])],
    }, faults=["news_search:error:*"])
    run = agent.run("Top AI agent developments this week")

    step1 = run.results[0]
    assert step1.status == "passed"
    assert [c.ok for c in step1.tool_calls] == [False, True]
    assert step1.tool_calls[0].attempts == 3  # 1 try + 2 retries
    assert kinds(agent.tracer).count("tool_retry") == 2
    # the failure was reported back to the LLM as an observation
    exec_msgs = [m for role, m in client.calls if role == "exec"][1]
    assert "FAILED after 3 attempt(s)" in exec_msgs[-1]["content"]
    assert run.report["execution"]["injected_faults"] == ["news_search:error:*"]


def test_failed_step_is_retried_then_replanned(make_agent):
    agent, client = make_agent({
        "plan": [plan_json(("Search news", "news_search"), ("Rank findings", "none"))],
        "exec": [
            finish("I think there were some agent launches."),              # attempt 1: no tool -> fails
            finish("Probably a few agent launches happened this week."),    # attempt 2: still no tool
            call("web_search", query="AI agent launch"), finish(LONG, [WEB_URL]),  # replanned step
        ],
        "review": [PASS],
        "replan": [{"reasoning": "news search step produced nothing grounded; use web search",
                    "steps": [{"id": 3, "title": "Web search instead", "instruction": "Use web_search for launches",
                               "tool_hint": "web_search", "success_criteria": "at least one URL"}]}],
        "report": [report_json([WEB_URL])],
    })
    run = agent.run("Top AI agent developments this week")

    assert [(r.step_id, r.status) for r in run.results] == [(1, "failed"), (3, "passed")]
    assert run.results[0].attempts == 2
    k = kinds(agent.tracer)
    assert "step_retry" in k and "replan" in k
    assert run.report["execution"]["replans"][0]["after_step"] == 1
    # the original step 2 was replaced by the new plan, and the step counter follows the new ids
    assert "Rank findings" not in [r.title for r in run.results]
    starts = [(e["step_id"], e["total"]) for e in agent.tracer.events if e["event"] == "step_start"]
    assert starts == [(1, 2), (1, 2), (3, 3)]
    # reviewer feedback was passed into the retry
    retry_prompt = [m for role, m in client.calls if role == "exec"][1][1]["content"]
    assert "REVIEWER FEEDBACK" in retry_prompt
    assert "call tools again" in retry_prompt  # the step needs a tool, and the old results are gone


def test_llm_review_failure_triggers_retry_with_feedback(make_agent):
    agent, client = make_agent({
        "plan": [plan_json(("Search news", "news_search"), ("Rank", "none"))],
        "exec": [call("news_search", query="x"), finish(LONG, [NEWS_URL]),
                 call("news_search", query="AI agent framework release"), finish(LONG + " more", [NEWS_URL]),
                 finish("Ranked findings with the Acme framework launch first, citing example.com.")],
        "review": [FAIL, PASS, PASS],
        "report": [report_json([NEWS_URL])],
    })
    run = agent.run("goal")
    assert run.results[0].status == "passed" and run.results[0].attempts == 2
    assert len(run.results[0].tool_calls) == 2  # calls from both attempts are kept


def test_invalid_json_is_repaired(make_agent):
    plan = plan_json(("Search news", "news_search"), ("Rank", "none"))
    agent, client = make_agent({
        "plan": [plan, plan],
        "exec": [call("news_search", query="x"), finish(LONG, [NEWS_URL]),
                 finish("Ranked findings with the Acme framework launch first, citing example.com.")],
        "review": [PASS, PASS],
        "report": [report_json([NEWS_URL])],
    }, faults=["llm:bad_json:1"])  # the planner's first reply gets truncated
    run = agent.run("goal")
    assert "json_repair" in kinds(agent.tracer)
    assert client.count("plan") == 2
    assert len(run.plan.steps) == 2


def test_unusable_plan_falls_back_to_default_plan(make_agent):
    bad = plan_json(("Search", "google_it"), ("Rank", "none"))  # unknown tool, every time
    agent, _ = make_agent({
        "plan": [bad, bad, bad],
        "exec": [finish("x" * 60)] * 20,
        "review": [PASS] * 20,
        "replan": [{"reasoning": "r", "steps": [{"id": 9, "title": "noop", "instruction": "nothing to do here",
                                                  "tool_hint": "none", "success_criteria": "none at all"}]}],
        "report": [report_json([])],
    })
    run = agent.run("goal")
    assert "fallback_plan" in kinds(agent.tracer)
    assert run.plan.steps[0].tool_hint == "news_search"


def test_unknown_tool_and_duplicate_calls_are_handled(make_agent):
    agent, _ = make_agent({
        "plan": [plan_json(("Search news", "news_search"), ("Rank", "none"))],
        "exec": [call("google", query="x"), call("news_search", query="x"), call("news_search", query="x"),
                 finish(LONG, [NEWS_URL]),
                 finish("Ranked findings with the Acme framework launch first, citing example.com.")],
        "review": [PASS, PASS],
        "report": [report_json([NEWS_URL])],
    })
    run = agent.run("goal")
    k = kinds(agent.tracer)
    assert "invalid_action" in k and "duplicate_call" in k
    assert run.results[0].status == "passed"


def test_action_limit_forces_finish(make_agent):
    agent, client = make_agent({
        "plan": [plan_json(("Search news", "news_search"), ("Rank", "none"))],
        "exec": [call("news_search", query="a"), call("news_search", query="b"), call("news_search", query="c"),
                 finish(LONG, [NEWS_URL]),
                 finish("Ranked findings with the Acme framework launch first, citing example.com.")],
        "review": [PASS, PASS],
        "report": [report_json([NEWS_URL])],
    }, max_actions_per_step=3)
    agent.run("goal")
    assert "forced_finish" in kinds(agent.tracer)


def test_report_writer_failure_uses_fallback_report(make_agent):
    agent, _ = make_agent({
        "plan": [plan_json(("Search news", "news_search"), ("Rank", "none"))],
        "exec": [call("news_search", query="x"), finish(LONG, [NEWS_URL]),
                 finish("Ranked findings with the Acme framework launch first, citing example.com.")],
        "review": [PASS, PASS],
        "report": ["not json", "still not json", "nope"],
    })
    run = agent.run("goal")
    assert "fallback_report" in kinds(agent.tracer)
    assert run.report["confidence"] == "low"
    assert run.report["findings"][0]["sources"][0] == {"url": NEWS_URL, "verified": True}


def test_budget_exhaustion_skips_remaining_steps_but_still_reports(make_agent):
    agent, _ = make_agent({
        "plan": [plan_json(("Search news", "news_search"), ("Search web", "web_search"), ("Rank", "none"))],
        "exec": [call("news_search", query="x"), finish(LONG, [NEWS_URL])],
        "review": [PASS],
        "report": [report_json([NEWS_URL])],
    }, token_budget=450)  # each fake call costs 120 tokens
    run = agent.run("goal")
    statuses = [r.status for r in run.results]
    assert statuses[0] == "passed" and statuses[-1] == "skipped"
    assert "budget" in kinds(agent.tracer)
    assert run.report["title"] == "AI agents this week"  # report call is 'critical' and still ran
    assert run.report["confidence"] == "low"  # capped: the draft said "medium" but only 1 of 3 steps passed
    assert run.report["limitations"][0].startswith("Only 1 of 3 steps passed review")


def test_ambiguous_goal_asks_for_clarification(make_agent):
    ambiguous = plan_json(("Search news", "news_search"), ("Rank", "none"),
                          ambiguities=["'Python' could mean the language or the snake"],
                          assumptions=["Assume the programming language"])
    agent, client = make_agent({
        "plan": [ambiguous, plan_json(("Search news", "news_search"), ("Rank", "none"))],
        "exec": [call("news_search", query="x"), finish(LONG, [NEWS_URL]),
                 finish("Ranked findings with the Acme framework launch first, citing example.com.")],
        "review": [PASS, PASS],
        "report": [report_json([NEWS_URL])],
    })
    asked = []
    run = agent.run("latest about Python", clarify=lambda amb: asked.append(amb) or "the programming language")
    assert asked and "snake" in asked[0][0]
    assert "User clarification: the programming language" in run.goal
    assert client.count("plan") == 2


def test_llm_outage_mid_run_still_produces_a_report(make_agent):
    import httpx
    import openai
    down = openai.InternalServerError("down", response=httpx.Response(
        503, request=httpx.Request("POST", "https://api.test")), body=None)
    agent, _ = make_agent({
        "plan": [plan_json(("Search news", "news_search"), ("Search web", "web_search"), ("Rank", "none"))],
        "exec": [call("news_search", query="x"), finish(LONG, [NEWS_URL]), down, down, down],
        "review": [PASS],
        "report": [report_json([NEWS_URL])],
    })
    run = agent.run("goal")
    assert [r.status for r in run.results] == ["passed", "skipped", "skipped"]
    assert "llm_unavailable" in kinds(agent.tracer)
    assert run.report["title"] == "AI agents this week"


def test_tool_calls_of_an_interrupted_step_still_count(make_agent):
    import httpx
    import openai
    down = openai.InternalServerError("down", response=httpx.Response(
        503, request=httpx.Request("POST", "https://api.test")), body=None)
    agent, _ = make_agent({
        "plan": [plan_json(("Search news", "news_search"), ("Search web", "web_search"), ("Rank", "none"))],
        "exec": [call("news_search", query="x"), finish(LONG, [NEWS_URL]),
                 call("web_search", query="y"), down, down, down],
        "review": [PASS],
        "report": [report_json([NEWS_URL])],
    })
    run = agent.run("goal")
    assert [r.status for r in run.results] == ["passed", "skipped", "skipped"]
    assert [c.tool for c in run.results[1].tool_calls] == ["web_search"]
    assert run.report["stats"]["tool_calls"] == 2


RANKED ="Ranked: 1) Acme agent framework launch, the most significant item this week."


def test_reviewer_sees_the_sources_the_step_cited(make_agent):
    agent, client = make_agent({
        "plan": [plan_json(("Search news", "news_search"), ("Rank findings", "none"))],
        "exec": [call("news_search", query="AI agents"), finish(LONG, [NEWS_URL]), finish(RANKED)],
        "review": [PASS, PASS],
        "report": [report_json([NEWS_URL])],
    })
    agent.run("Top AI agent developments this week")
    review_prompt = [m for role, m in client.calls if role == "review"][0][-1]["content"]
    assert f"SOURCES CITED FOR THIS RESULT:\n- {NEWS_URL}" in review_prompt
    assert "-> ok: 1 news result for 'AI agents':" in review_prompt  # so it can see how many items the tool found


def test_a_step_citing_urls_no_tool_returned_fails_before_the_reviewer_is_asked(make_agent):
    invented = "https://made-up.example/agents-2"
    agent, client = make_agent({
        "plan": [plan_json(("Search news", "news_search"), ("Rank findings", "none"))],
        "exec": [call("news_search", query="AI agents"), finish(LONG + f" Also {invented}.", [NEWS_URL]),
                 call("news_search", query="AI agent launches"), finish(LONG, [NEWS_URL]),
                 finish(RANKED)],
        "review": [PASS, PASS],
        "report": [report_json([NEWS_URL])],
    })
    run = agent.run("Top AI agent developments this week")
    assert (run.results[0].status, run.results[0].attempts) == ("passed", 2)
    assert client.count("review") == 2  # attempt 1 was failed by code, not by the reviewer
    retry_prompt = [m for role, m in client.calls if role == "exec"][2][1]["content"]
    assert f"1 of 2 cited URLs were not returned by any tool in this run, so those items may be invented: " \
           f"{invented}" in retry_prompt


def test_cited_urls_match_returned_ones_despite_copying_differences():
    returned = ["https://www.reuters.com/tech/agents-2026-09-20/?utm_source=x",
                "https://news.google.com/rss/articles/CBMi1?oc=5"]
    cited = ["http://reuters.com/tech/agents-2026-09-20", "https://news.google.com/rss/articles/CBMi1",
             "https://reuters.com", "https://www.reuters.com/tech/agents-2026-09-21", "https://made-up.example/x"]
    assert ungrounded(cited, returned) == ["https://www.reuters.com/tech/agents-2026-09-21", "https://made-up.example/x"]


def test_trace_records_step_output_and_streams_events_to_listeners(make_agent):
    agent, _ = make_agent({
        "plan": [plan_json(("Search news", "news_search"), ("Rank findings", "none"))],
        "exec": [call("news_search", query="AI agents"), finish(LONG, [NEWS_URL]), finish(RANKED)],
        "review": [PASS, PASS],
        "report": [report_json([NEWS_URL])],
    })
    streamed = []
    agent.tracer.listeners.append(streamed.append)
    agent.run("Top AI agent developments this week")

    outputs = [e for e in agent.tracer.events if e["event"] == "step_output"]
    assert [(e["step_id"], e["output"], e["sources"]) for e in outputs] == [(1, LONG.strip(), [NEWS_URL]),
                                                                             (2, RANKED, [])]
    assert streamed == agent.tracer.events
    assert "Top AI agent developments" in agent.tracer.console.export_text()  # quiet still records


def test_plan_keeps_at_most_one_analysis_only_step(make_agent):
    agent, _ = make_agent({
        "plan": [plan_json(("Search news", "news_search"), ("Extract details", "none"),
                           ("Summarise", "none"), ("Rank findings", "none"))],
        "exec": [call("news_search", query="AI agents"), finish(LONG, [NEWS_URL]), finish(RANKED)],
        "review": [PASS, PASS],
        "report": [report_json([NEWS_URL])],
    })
    run = agent.run("Top AI agent developments this week")
    assert [s.title for s in run.plan.steps] == ["Search news", "Rank findings"]
    notes = [e["message"] for e in agent.tracer.events if e["event"] == "info"]
    assert any("Extract details; Summarise" in n for n in notes)


def test_replan_keeps_only_evidence_gathering_steps_and_at_most_three(make_agent):
    def step(i, title, tool):
        return {"id": i, "title": title, "instruction": f"Do {title} now", "tool_hint": tool,
                "success_criteria": "at least one URL"}

    agent, _ = make_agent({
        "plan": [plan_json(("Search news", "news_search"), ("Rank findings", "none"))],
        "exec": [finish("I think there were some agent launches."), finish("Probably a few launches happened."),
                 *[x for i in range(3) for x in (call("web_search", query=f"q{i}"), finish(LONG, [WEB_URL]))]],
        "review": [PASS] * 3,
        "replan": [{"reasoning": "use web search", "steps": [
            step(3, "Web search A", "web_search"), step(4, "Validate", "none"), step(5, "Web search B", "web_search"),
            step(6, "Web search C", "web_search"), step(7, "Web search D", "web_search"),
            step(8, "Compile the final report", "none")]}],
        "report": [report_json([WEB_URL])],
    })
    run = agent.run("Top AI agent developments this week")
    assert [r.title for r in run.results][1:] == ["Web search A", "Web search B", "Web search C"]
    assert [r.step_id for r in run.results][1:] == [3, 4, 5]


def test_urls_written_in_a_steps_output_count_as_its_sources(make_agent):
    agent, _ = make_agent({
        "plan": [plan_json(("Search news", "news_search"), ("Rank findings", "none"))],
        "exec": [call("news_search", query="AI agents"),
                 finish(f"1. Acme launched an agent framework on 2026-09-20 ({NEWS_URL})."), finish(RANKED)],
        "review": [PASS, PASS],
        "report": [report_json([NEWS_URL])],
    })
    run = agent.run("Top AI agent developments this week")
    assert run.results[0].sources == [NEWS_URL]


@pytest.mark.parametrize("text,urls", [
    ("1. Acme (https://a.test/x). 2. [Beta](https://b.test/y), see **https://c.test/z**",
     ["https://a.test/x", "https://b.test/y", "https://c.test/z"]),
    ("Python: https://en.wikipedia.org/wiki/Python_(language), done", ["https://en.wikipedia.org/wiki/Python_(language)"]),
])
def test_urls_in_trims_markdown_and_punctuation(text, urls):
    assert urls_in(text) == urls


def test_reviewer_runs_on_the_review_model(make_agent):
    agent, client = make_agent({
        "plan": [plan_json(("Search news", "news_search"), ("Rank findings", "none"))],
        "exec": [call("news_search", query="AI agents"), finish(LONG, [NEWS_URL]), finish(RANKED)],
        "review": [PASS, PASS],
        "report": [report_json([NEWS_URL])],
    }, review_model="small-reviewer")
    agent.run("Top AI agent developments this week")
    roles = {role: model for (role, _), model in zip(client.calls, client.models)}
    assert roles["review"] == "small-reviewer" and roles["exec"] == roles["plan"] == agent.settings.model
