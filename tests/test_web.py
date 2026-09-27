"""The web UI's server and the demo stand-in LLM, offline: fake tools, real HTTP on a free local port."""
import json
import threading
import urllib.error
import urllib.request
from collections import Counter

import pytest

from conftest import fake_tools
from research_agent.agent import ResearchAgent
from research_agent.cli import main as cli_main
from research_agent.config import Settings
from research_agent.demo import DemoClient, topic
from research_agent.failures import FailureInjector
from research_agent.llm import LLM
from research_agent.tools import ToolRegistry
from research_agent.trace import Tracer
from research_agent.web import Runner, make_server

GOAL = "Top 3 developments in AI agents from the last week"


def demo_agent(settings, injector, tracer, demo=True):
    llm = LLM(settings, injector, emit=tracer.emit, client=DemoClient(pause=0), sleep=lambda _: None)
    registry = ToolRegistry(injector, timeout=5, retries=1, backoff_base=0, emit=tracer.emit, sleep=lambda _: None)
    for t in fake_tools():
        registry.register(t)
    return ResearchAgent(settings, llm, registry, tracer)


def run_demo(faults=()):
    tracer = Tracer(quiet=True)
    settings = Settings(api_key="demo", tpm_limit=0, backoff_base=0)
    run = demo_agent(settings, FailureInjector.from_strings(list(faults)), tracer).run(GOAL)
    return run, Counter(r["kind"] for r in tracer.recoveries)


# ── the demo stand-in drives the real control loop ────────────────────────────
def test_demo_client_completes_a_grounded_run():
    run, recoveries = run_demo()
    assert [r.status for r in run.results] == ["passed", "passed", "passed"]  # fetch_page isn't registered here
    assert run.report["grounding"]["rate"] == 1.0 and not recoveries


def test_demo_client_switches_tool_when_one_is_down():
    run, recoveries = run_demo(["news_search:error:*"])
    assert run.results[0].status == "passed"
    assert [c.tool for c in run.results[0].tool_calls] == ["news_search", "web_search"]


def test_demo_client_answers_again_after_a_json_repair():
    run, recoveries = run_demo(["llm:bad_json:2"])
    assert recoveries["json_repair"] == 2 and all(r.status == "passed" for r in run.results)


def test_demo_client_replans_when_every_search_fails():
    run, recoveries = run_demo(["news_search:error:*", "web_search:error:*"])
    assert recoveries["step_retry"] >= 1 and recoveries["replan"] == 1
    assert run.results[0].status == "failed" and run.report["findings"][0]["title"] == "No usable results"


def test_demo_ranking_drops_repeated_urls_and_keeps_undated_items_in_order():
    ranked = DemoClient._rank("Results of earlier steps:\n- A | undated | https://a.org\n- B | undated | https://b.org\n"
                              "- A again | undated | https://a.org\n- C | AP, 2026-09-20 | https://c.org")
    assert ranked["sources"] == ["https://c.org", "https://a.org", "https://b.org"]
    assert "3 distinct items" in ranked["thought"]


def test_cli_demo_mode_runs_without_a_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("research_agent.cli.load_dotenv", lambda: None)  # a key in .env must not matter
    monkeypatch.setattr("research_agent.cli.DemoClient", lambda: DemoClient(pause=0))
    monkeypatch.setenv("TOOL_RETRIES", "0")
    faults = ["news_search:error:*", "web_search:error:*", "fetch_page:error:*"]  # nothing reaches the network
    code = cli_main([GOAL, "--demo", "--quiet", "--out", str(tmp_path), *(a for f in faults for a in ("--inject", f))])
    (run_dir,) = tmp_path.iterdir()
    assert code == 0 and "_demo_" in run_dir.name
    assert json.loads((run_dir / "report.json").read_text(encoding="utf-8"))["findings"][0]["title"] == "No usable results"


@pytest.mark.parametrize("args", [["   "], [GOAL, "--inject", "news_serch:error"], [GOAL, "--inject", "llm:bad_json:0"]])
def test_cli_stops_before_running_on_bad_input(tmp_path, monkeypatch, args):
    monkeypatch.setattr("research_agent.cli.load_dotenv", lambda: None)
    assert cli_main([*args, "--demo", "--quiet", "--out", str(tmp_path)]) == 2
    assert not list(tmp_path.iterdir())


def test_topic_keeps_the_keywords():
    assert topic("Research and summarize the top 3 developments in AI agents from the last week") == "AI agents"


# ── HTTP server ───────────────────────────────────────────────────────────────
@pytest.fixture
def server(tmp_path):
    def start(factory=None):
        runner = Runner(tmp_path, factory=factory or demo_agent)
        srv = make_server("127.0.0.1", 0, runner)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_address[1]}", runner

    servers = []
    yield start
    for s in servers:
        s.shutdown()
        s.server_close()


def post(base, body, headers=None):
    req = urllib.request.Request(base + "/api/runs", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def stream(base, run_id):
    with urllib.request.urlopen(f"{base}/api/runs/{run_id}/events", timeout=30) as r:
        body = r.read().decode("utf-8")  # the server closes the stream after the "done" event
    return [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]


def test_run_streams_every_stage_and_saves_the_report(server):
    base, runner = server()
    with urllib.request.urlopen(base + "/", timeout=5) as r:
        assert b"Research Agent" in r.read()
    status, run = post(base, {"goal": GOAL, "demo": True, "inject": ["news_search:error:1"]})
    assert status == 202 and run["faults"] == ["news_search:error:1"]

    events = stream(base, run["id"])
    kinds = [e["event"] for e in events]
    for kind in ("goal", "plan", "step_start", "thought", "tool_call", "tool_result", "step_output", "verdict",
                 "step_done", "recovery", "section", "report"):
        assert kind in kinds, kind
    assert kinds[-1] == "done"
    assert events[0]["injected_faults"] == ["news_search:error:1"]

    with urllib.request.urlopen(f"{base}/api/runs/{run['id']}/report.md", timeout=5) as r:
        assert r.read().decode("utf-8").startswith("# Demo report")
    saved = runner.runs[run["id"]].run_dir
    assert {p.name for p in saved.iterdir()} == {"report.json", "report.md", "trace.jsonl", "transcript.txt"}


def test_a_reconnecting_stream_resumes_after_the_last_event(server):
    base, _ = server()
    _, run = post(base, {"goal": GOAL, "demo": True})
    events = stream(base, run["id"])
    req = urllib.request.Request(f"{base}/api/runs/{run['id']}/events", headers={"Last-Event-ID": str(len(events) - 2)})
    with urllib.request.urlopen(req, timeout=10) as r:
        resumed = [line for line in r.read().decode().splitlines() if line.startswith("data: ")]
    assert [json.loads(line[6:])["event"] for line in resumed] == ["done"]


def test_only_one_run_at_a_time(server):
    release = threading.Event()

    def slow_agent(settings, injector, tracer, demo):
        release.wait(10)
        return demo_agent(settings, injector, tracer)

    base, _ = server(slow_agent)
    status, first = post(base, {"goal": GOAL, "demo": True})
    status2, busy = post(base, {"goal": GOAL, "demo": True})
    release.set()
    assert (status, status2) == (202, 409) and busy["run_id"] == first["id"]
    stream(base, first["id"])
    assert post(base, {"goal": GOAL, "demo": True})[0] == 202  # free again once it finished


@pytest.mark.parametrize("body, message", [
    ({"goal": "hi", "demo": True}, "5 to 400"),
    ({"goal": GOAL, "demo": True, "inject": ["news_search:explode"]}, "unknown fault mode"),
    ({"goal": GOAL, "demo": True, "inject": ["news_serch:error"]}, "unknown fault target"),
    ({"goal": GOAL, "demo": True, "budget": 10}, "token budget"),
])
def test_bad_requests_are_explained(server, body, message):
    base, _ = server()
    status, data = post(base, body)
    assert status == 400 and message in data["error"]


def test_live_mode_needs_a_key(server, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    base, _ = server()
    status, data = post(base, {"goal": GOAL})
    assert status == 400 and "GROQ_API_KEY" in data["error"]


def test_requests_for_another_host_name_are_refused(server):
    base, _ = server()
    status, _ = post(base, {"goal": GOAL, "demo": True}, headers={"Host": "evil.example:8000"})
    assert status == 403


def test_non_json_posts_are_refused(server):
    base, _ = server()
    req = urllib.request.Request(base + "/api/runs", data=b"goal=x", method="POST",
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=5)
    assert e.value.code == 415


def test_a_negative_content_length_is_refused_instead_of_waiting(server):
    base, _ = server()
    req = urllib.request.Request(base + "/api/runs", data=b"", method="POST",
                                 headers={"Content-Type": "application/json", "Content-Length": "-1"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=5)
    assert e.value.code == 400
