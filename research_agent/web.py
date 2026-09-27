"""Web UI: start a run in the browser and watch the agent plan, act, check itself and report, live.

    python -m research_agent.web              # then open http://127.0.0.1:8000
    python -m research_agent.web --port 8080

Standard library only: ``http.server`` plus Server-Sent Events, which carry the tracer's
events to the page as they happen. One run at a time. Each run's artefacts are saved under
runs/, as the CLI saves them. "Demo mode" swaps the LLM for a rule-based stand-in
(research_agent/demo.py), so the page works without an API key.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlsplit

from dotenv import load_dotenv

from .agent import ResearchAgent
from .cli import _slug
from .config import Settings
from .demo import DemoClient, demo_settings
from .failures import TARGETS, FailureInjector
from .reporter import render_markdown
from .trace import RECOVERY_LABELS, Tracer

STATIC = Path(__file__).parent / "static"
MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
FAULT_PRESETS = [
    {"spec": "ddgs:error:*", "label": "DuckDuckGo is down", "effect": "search falls back to Bing News / Wikipedia"},
    {"spec": "news_search:error:*", "label": "news_search always fails", "effect": "the agent must switch tools"},
    {"spec": "fetch_page:timeout:1", "label": "First page fetch times out", "effect": "retried with backoff"},
    {"spec": "web_search:empty:2", "label": "web_search returns nothing, twice", "effect": "the agent rephrases or switches"},
    {"spec": "llm:bad_json:2", "label": "LLM returns broken JSON, twice", "effect": "repaired by re-asking"},
]
EXAMPLES = [
    "Research and summarize the top 3 developments in AI agents from the last week",
    "Compare the latest quarterly revenue of Nvidia and AMD",
    "Tell me the latest about Python",
]
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}

AgentFactory = Callable[[Settings, FailureInjector, Tracer, bool], ResearchAgent]


def default_factory(settings: Settings, injector: FailureInjector, tracer: Tracer, demo: bool) -> ResearchAgent:
    return ResearchAgent.build(settings, injector, tracer, llm_client=DemoClient() if demo else None)


class Busy(Exception):
    def __init__(self, run_id: str):
        super().__init__(run_id)
        self.run_id = run_id


class Run:
    """One agent run: its events so far, appended by the agent's thread and read by any number of streams."""

    def __init__(self, goal: str, demo: bool, faults: list[str]):
        self.id = uuid.uuid4().hex[:10]
        self.goal, self.demo, self.faults = goal, demo, faults
        self.events: list[dict] = []
        self.done = False
        self.run_dir: Optional[Path] = None
        self._cond = threading.Condition()

    def push(self, event: dict) -> None:
        with self._cond:
            self.events.append(event)
            self._cond.notify_all()

    def finish(self) -> None:
        with self._cond:
            self.done = True
            self._cond.notify_all()

    def wait(self, seen: int, timeout: float) -> tuple[list[dict], bool]:
        """Events after the first ``seen``, waiting up to ``timeout`` seconds for one to arrive."""
        with self._cond:
            if len(self.events) <= seen and not self.done:
                self._cond.wait(timeout)
            return self.events[seen:], self.done

    def summary(self) -> dict:
        return {"id": self.id, "goal": self.goal, "demo": self.demo, "faults": self.faults, "done": self.done}


class Runner:
    def __init__(self, out_dir: Path, factory: AgentFactory = default_factory):
        self.out_dir, self.factory = out_dir, factory
        self.runs: dict[str, Run] = {}
        self.current: Optional[Run] = None
        self._lock = threading.Lock()

    def config(self) -> dict:
        s = Settings.from_env()
        return {"has_key": bool(s.api_key), "model": s.model, "models": list(dict.fromkeys([s.model, *MODELS])),
                "budget": s.token_budget, "faults": FAULT_PRESETS, "examples": EXAMPLES,
                "recovery_labels": RECOVERY_LABELS,
                "current": self.current.summary() if self.current else None}

    def start(self, goal: str, faults: list[str], model: Optional[str] = None, demo: bool = False,
              budget: Optional[int] = None) -> Run:
        goal = " ".join(goal.split())
        if not 5 <= len(goal) <= 400:
            raise ValueError("The goal must be 5 to 400 characters long.")
        if model and not re.fullmatch(r"[\w./:-]{1,80}", model):
            raise ValueError("That model id doesn't look valid.")
        if budget is not None and not 5_000 <= budget <= 200_000:
            raise ValueError("The token budget must be between 5,000 and 200,000.")
        injector = FailureInjector.from_strings(faults, targets=TARGETS)  # ValueError names the bad spec
        if demo:
            settings = demo_settings(Settings.from_env(token_budget=budget))
        else:
            settings = Settings.from_env(model=model or None, token_budget=budget)
            if not settings.api_key:
                raise ValueError("GROQ_API_KEY is not set. Add it to .env and restart, or use demo mode.")
        with self._lock:
            if self.current and not self.current.done:
                raise Busy(self.current.id)
            run = self.current = Run(goal, demo, injector.describe())
            self.runs[run.id] = run
        threading.Thread(target=self._execute, args=(run, settings, injector), daemon=True).start()
        return run

    def _execute(self, run: Run, settings: Settings, injector: FailureInjector) -> None:
        tracer = Tracer(quiet=True)
        tracer.listeners.append(run.push)
        run_dir = self.out_dir / f"{dt.datetime.now():%Y%m%d-%H%M%S}_{'demo_' if run.demo else ''}{_slug(run.goal)}"
        try:
            result = self.factory(settings, injector, tracer, run.demo).run(run.goal)
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "report.json").write_text(json.dumps(result.report, indent=2, ensure_ascii=False),
                                                 encoding="utf-8")
            (run_dir / "report.md").write_text(render_markdown(result.report), encoding="utf-8")
        except Exception as e:  # noqa: BLE001 - show any failure on the page instead of a silent stop
            tracer.error(f"run aborted: {type(e).__name__}: {e}")
        finally:
            try:
                tracer.export(run_dir)
                run.run_dir = run_dir
            except OSError as e:
                tracer.error(f"could not save the run: {e}")
            run.push({"event": "done", "run_dir": str(run_dir)})
            run.finish()


class Handler(BaseHTTPRequestHandler):
    runner: Runner  # set by make_server
    server_version = "ResearchAgent"

    def log_message(self, fmt: str, *args) -> None:  # the page shows what matters; keep the console quiet
        pass

    def _local(self) -> bool:
        # When bound to this machine only, also refuse requests addressed to another host name
        # (DNS rebinding), so a web page can't drive the agent with your API key.
        if self.server.server_address[0] not in LOCAL_HOSTS:
            return True  # --host chose to expose the UI
        return urlsplit(f"//{self.headers.get('Host', '')}").hostname in LOCAL_HOSTS

    def do_GET(self) -> None:
        if not self._local():
            return self._json({"error": "forbidden host"}, 403)
        path = urlsplit(self.path).path
        if path == "/":
            return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/config":
            return self._json(self.runner.config())
        m = re.fullmatch(r"/api/runs/(\w+)/(events|report\.md|report\.json)", path)
        run = self.runner.runs.get(m.group(1)) if m else None
        if run is None:
            return self._json({"error": "not found"}, 404)
        if m.group(2) == "events":
            return self._stream(run)
        file = run.run_dir / m.group(2) if run.run_dir else None
        if file is None or not file.exists():
            return self._json({"error": "the report is not ready"}, 404)
        kind = "text/markdown; charset=utf-8" if file.suffix == ".md" else "application/json"
        self._send(200, file.read_bytes(), kind, {"Content-Disposition": f'attachment; filename="{file.name}"'})

    def do_POST(self) -> None:
        if not self._local():
            return self._json({"error": "forbidden host"}, 403)
        if urlsplit(self.path).path != "/api/runs":
            return self._json({"error": "not found"}, 404)
        # JSON only: a cross-site form can't send it without a CORS preflight, which this server never grants
        if not (self.headers.get("Content-Type") or "").startswith("application/json"):
            return self._json({"error": "send JSON"}, 415)
        raw_length = self.headers.get("Content-Length") or "0"
        if not raw_length.isdecimal():  # a negative length would make the read below wait for the client to hang up
            return self._json({"error": "bad Content-Length"}, 400)
        length = int(raw_length)
        if length > 10_000:
            return self._json({"error": "request too large"}, 413)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            faults = [str(f).strip() for f in body.get("inject") or [] if str(f).strip()]
            budget = int(body["budget"]) if body.get("budget") else None
            run = self.runner.start(str(body.get("goal", "")), faults, model=body.get("model") or None,
                                    demo=bool(body.get("demo")), budget=budget)
        except Busy as e:
            return self._json({"error": "A run is already in progress.", "run_id": e.run_id}, 409)
        except (ValueError, TypeError, AttributeError) as e:
            return self._json({"error": str(e) or "invalid request"}, 400)
        self._json(run.summary(), 202)

    def _stream(self, run: Run) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        last = self.headers.get("Last-Event-ID", "")
        seen = int(last) + 1 if last.isdigit() else 0  # a reconnecting browser resumes where it stopped
        try:
            while True:
                events, done = run.wait(seen, timeout=15)
                if not events and not done:
                    self.wfile.write(b": keep-alive\n\n")
                for e in events:
                    self.wfile.write(f"id: {seen}\ndata: {json.dumps(e, ensure_ascii=False, default=str)}\n\n"
                                     .encode("utf-8"))
                    seen += 1
                self.wfile.flush()
                if done and seen >= len(run.events):
                    return
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return  # the page was closed; the run carries on

    def _json(self, data: dict, status: int = 200) -> None:
        self._send(status, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json")

    def _send(self, status: int, body: bytes, kind: str, headers: Optional[dict] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)


def make_server(host: str, port: int, runner: Runner) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), type("BoundHandler", (Handler,), {"runner": runner}))
    server.daemon_threads = True
    return server


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m research_agent.web", description="Web UI for the research agent.")
    p.add_argument("--host", default="127.0.0.1", help="interface to bind (default: this machine only)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--out", default="runs", help="directory for run artefacts (default: runs/)")
    args = p.parse_args(argv)
    load_dotenv()
    server = make_server(args.host, args.port, Runner(Path(args.out)))
    print(f"Research agent UI: http://{args.host}:{server.server_address[1]}/  (Ctrl+C to stop)")
    if not Settings.from_env().api_key:
        print("GROQ_API_KEY is not set: only demo mode will work until you add it to .env and restart.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
