# Sample runs

Four live runs against Groq's free tier: three on 2026-09-23, and one on 2026-09-26 with the final code. Each folder holds the unedited output of one run:
- `transcript.txt`: the console trace;
- `trace.jsonl`: one event per line;
- `report.md` and `report.json`: the report.

The commands are in [scripts/run_samples.ps1](../../scripts/run_samples.ps1). Sample 4 is sample 1's command, run again after the fixes described in the [live runs](../../README.md#live-runs-on-26-sep).

| # | Goal | Injected faults | Model | Steps passed | Grounded URLs | Confidence | Tokens | Time |
|---|---|---|---|---|---|---|---|---|
| 1 | Top 3 AI agent developments, last week | none | gpt-oss-120b | 7 / 8 | 3 / 3 | high | 67K | 13.6 min |
| 2 | same | `ddgs:error:*`, `fetch_page:timeout:1` | gpt-oss-120b | 2 / 6 | 0 / 3 | medium | 63K | 11.4 min |
| 3 | Tell me the latest about Python | `llm:bad_json:2` | gpt-oss-20b | 4 / 4 | 6 / 7 | medium | 52K | 9.3 min |
| 4 | same as 1, final code | none | gpt-oss-120b | 3 / 3 | 3 / 3 | medium | 42K | 3.8 min |

Samples 1–3 were made while the code was still changing, and all transcripts are kept as they were produced. Where a later fix changes what a run would show, the notes below say so.

**One later fix affects samples 1–3.** The reviewer used to see a step's output text but not the list of URLs the step cited. So a result that kept its URLs in its `sources` field could be rejected for "missing URLs". Rejections that mention missing URLs happened in sample 1 steps 1 and 5, sample 2 steps 1 and 5, and sample 3 step 2, and some of them may have been wrong. These traces can't show it, because step outputs weren't recorded yet. Now the reviewer sees the cited URLs, and `trace.jsonl` records each step's output and sources.

## 1. Brief goal, no injected faults: [1_brief-goal/](1_brief-goal/)

**Shows** planning, a real failure, a re-plan, and a fully grounded report.
- The planner's first reply was invalid (a step missing all its fields). One JSON repair fixed it.
- In step 3, one article failed to fetch. On the retry, the model said it had "fetched an archive.org version" without calling any tool. The deterministic check rejected this ("no tool was called"), and the re-plan swapped in steps 5–9, which found a readable source for that story.
- All 3 cited URLs were returned by tools (3 / 3 verified).
- Recoveries:

  | Recovery | Count |
  |---|---|
  | `llm_throttle` (pacing) | 15 |
  | `step_retry` | 4 |
  | `reasoning_salvage` | 4 |
  | `json_repair` | 1 |
  | `json_mode_off` | 1 |
  | `forced_finish` | 1 |
  | `replan` | 1 |

**Later fixes:**
- **Step counter:** the transcript shows "Step 9/8" after the re-plan. The counter now uses the last planned step id.
- **Recency cap:** the third finding is dated 2026-09-11, which is 12 days before the run, yet it was reported with *high* confidence. This run is why the recency check exists. Today the finding would be flagged "outside requested window" and confidence capped at medium.

## 2. Search provider down, one page fetch timed out: [2_injected-failures/](2_injected-failures/)

**Shows** degradation under sustained tool failure, and why grounding has to be checked in code.
- DuckDuckGo was down for the whole run, so every search fell back to Google News RSS (`provider_fallback` ×21). The first page fetch timed out and was retried (`tool_retry` ×9 in total).
- Google News returns redirect links that don't resolve to the article, so step 3 ("read each article") failed twice. The re-plan's first new step failed too. The token budget then ran out, the last two steps were skipped, and the report was still written.
- Recoveries:

  | Recovery | Count |
  |---|---|
  | `provider_fallback` | 21 |
  | `llm_throttle` | 14 |
  | `tool_retry` | 9 |
  | `reasoning_salvage` | 7 |
  | `forced_finish` | 3 |
  | `step_retry` | 3 |
  | `json_mode_off` | 1 |
  | `duplicate_call` | 1 |
  | `replan` | 1 |
  | `json_repair` | 1 |
  | `budget` | 1 |

**The honest failure:** with only headlines to work from, the report writer invented three plausible article URLs, one of them dated tomorrow (`techcrunch.com/2026/09/24/...`). It rated its confidence *medium*. The grounding check marked all three ✗ unverified (0 / 3), but at the time that didn't affect the confidence. Two fixes followed this run:
- the writer is now shown only URLs that a tool actually returned;
- if fewer than half the cited URLs are verified, confidence is capped at low and a limitation lists the unverified URLs. (Since then, confidence is also capped at low when fewer than half the steps pass, as here with 2 of 6.)

The retry prompt was also changed after this run: it now tells the model that its previous tool results are gone and it must call tools again.

## 3. Ambiguous goal, LLM returns bad JSON: [3_ambiguous-goal-bad-json/](3_ambiguous-goal-bad-json/)

**Shows** the last line of defence for planning, and grounding at the level of the exact URL.
- The two injected bad-JSON replies used up the first planning call and one repair, and the second repair returned a 1-step plan. The agent switched to its deterministic **fallback plan** (search news → web search → read sources → rank) and recorded that as an assumption in the report.
- Because of that, this run shows the goal handled *without* a planner. The ambiguity step (Python the language vs. anything else) never ran. It is covered offline by `test_ambiguous_goal_asks_for_clarification`, and `--interactive` asks the user.
- DuckDuckGo failed once in step 1 and Google News took over. In step 3 two page fetches failed and the model switched to other sources.
- The reviewer sent back steps 2, 3 and 4 once each: no URLs, generic features instead of concrete facts, and duplicate sources. All three passed on the retry.
- **Grounding caught a subtle error.** The tool returned `linuxjournal.com/.../fedora-linux-45-beta-released-python-315-gcc-162-major-security-changes`, but the report cites it cut short at `...-315-gcc`. That URL was never returned, so it is marked ✗ (6 / 7 verified). The model noticed as well: its limitations say the URL "appears truncated".
- Recoveries:

  | Recovery | Count |
  |---|---|
  | `json_repair` | 16 |
  | `llm_throttle` | 14 |
  | `step_retry` | 3 |
  | `llm_rate_limit` (real 429, handled with Retry-After) | 2 |
  | `fallback_plan` | 1 |
  | `provider_fallback` | 1 |
  | `forced_finish` | 1 |

**Model:** gpt-oss-20b, because samples 1 and 2 had used most of gpt-oss-120b's daily token quota. It is a drop-in replacement (`--model openai/gpt-oss-20b`).

**Later fixes:**
- **Missing `action`:** 14 of the 16 JSON repairs were executor replies without a valid `action` field. Each repair cost an LLM call, about 40% of this run's calls. A pydantic `before` validator now infers the action when `tool` or `output` shows the intent. This was only a partial fix. The later eval runs on gpt-oss-20b still needed 13–16 `action: Field required` repairs each, so many of these replies must have some other shape. Each repair now logs the raw reply, so the next live run will show which shape it is. See the [eval results](../../README.md#eval-results-so-far).
- **Empty confidence reason:** the model returned an empty confidence reason. The report now says "The model gave no reason." instead of leaving the field blank.

## 4. Brief goal again, final code: [4_brief-goal-rerun/](4_brief-goal-rerun/)

**Shows** the step-level URL check catching a real error, the news top-up, and a complete run in under 4 minutes.
- DuckDuckGo had nothing from the past week for "AI agent", so `news_search` topped up from Bing News (`provider_fallback`).
- The reviewer sent step 1 back once because its list left out the dates the criteria asked for.
- **In step 2 the model cut a URL short.** It cited `…/openai-admits-ai-agent` for `…/openai-admits-ai-agents-exposed-221956429.html`. No tool had returned the short URL, so the step failed in code before any review, and the retry cited the full URL.
- Step 3 read all three pages in one call. It was sent back once for missing dates and "quoted experts", both of which the planner had put in its criteria, and it passed on the retry.
- All 3 steps passed, all 3 cited URLs are verified, and all findings are inside the 7-day window. The run took 228 s, 42K tokens and 23 LLM calls.
- Recoveries:

  | Recovery | Count |
  |---|---|
  | `llm_throttle` (pacing) | 14 |
  | `reasoning_salvage` | 4 |
  | `step_retry` | 3 |
  | `forced_finish` | 2 |
  | `json_mode_off` | 1 |
  | `provider_fallback` | 1 |

**Later fixes:**
- **Top-up message:** it now reads "no results from the last 7 days" instead of "only 0 results".
- **Page dates:** step 3 spent two calls searching for publication dates. `fetch_page` now reports a page's date (`PUBLISHED:`) when its metadata has one.
