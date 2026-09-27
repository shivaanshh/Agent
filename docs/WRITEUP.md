# Write-up: design decisions, limitations, next steps

## Design decisions

**Plan-and-Execute outside, bounded ReAct inside.** One ReAct loop over the whole goal has no plan to show or check, and a fixed plan breaks when a tool fails. So the planner writes 2–6 steps, each with a `tool_hint` and *checkable* `success_criteria`. Each step runs a ReAct loop of at most 4 actions, so it can adapt, for example by switching to `web_search` when `news_search` is down. Re-planning replaces only the remaining steps, at most once, so every run ends.

**JSON actions, not native tool calling.** Tool-calling quality varies across models and providers. One pydantic-validated action per turn works on any model and puts the model's `thought` in the trace. Validation errors go back to the model for repair.

**Verify before moving on.** Checks in code run first and cost no tokens. The output must not be trivial, a tool call must have succeeded, and every cited URL must have been returned by a tool. These checks catch the most common failures: answering from memory, and padding a list with invented items. An LLM reviewer (gpt-oss-20b) then judges the output against the criteria. Its feedback drives the step's single retry.

**Recovery is layered, cheapest first:** transport retry → provider fallback → the LLM switches tool → step retry with feedback → re-plan → deterministic fallback plan or report. Tools never raise; errors come back as data. The report call is exempt from the token budget, so every run ends with a report. `--inject` triggers these failures on purpose, through the same code paths as real errors, and the tests use it too.

**Grounding over trust.** Code checks the report's URLs against what tools returned, and its dates against the goal's window. It then caps the model's confidence. Each cap came from a real run:
- sample 1 called a 12-day-old story "this week" with high confidence;
- sample 2 invented three plausible URLs while search was down, so the writer now sees only URLs a tool returned;
- an eval run claimed medium confidence with 0 of 4 steps passed.

## What four live runs of the brief's goal showed

1. **Invented items.** DuckDuckGo found one fresh story for a "top 5" step. The model wrote four more from memory, and the reviewer passed them. The report's check still capped confidence at low. → URLs are now checked on every step, and `news_search` tops up from the next provider.
2. **Unreadable pages.** The plan demanded full article texts. msn.com needs JavaScript, nytimes.com refused, and washingtonpost.com stalled for 49 s. → Stalled pages aren't retried, and plans no longer hinge on particular pages.
3. **A strict reviewer.** It failed a step for summarising pages. → It is now told that results are short by design.
4. **The final run.** All 3 steps passed in 228 s, and the new check caught a URL the model had cut short.

## Limitations

- **Grounding covers URLs and dates, not claims.** A sentence citing a verified URL can still overstate what the page says. The reviewer is also the same model family as the worker.
- **LLM judgement drifts.** The planner still writes wish-list criteria such as "quoted experts", and the reviewer enforces them, which costs retries. "Top 3" is the model's call; there is no scoring model.
- **Free sources are fragile.** DuckDuckGo rate-limits, and it sometimes answers "past week" with old stories. Many news pages block bots, so findings often rest on search snippets.
- **The free tier sets the speed.** gpt-oss-120b gets 8K tokens a minute and 200K a day, and a run uses 40–65K.
  - Waiting for the minute limit took at least 180 of the final run's 228 s.
  - Four changes took the same goal from 818 s to 355 s: a token bucket synced to Groq's headers, reviews on 20b's separate limit, reading 3 pages per fetch, and ending steps early. A later run with no failed steps took 228 s.
  - Runs with failures still take 6–7 minutes, and the quota allows 3–4 runs a day.
- **Evals are behavioural:** plan validity, pass rate, recoveries and grounding. There are no gold answers for "this week's news", and the set wasn't re-run after the last fixes.

## With more time

1. **Claim-level checks:** test each finding against its page's text.
2. **Better sources:** a paid search API, and a headless browser for JavaScript pages.
3. **Cheaper routing:** run routine executor turns on 20b. Measure the optional Cerebras lane live; it is built and tested offline.
4. **Parallel steps:** plan as a graph and run independent branches concurrently.
5. **Replayable evals:** frozen search snapshots, reference answers and an LLM judge, run in CI.
