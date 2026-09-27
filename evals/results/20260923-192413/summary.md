| id | status | inject | plan_steps | steps_passed | findings | recoveries | recovery_kinds | grounding | confidence | tokens | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| compare-numbers | ok | - | 5 | 4/5 | 2 | 29 | budget,forced_finish,json_repair,llm_rate_limit,llm_throttle,step_retry | 1.0 | low | 42902 | 502.9 |
| llm-garbage | ok | llm:bad_json:2 | 4 | 0/4 | 1 | 65 | budget,duplicate_call,fallback_plan,forced_finish,json_repair,llm_rate_limit,llm_throttle,replan,step_retry | 1.0 | medium | 42751 | 509.0 |
