# Generates the sample run transcripts in runs/samples/ (needs GROQ_API_KEY in .env).
# Usage (from the repo root):  powershell -ExecutionPolicy Bypass -File scripts\run_samples.ps1
$ErrorActionPreference = "Continue"
$out = "runs/samples"

# 1. Happy path: the goal from the brief.
python -m research_agent "Research and summarize the top 3 developments in AI agents from the last week" --out $out

# 2. Deliberately induced failures: primary search provider down for the whole run,
#    and the first page fetch times out. The agent must fall back and still report.
python -m research_agent "Research and summarize the top 3 developments in AI agents from the last week" `
    --out $out --inject "ddgs:error:*" --inject "fetch_page:timeout:1"

# 3. Ambiguous goal + corrupted LLM output: the planner surfaces the ambiguity and
#    records its assumption; the first two LLM replies are truncated and repaired.
python -m research_agent "Tell me the latest about Python" --out $out --inject "llm:bad_json:2"

Write-Host "`nDone. Transcripts are in $out/*/transcript.txt"
