#!/usr/bin/env sh
# Same as run_samples.ps1, for macOS/Linux.
set -u
OUT=runs/samples
python -m research_agent "Research and summarize the top 3 developments in AI agents from the last week" --out $OUT
python -m research_agent "Research and summarize the top 3 developments in AI agents from the last week" \
    --out $OUT --inject 'ddgs:error:*' --inject fetch_page:timeout:1
python -m research_agent "Tell me the latest about Python" --out $OUT --inject llm:bad_json:2
echo "Done. Transcripts are in $OUT/*/transcript.txt"
