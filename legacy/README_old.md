# MyAgent — Local AI Agent (OpenAI GPT-4o)

A fully working, locally hosted AI agent with a beautiful dark UI.
Powered by GPT-4o · ReAct loop · Web Search · File Ops · Memory

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenAI API key
export OPENAI_API_KEY=sk-...        # Mac/Linux
set OPENAI_API_KEY=sk-...           # Windows CMD
$env:OPENAI_API_KEY="sk-..."        # Windows PowerShell

# 3. Run
python app.py

# 4. Open browser
# → http://localhost:5000
```

## Project Structure

```
agent_openai/
├── app.py              ← Flask backend + ReAct loop + all tools
├── requirements.txt
├── memory.json         ← Auto-created, persists across runs
├── workspace/          ← Files the agent creates
└── templates/
    └── index.html      ← Beautiful dark UI
```

## Tools

| Tool | What it does |
|------|-------------|
| `web_search` | Real DuckDuckGo search — no extra API key needed |
| `calculator` | Accurate math with full Python math module |
| `write_file` | Saves files to workspace/ folder |
| `read_file` | Reads files from workspace/ |
| `list_files` | Lists workspace contents |
| `remember_fact` | Stores facts in memory.json (persists across restarts) |
| `recall_memory` | Retrieves all stored memories |
| `get_datetime` | Current date and time |
| `add_task` | Adds to task list (shown in right panel) |
| `list_tasks` | Lists pending tasks |

## UI Layout

```
┌──────────────┬─────────────────────────┬──────────────┐
│  Tool list   │   Chat conversation     │  Workspace   │
│  + Memory    │   (ReAct steps shown)   │  + Tasks     │
│  panel       │   Input bar at bottom   │  panel       │
└──────────────┴─────────────────────────┴──────────────┘
```

## Demo flow for interview

1. "Search for the latest AI agent frameworks 2026"
   → web_search fires, shows real results

2. "Calculate compound interest on ₹1,00,000 at 8.5% for 20 years"
   → calculator tool fires with formula

3. "Remember that I'm an AI Engineer interviewing tomorrow"
   → remember_fact fires, Memory panel updates

4. "Write a file called react_loop.md explaining the ReAct pattern"
   → write_file fires, Workspace panel updates

5. "Add a task: Practice the OpenClaw demo"
   → add_task fires, Tasks panel updates
