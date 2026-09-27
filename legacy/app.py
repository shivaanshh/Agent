"""
MyAgent — Robust Local AI Agent (OpenAI GPT-4o)
Features: Web Search, Email (SMTP/IMAP), Calculator API, File Upload/Read/Write,
          Notes, Tasks with priorities/due dates, Memory, Unit Converter,
          Summarizer, Translator, Code Runner, Password Generator, Word Count
"""
import os, json, math, datetime, smtplib, imaplib, email as email_lib
import subprocess, sys, re, traceback, secrets, string
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import decode_header
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv

import fast_path

load_dotenv()

app = Flask(__name__)

# Same-origin only. The UI is served by this same Flask app, so it never
# needed cross-origin access -- but a bare CORS(app) allows ANY site to POST
# to /chat, and /chat can reach run_code. That turns any page you happen to
# visit into arbitrary code execution on this machine, through your own
# browser, with no prompt. Without the permissive header the preflight fails
# and the browser refuses to send it.
CORS(app, origins=["http://localhost:5000", "http://127.0.0.1:5000"])

# Groq serves an OpenAI-compatible endpoint, so the same SDK, the same tool
# schemas and the same ReAct loop run against either one; only the key, the
# base URL and the model name differ. Groq is preferred when its key is set
# because it has a free tier, and an assistant that stops answering "what
# time is it" because a balance hit zero is not much of an assistant.
#
# Set LLM_PROVIDER=openai to force OpenAI, or =groq to force Groq.
def _pick_provider():
    want = os.environ.get("LLM_PROVIDER", "auto").strip().lower()
    groq_key   = os.environ.get("GROQ_API_KEY", "").strip()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()

    if want == "groq" or (want == "auto" and groq_key):
        return ("groq",
                OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1"),
                os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"))
    return ("openai",
            OpenAI(api_key=openai_key),
            os.environ.get("OPENAI_MODEL", "gpt-4o"))


PROVIDER, client, MODEL = _pick_provider()

# Requests that need no model at all are answered locally. Set FAST_PATH=0
# to send everything through the LLM -- useful when checking whether a wrong
# answer came from a bad local match or from the model.
FAST_PATH = os.environ.get("FAST_PATH", "1").strip() != "0"

WORKSPACE  = Path("workspace");  WORKSPACE.mkdir(exist_ok=True)
UPLOADS    = Path("uploads");    UPLOADS.mkdir(exist_ok=True)
NOTES_FILE = Path("notes.json")
MEM_FILE   = Path("memory.json")

EMAIL_ADDRESS  = os.environ.get("EMAIL_ADDRESS", "")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "")
SMTP_HOST      = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT      = int(os.environ.get("SMTP_PORT", "587"))
IMAP_HOST      = os.environ.get("IMAP_HOST", "imap.gmail.com")

# ── PERSISTENCE ──────────────────────────────────────────────────────────────
def load_memory():
    if MEM_FILE.exists():
        return json.loads(MEM_FILE.read_text())
    return {"facts": [], "tasks": [], "notes": []}

def save_memory(mem):
    MEM_FILE.write_text(json.dumps(mem, indent=2, default=str))

def load_notes():
    if NOTES_FILE.exists():
        return json.loads(NOTES_FILE.read_text())
    return []

def save_notes(notes):
    NOTES_FILE.write_text(json.dumps(notes, indent=2, default=str))

# ── TOOLS ─────────────────────────────────────────────────────────────────────
TOOLS = [
    {"type":"function","function":{"name":"web_search","description":"Search the web for current real-time information.","parameters":{"type":"object","properties":{"query":{"type":"string"},"max_results":{"type":"integer","default":5}},"required":["query"]}}},
    {"type":"function","function":{"name":"calculator","description":"Evaluate any math: arithmetic, powers, sqrt, log, trig, compound interest, EMI. ALWAYS use for math.","parameters":{"type":"object","properties":{"expression":{"type":"string","description":"Python math expression e.g. '100000*(1.085**20)'"}},"required":["expression"]}}},
    {"type":"function","function":{"name":"unit_convert","description":"Convert units: length, weight, temperature, speed, data, area.","parameters":{"type":"object","properties":{"value":{"type":"number"},"from_unit":{"type":"string"},"to_unit":{"type":"string"}},"required":["value","from_unit","to_unit"]}}},
    {"type":"function","function":{"name":"write_file","description":"Write text to a file in workspace.","parameters":{"type":"object","properties":{"filename":{"type":"string"},"content":{"type":"string"}},"required":["filename","content"]}}},
    {"type":"function","function":{"name":"read_file","description":"Read a file from workspace or uploads.","parameters":{"type":"object","properties":{"filename":{"type":"string"}},"required":["filename"]}}},
    {"type":"function","function":{"name":"list_files","description":"List all files in workspace and uploads.","parameters":{"type":"object","properties":{},"required":[]}}},
    {"type":"function","function":{"name":"delete_file","description":"Delete a file from workspace.","parameters":{"type":"object","properties":{"filename":{"type":"string"}},"required":["filename"]}}},
    {"type":"function","function":{"name":"summarize_file","description":"Read and summarize any text file from uploads or workspace.","parameters":{"type":"object","properties":{"filename":{"type":"string"}},"required":["filename"]}}},
    {"type":"function","function":{"name":"add_task","description":"Add a task with priority (high/medium/low), optional due date and category.","parameters":{"type":"object","properties":{"task":{"type":"string"},"priority":{"type":"string","enum":["high","medium","low"],"default":"medium"},"due":{"type":"string"},"category":{"type":"string"}},"required":["task"]}}},
    {"type":"function","function":{"name":"list_tasks","description":"List tasks filtered by status or priority.","parameters":{"type":"object","properties":{"filter":{"type":"string","enum":["all","pending","done","high","medium","low"],"default":"pending"}},"required":[]}}},
    {"type":"function","function":{"name":"complete_task","description":"Mark task as done by its number from list_tasks.","parameters":{"type":"object","properties":{"task_number":{"type":"integer"}},"required":["task_number"]}}},
    {"type":"function","function":{"name":"delete_task","description":"Delete a task by its number.","parameters":{"type":"object","properties":{"task_number":{"type":"integer"}},"required":["task_number"]}}},
    {"type":"function","function":{"name":"add_note","description":"Save a quick note with title, content, and tags.","parameters":{"type":"object","properties":{"title":{"type":"string"},"content":{"type":"string"},"tags":{"type":"string","description":"comma-separated tags"}},"required":["title","content"]}}},
    {"type":"function","function":{"name":"list_notes","description":"List notes, optionally filter by tag.","parameters":{"type":"object","properties":{"tag":{"type":"string"}},"required":[]}}},
    {"type":"function","function":{"name":"delete_note","description":"Delete a note by number.","parameters":{"type":"object","properties":{"note_number":{"type":"integer"}},"required":["note_number"]}}},
    {"type":"function","function":{"name":"remember_fact","description":"Store a fact or user preference in persistent memory.","parameters":{"type":"object","properties":{"fact":{"type":"string"}},"required":["fact"]}}},
    {"type":"function","function":{"name":"recall_memory","description":"Retrieve all stored facts about the user.","parameters":{"type":"object","properties":{},"required":[]}}},
    {"type":"function","function":{"name":"forget_fact","description":"Remove a fact by its number from memory.","parameters":{"type":"object","properties":{"fact_number":{"type":"integer"}},"required":["fact_number"]}}},
    {"type":"function","function":{"name":"get_datetime","description":"Get current date, time, week number, day of year.","parameters":{"type":"object","properties":{},"required":[]}}},
    {"type":"function","function":{"name":"date_diff","description":"Calculate days between two dates or days until a date.","parameters":{"type":"object","properties":{"date1":{"type":"string","description":"YYYY-MM-DD or 'today'"},"date2":{"type":"string","description":"YYYY-MM-DD"}},"required":["date1","date2"]}}},
    {"type":"function","function":{"name":"send_email","description":"Send a real email via SMTP using .env credentials.","parameters":{"type":"object","properties":{"to":{"type":"string"},"subject":{"type":"string"},"body":{"type":"string"}},"required":["to","subject","body"]}}},
    {"type":"function","function":{"name":"read_inbox","description":"Read latest N emails from Gmail.","parameters":{"type":"object","properties":{"count":{"type":"integer","default":5}},"required":[]}}},
    {"type":"function","function":{"name":"search_email","description":"Search Gmail inbox by keyword.","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
    {"type":"function","function":{"name":"summarize_text","description":"Summarize any long text.","parameters":{"type":"object","properties":{"text":{"type":"string"},"length":{"type":"string","enum":["brief","detailed"],"default":"brief"}},"required":["text"]}}},
    {"type":"function","function":{"name":"translate_text","description":"Translate text to another language.","parameters":{"type":"object","properties":{"text":{"type":"string"},"target_language":{"type":"string"}},"required":["text","target_language"]}}},
    {"type":"function","function":{"name":"run_code","description":"Run Python code and return output.","parameters":{"type":"object","properties":{"code":{"type":"string"}},"required":["code"]}}},
    {"type":"function","function":{"name":"generate_password","description":"Generate a secure random password.","parameters":{"type":"object","properties":{"length":{"type":"integer","default":16},"include_symbols":{"type":"boolean","default":True}},"required":[]}}},
    {"type":"function","function":{"name":"word_count","description":"Count words, characters, sentences in text.","parameters":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}}},
]

# ── EXECUTOR ──────────────────────────────────────────────────────────────────
def execute_tool(name, args, mem):
    if name == "web_search":
        try:
            # duckduckgo_search was renamed to ddgs. The old package still
            # imports, which is the problem: it returns zero results instead
            # of failing, so the model concludes the web has nothing to say
            # and answers from memory. Prefer the new name, keep the old as
            # a fallback for a machine that only has that one.
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
            results = []
            with DDGS() as d:
                for r in d.text(args["query"], max_results=args.get("max_results",5)):
                    results.append(f"**{r['title']}**\n{r['href']}\n{r['body']}")
            return "\n\n---\n\n".join(results) if results else "No results found."
        except Exception as e:
            return f"Search error: {e}"

    if name == "calculator":
        try:
            safe = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
            safe.update({"abs":abs,"round":round,"pow":pow,"sum":sum,"min":min,"max":max})
            result = eval(args["expression"], {"__builtins__":{}}, safe)
            if isinstance(result, float):
                return f"= {result:,.6f}".rstrip("0").rstrip(".")
            return f"= {result:,}"
        except Exception as e:
            return f"Math error: {e}"

    if name == "unit_convert":
        v, f, t = args["value"], args["from_unit"].lower().strip(), args["to_unit"].lower().strip()
        conv = {
            ("km","m"):1000,("m","km"):0.001,("cm","m"):0.01,("m","cm"):100,
            ("km","miles"):0.621371,("miles","km"):1.60934,
            ("m","feet"):3.28084,("feet","m"):0.3048,("inch","cm"):2.54,("cm","inch"):0.393701,
            ("kg","g"):1000,("g","kg"):0.001,("kg","lb"):2.20462,("lb","kg"):0.453592,
            ("kg","oz"):35.274,("oz","kg"):0.0283495,
            ("gb","mb"):1024,("mb","gb"):1/1024,("tb","gb"):1024,("gb","tb"):1/1024,
            ("mb","kb"):1024,("kb","mb"):1/1024,
            ("kmh","ms"):0.277778,("ms","kmh"):3.6,("mph","kmh"):1.60934,("kmh","mph"):0.621371,
            ("sqm","sqft"):10.7639,("sqft","sqm"):0.092903,
            ("hectare","sqm"):10000,("sqm","hectare"):0.0001,
        }
        if f=="celsius" and t=="fahrenheit": return f"{v}°C = {v*9/5+32:.2f}°F"
        if f=="fahrenheit" and t=="celsius": return f"{v}°F = {(v-32)*5/9:.2f}°C"
        if f=="celsius" and t=="kelvin": return f"{v}°C = {v+273.15:.2f}K"
        if f=="kelvin" and t=="celsius": return f"{v}K = {v-273.15:.2f}°C"
        if (f,t) in conv:
            return f"{v} {args['from_unit']} = {v*conv[(f,t)]:,.4f} {args['to_unit']}"
        return f"Conversion {f} to {t} not supported."

    if name == "write_file":
        (WORKSPACE / args["filename"]).write_text(args["content"], encoding="utf-8")
        return f"Saved '{args['filename']}' ({len(args['content'])} chars)"

    if name == "read_file":
        for folder in [WORKSPACE, UPLOADS]:
            p = folder / args["filename"]
            if p.exists():
                c = p.read_text(encoding="utf-8", errors="replace")
                return c[:4000] + ("\n...(truncated)" if len(c)>4000 else "")
        return f"File '{args['filename']}' not found."

    if name == "list_files":
        ws = [(f.name, f.stat().st_size, "workspace") for f in sorted(WORKSPACE.iterdir())]
        up = [(f.name, f.stat().st_size, "uploads")   for f in sorted(UPLOADS.iterdir())]
        if not ws and not up: return "No files yet."
        return "\n".join(f"[{loc}] {n} ({s/1024:.1f}KB)" for n,s,loc in ws+up)

    if name == "delete_file":
        p = WORKSPACE / args["filename"]
        if p.exists(): p.unlink(); return f"Deleted '{args['filename']}'"
        return "File not found."

    if name == "summarize_file":
        for folder in [WORKSPACE, UPLOADS]:
            p = folder / args["filename"]
            if p.exists():
                c = p.read_text(encoding="utf-8", errors="replace")[:6000]
                r = client.chat.completions.create(model=MODEL,
                    messages=[{"role":"user","content":f"Summarize concisely:\n\n{c}"}], max_tokens=500)
                return r.choices[0].message.content
        return f"File not found."

    if name == "add_task":
        t = {"task":args["task"],"priority":args.get("priority","medium"),
             "due":args.get("due",""),"category":args.get("category","general"),
             "done":False,"created":datetime.datetime.now().isoformat()}
        mem.setdefault("tasks",[]).append(t)
        save_memory(mem)
        return f"Added [{t['priority'].upper()}] {t['task']}" + (f" due {t['due']}" if t["due"] else "")

    if name == "list_tasks":
        tasks = mem.get("tasks",[])
        f = args.get("filter","pending")
        if f == "pending":   tasks = [t for t in tasks if not t.get("done")]
        elif f == "done":    tasks = [t for t in tasks if t.get("done")]
        elif f in ("high","medium","low"): tasks = [t for t in tasks if t.get("priority")==f and not t.get("done")]
        if not tasks: return f"No {f} tasks."
        icons = {"high":"🔴","medium":"🟡","low":"🟢"}
        return "\n".join(f"{i}. {'✅' if t.get('done') else '○'} {icons.get(t.get('priority','medium'),'⚪')} {t['task']}" + (f" [due: {t['due']}]" if t.get("due") else "") + (f" #{t.get('category','')}" if t.get("category","general")!="general" else "") for i,t in enumerate(tasks,1))

    if name == "complete_task":
        pending = [t for t in mem.get("tasks",[]) if not t.get("done")]
        n = args["task_number"]-1
        if 0<=n<len(pending):
            pending[n]["done"]=True; pending[n]["completed_at"]=datetime.datetime.now().isoformat()
            save_memory(mem); return f"Completed: {pending[n]['task']}"
        return "Task not found."

    if name == "delete_task":
        pending = [t for t in mem.get("tasks",[]) if not t.get("done")]
        n = args["task_number"]-1
        if 0<=n<len(pending):
            text=pending[n]["task"]; mem["tasks"].remove(pending[n])
            save_memory(mem); return f"Deleted: {text}"
        return "Task not found."

    if name == "add_note":
        notes = load_notes()
        note = {"id":len(notes)+1,"title":args["title"],"content":args["content"],
                "tags":[t.strip() for t in args.get("tags","").split(",") if t.strip()],
                "created":datetime.datetime.now().isoformat()}
        notes.append(note); save_notes(notes)
        return f"Note saved: '{note['title']}'" + (f" [tags: {', '.join(note['tags'])}]" if note["tags"] else "")

    if name == "list_notes":
        notes = load_notes()
        tag = args.get("tag","")
        if tag: notes = [n for n in notes if tag.lower() in [t.lower() for t in n.get("tags",[])]]
        if not notes: return "No notes found."
        return "\n\n".join(f"**{i}. {n['title']}**\n{n['content'][:200]}{'...' if len(n['content'])>200 else ''}\nTags: {', '.join(n.get('tags',[])) or 'none'}" for i,n in enumerate(notes,1))

    if name == "delete_note":
        notes = load_notes(); n=args["note_number"]-1
        if 0<=n<len(notes):
            title=notes[n]["title"]; notes.pop(n); save_notes(notes); return f"Deleted note: '{title}'"
        return "Note not found."

    if name == "remember_fact":
        mem.setdefault("facts",[]).append({"fact":args["fact"],"ts":datetime.datetime.now().isoformat()})
        save_memory(mem); return f"Remembered: {args['fact']}"

    if name == "recall_memory":
        facts = mem.get("facts",[])
        if not facts: return "No memories."
        return "\n".join(f"{i}. {f['fact']}" for i,f in enumerate(facts,1))

    if name == "forget_fact":
        facts = mem.get("facts",[]); n=args["fact_number"]-1
        if 0<=n<len(facts):
            r=facts.pop(n)["fact"]; save_memory(mem); return f"Forgotten: {r}"
        return "Fact not found."

    if name == "get_datetime":
        now = datetime.datetime.now()
        return f"Date: {now.strftime('%A, %d %B %Y')}\nTime: {now.strftime('%I:%M:%S %p')} IST\nWeek #{now.isocalendar()[1]}  |  Day #{now.timetuple().tm_yday} of year"

    if name == "date_diff":
        try:
            d1 = datetime.date.today() if args["date1"]=="today" else datetime.date.fromisoformat(args["date1"])
            d2 = datetime.date.fromisoformat(args["date2"])
            diff = (d2-d1).days
            w,d = divmod(abs(diff),7)
            return f"{abs(diff)} days {'until' if diff>0 else 'since'} {args['date2']} ({w}w {d}d)"
        except Exception as e: return f"Date error: {e}"

    if name == "send_email":
        if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
            return "Email not configured. Add EMAIL_ADDRESS and EMAIL_PASSWORD to .env"
        try:
            msg = MIMEMultipart()
            msg["From"]=EMAIL_ADDRESS; msg["To"]=args["to"]; msg["Subject"]=args["subject"]
            msg.attach(MIMEText(args["body"],"plain"))
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as srv:
                srv.ehlo(); srv.starttls(); srv.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                srv.sendmail(EMAIL_ADDRESS, args["to"], msg.as_string())
            return f"Email sent to {args['to']} | Subject: {args['subject']}"
        except smtplib.SMTPAuthenticationError:
            return "Auth failed. Use a Gmail App Password (Google Account > Security > App Passwords)"
        except Exception as e: return f"Send failed: {e}"

    def _imap():
        if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
            return None, "Email not configured in .env"
        try:
            m = imaplib.IMAP4_SSL(IMAP_HOST); m.login(EMAIL_ADDRESS, EMAIL_PASSWORD); return m, None
        except Exception as e: return None, f"IMAP error: {e}"

    def _dh(v):
        if not v: return ""
        parts = decode_header(v)
        return " ".join((p.decode(e or "utf-8",errors="replace") if isinstance(p,bytes) else str(p)) for p,e in parts)

    if name == "read_inbox":
        mail, err = _imap()
        if err: return err
        try:
            mail.select("INBOX")
            _, data = mail.search(None,"ALL")
            ids = data[0].split()
            recent = list(reversed(ids[-args.get("count",5):]))
            results = []
            for eid in recent:
                _, md = mail.fetch(eid,"(RFC822)")
                msg = email_lib.message_from_bytes(md[0][1])
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type()=="text/plain":
                            body = part.get_payload(decode=True).decode("utf-8",errors="replace")[:200]; break
                else:
                    body = msg.get_payload(decode=True).decode("utf-8",errors="replace")[:200]
                results.append(f"From: {_dh(msg.get('From',''))}\nSubject: {_dh(msg.get('Subject',''))}\nDate: {msg.get('Date','')}\nPreview: {body.strip()}")
            mail.logout()
            return f"Latest {len(results)} emails:\n\n" + "\n\n---\n\n".join(results)
        except Exception as e: return f"Read inbox error: {e}"

    if name == "search_email":
        mail, err = _imap()
        if err: return err
        try:
            mail.select("INBOX")
            q = args["query"]
            _, s1 = mail.search(None, f'SUBJECT "{q}"')
            _, s2 = mail.search(None, f'FROM "{q}"')
            ids = list(set(s1[0].split()+s2[0].split()))
            if not ids: return f"No emails found for '{q}'"
            results = []
            for eid in ids[-5:]:
                _, md = mail.fetch(eid,"(RFC822)")
                msg = email_lib.message_from_bytes(md[0][1])
                results.append(f"From: {_dh(msg.get('From',''))}\nSubject: {_dh(msg.get('Subject',''))}\nDate: {msg.get('Date','')}")
            mail.logout()
            return f"Found {len(results)} for '{q}':\n\n" + "\n\n---\n\n".join(results)
        except Exception as e: return f"Search error: {e}"

    if name == "summarize_text":
        length = args.get("length","brief")
        instr = "in 3-5 sentences" if length=="brief" else "in detail with key points as bullet list"
        r = client.chat.completions.create(model=MODEL,
            messages=[{"role":"user","content":f"Summarize {instr}:\n\n{args['text'][:5000]}"}], max_tokens=600)
        return r.choices[0].message.content

    if name == "translate_text":
        r = client.chat.completions.create(model=MODEL,
            messages=[{"role":"user","content":f"Translate to {args['target_language']}. Return ONLY the translation:\n\n{args['text']}"}], max_tokens=600)
        return r.choices[0].message.content

    if name == "run_code":
        try:
            # sys.executable, not "python3": on Windows that name is either
            # absent or a Microsoft Store stub, and on a machine with several
            # Pythons it is whichever one the PATH happens to name -- not the
            # one running this app, which is the one with these packages.
            result = subprocess.run([sys.executable, "-c", args["code"]],
                                    capture_output=True, text=True, timeout=10)
            out = result.stdout[:1500]; err = result.stderr[:500]
            return out or (f"Error:\n{err}" if err else "(no output)")
        except subprocess.TimeoutExpired: return "Timed out (10s limit)"
        except Exception as e: return f"Run error: {e}"

    if name == "generate_password":
        length = args.get("length",16)
        chars = string.ascii_letters + string.digits + ("!@#$%^&*()_+-=[]{}|" if args.get("include_symbols",True) else "")
        pw = "".join(secrets.choice(chars) for _ in range(length))
        return f"Password ({length} chars):\n{pw}"

    if name == "word_count":
        text = args["text"]
        words = len(text.split()); chars = len(text)
        sentences = len(re.findall(r'[.!?]+',text)) or 1
        return f"Words: {words:,}  |  Chars: {chars:,}  |  Sentences: {sentences}  |  Reading time: ~{max(1,round(words/200))} min"

    return f"Unknown tool: {name}"

# ── ReAct LOOP ────────────────────────────────────────────────────────────────
def react(user_message, conversation_history, mem, uploaded_file=None):
    steps = []
    facts = "\n".join(f"• {f['fact']}" for f in mem.get("facts",[])) or "none"
    pending_tasks = [t for t in mem.get("tasks",[]) if not t.get("done")]
    tasks_str = "\n".join(f"• [{t.get('priority','?')}] {t['task']}" for t in pending_tasks) or "none"
    notes = load_notes()
    note_titles = ", ".join(n["title"] for n in notes) if notes else "none"

    system = f"""You are MyAgent — a smart, warm, highly capable personal AI assistant running locally.
You use the ReAct loop: reason carefully, call tools, observe, continue until done.

USER CONTEXT:
• Facts/preferences: {facts}
• Pending tasks ({len(pending_tasks)}): {tasks_str}
• Saved notes: {note_titles}
• Today: {datetime.datetime.now().strftime('%A, %d %B %Y, %I:%M %p')}
{f"• Uploaded file: {uploaded_file} (use read_file or summarize_file to access it)" if uploaded_file else ""}

RULES:
- ALWAYS use calculator for any math — never compute mentally
- ALWAYS use web_search for current events, prices, news, or facts
- For tasks: always set a priority (high/medium/low) and category when adding
- For notes: always add relevant tags
- Be warm, concise, genuinely helpful
- Proactively suggest next steps or related actions when useful"""

    messages = list(conversation_history)
    messages.append({"role":"user","content":user_message})

    for _ in range(15):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"system","content":system}]+messages,
            tools=TOOLS, tool_choice="auto",
            temperature=0.3, max_tokens=2000
        )
        msg    = response.choices[0].message
        finish = response.choices[0].finish_reason

        if finish == "stop":
            steps.append({"type":"answer","content":msg.content or ""})
            messages.append({"role":"assistant","content":msg.content})
            break

        if finish == "tool_calls" and msg.tool_calls:
            if msg.content:
                steps.append({"type":"think","content":msg.content})
            messages.append({
                "role":"assistant","content":msg.content,
                "tool_calls":[{"id":tc.id,"type":"function","function":{"name":tc.function.name,"arguments":tc.function.arguments}} for tc in msg.tool_calls]
            })
            for tc in msg.tool_calls:
                try: a = json.loads(tc.function.arguments)
                except: a = {}
                steps.append({"type":"tool_call","tool":tc.function.name,"input":a})
                result = execute_tool(tc.function.name, a, mem)
                steps.append({"type":"tool_result","tool":tc.function.name,"content":result})
                messages.append({"role":"tool","tool_call_id":tc.id,"content":result})
        else:
            if msg.content:
                steps.append({"type":"answer","content":msg.content})
            messages.append({"role":"assistant","content":msg.content or ""})
            break

    save_memory(mem)
    return steps, messages

PROVIDER_NAMES = {"openai": "OpenAI", "groq": "Groq"}


def explain_llm_error(e):
    """Turn a provider exception into something a person can act on.

    The raw text is a JSON blob with a type, a param and a code in it. Pasted
    into a chat bubble it reads as a crash, and it buries the one sentence that
    matters: what to do next.
    """
    text  = str(e)
    low   = text.lower()
    named = PROVIDER_NAMES.get(PROVIDER, PROVIDER)
    here  = f"(provider: {named}, model: {MODEL})"

    if "insufficient_quota" in low or "credit_balance_exhausted" in low or "billing" in low:
        other = "groq" if PROVIDER == "openai" else "openai"
        return (f"**{named} is out of credit** {here}\n\n"
                f"Nothing is wrong with the app. Either add credit to the "
                f"{named} account, or switch providers by setting "
                f"`LLM_PROVIDER={other}` in `.env` and restarting.\n\n"
                f"Local tools — running code, unit conversion, arithmetic, dates, "
                f"word count, tasks, notes, files — keep working either way; they "
                f"never call a model.")

    if "invalid_api_key" in low or "incorrect api key" in low or "401" in low:
        return (f"**The {named} API key was rejected** {here}\n\n"
                f"Check the key in `.env`, then restart the app — keys are read "
                f"once at startup.")

    if "rate_limit" in low or "429" in low:
        return (f"**Rate limited by {named}** {here}\n\nWait a moment and try again.")

    if "model_not_found" in low or "does not exist" in low or "decommissioned" in low:
        return (f"**The model `{MODEL}` is not available on {named}** {here}\n\n"
                f"Set a current one in `.env` and restart.")

    if "connection" in low or "timeout" in low or "timed out" in low:
        return (f"**Could not reach {named}** {here}\n\nCheck the network and try again.")

    return f"Something went wrong talking to {named} {here}:\n\n```\n{text[:600]}\n```"


# ── ROUTES ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data     = request.json
    mem      = load_memory()
    message  = data.get("message","")
    uploaded = data.get("uploaded_file")
    history  = list(data.get("history",[]))
    try:
        # A request the fast path recognises is one whose whole answer is a
        # local function call, so calling a model to reach the same function
        # adds a network round trip, a bill and a chance of being talked out
        # of it. An upload is never fast-pathed: "count the words" means the
        # file, and only the model knows that.
        hit = fast_path.match(message) if (FAST_PATH and not uploaded) else None

        if hit:
            name, args = hit
            result = execute_tool(name, args, mem)
            save_memory(mem)
            steps = [
                {"type":"tool_call",  "tool":name, "input":args, "local":True},
                {"type":"tool_result","tool":name, "content":result},
                {"type":"answer",     "content":result},
            ]
            history += [{"role":"user","content":message},
                        {"role":"assistant","content":result}]
        else:
            steps, history = react(message, history, mem, uploaded)

        ws_files = [{"name":f.name,"size":f.stat().st_size,"folder":"workspace"} for f in sorted(WORKSPACE.iterdir())]
        up_files = [{"name":f.name,"size":f.stat().st_size,"folder":"uploads"}   for f in sorted(UPLOADS.iterdir())]
        return jsonify({"steps":steps,"history":history,"memory":mem,"files":ws_files+up_files,"notes":load_notes()})
    except Exception as e:
        traceback.print_exc()
        friendly = explain_llm_error(e)
        return jsonify({"error":str(e),"steps":[{"type":"answer","content":friendly}]}), 500

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files: return jsonify({"error":"no file"}),400
    f    = request.files["file"]
    path = UPLOADS / f.filename
    f.save(str(path))
    try: preview = path.read_text(encoding="utf-8",errors="replace")[:300]
    except: preview = "[binary file]"
    return jsonify({"filename":f.filename,"size":path.stat().st_size,"preview":preview})

@app.route("/memory")
def get_memory(): return jsonify(load_memory())

@app.route("/notes")
def get_notes(): return jsonify(load_notes())

@app.route("/files")
def get_files():
    ws = [{"name":f.name,"size":f.stat().st_size,"folder":"workspace","modified":datetime.datetime.fromtimestamp(f.stat().st_mtime).strftime("%H:%M")} for f in sorted(WORKSPACE.iterdir())]
    up = [{"name":f.name,"size":f.stat().st_size,"folder":"uploads","modified":datetime.datetime.fromtimestamp(f.stat().st_mtime).strftime("%H:%M")} for f in sorted(UPLOADS.iterdir())]
    return jsonify(ws+up)

@app.route("/files/<folder>/<filename>")
def read_file_api(folder, filename):
    base = WORKSPACE if folder=="workspace" else UPLOADS
    p = base/filename
    if p.exists():
        try: content = p.read_text(encoding="utf-8",errors="replace")
        except: content = "[binary file]"
        return jsonify({"content":content,"name":filename})
    return jsonify({"error":"not found"}),404

@app.route("/files/<folder>/<filename>", methods=["DELETE"])
def delete_file_api(folder, filename):
    base = WORKSPACE if folder=="workspace" else UPLOADS
    p = base/filename
    if p.exists(): p.unlink(); return jsonify({"ok":True})
    return jsonify({"error":"not found"}),404

@app.route("/tasks", methods=["GET"])
def get_tasks(): return jsonify(load_memory().get("tasks",[]))

@app.route("/tasks/<int:idx>/complete", methods=["POST"])
def complete_task_api(idx):
    mem = load_memory()
    pending = [t for t in mem.get("tasks",[]) if not t.get("done")]
    if 0<=idx<len(pending):
        pending[idx]["done"]=True; pending[idx]["completed_at"]=datetime.datetime.now().isoformat()
        save_memory(mem); return jsonify({"ok":True})
    return jsonify({"error":"not found"}),404

@app.route("/tasks/<int:idx>", methods=["DELETE"])
def delete_task_api(idx):
    mem = load_memory()
    pending = [t for t in mem.get("tasks",[]) if not t.get("done")]
    if 0<=idx<len(pending):
        mem["tasks"].remove(pending[idx]); save_memory(mem); return jsonify({"ok":True})
    return jsonify({"error":"not found"}),404

if __name__ == "__main__":
    named = PROVIDER_NAMES.get(PROVIDER, PROVIDER)
    key   = os.environ.get("GROQ_API_KEY" if PROVIDER == "groq" else "OPENAI_API_KEY", "")
    print(f"\n{'OK' if key else 'MISSING'} {named} key ({MODEL})"
          f"  |  local fast path: {'on' if FAST_PATH else 'off'}"
          f"  |  Email: {EMAIL_ADDRESS or 'not set'}")
    if not key:
        print("   Local tools still work: code, math, units, dates, tasks, notes, files.")
    print("MyAgent -> http://localhost:5000\n")

    # 127.0.0.1, not 0.0.0.0. /chat can reach run_code, which runs arbitrary
    # Python as you, so binding to every interface puts a remote shell on the
    # network you happen to be joined to. Set HOST=0.0.0.0 deliberately if you
    # really do want to reach this from your phone, and only on a network you
    # trust.
    app.run(debug=True, port=int(os.environ.get("PORT", "5000")),
            host=os.environ.get("HOST", "127.0.0.1"))
