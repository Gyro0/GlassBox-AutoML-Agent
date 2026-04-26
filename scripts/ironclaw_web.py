"""Browser frontend for the local ``ironclaw`` CLI.

Runs ``ironclaw --cli-only --no-onboard -m <prompt> --auto-approve`` for each
browser turn and keeps a WebSocket session open so user replies are sent as
follow-up prompts with the previous transcript and CSV context.

Run from the repo root::

    pip install -e ".[demo]"
    python scripts/ironclaw_web.py

Then open http://127.0.0.1:8000.
"""

from __future__ import annotations

import asyncio
import csv as csv_lib
import re
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

REPO_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = REPO_ROOT / "data"
UPLOAD_NAME = "_uploaded.csv"
ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\)|[@-Z\\-_])")
FINAL_ANSWER_RE = re.compile(r"\n\s*[\u2500\-]{20,}\s*\n")
TIMEOUT_SECONDS = 240
ANSWER_IDLE_SECONDS = 1.5
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB
MAX_CONTEXT_COLUMNS = 24
MAX_TRANSCRIPT_CHARS = 12000

app = FastAPI(title="GlassBox · IronClaw web demo")

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GlassBox · IronClaw</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #f7f7f2;
      --panel: #ffffff;
      --ink: #171717;
      --muted: #62676f;
      --line: #dad7cc;
      --accent: #1b7f6d;
      --accent-strong: #116253;
      --console: #101312;
      --console-ink: #d9f4ea;
      --warn: #9b4d12;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
      line-height: 1.5;
    }
    main { max-width: 1180px; margin: 0 auto; padding: 28px 18px 34px; }
    header {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 20px;
      margin-bottom: 20px;
    }
    h1 { font-size: clamp(1.7rem, 3vw, 2.8rem); line-height: 1.05; margin: 0 0 .35rem; }
    .sub { color: var(--muted); margin: 0; max-width: 760px; }
    a { color: var(--accent-strong); }
    .status-pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: .45rem .75rem;
      color: var(--muted);
      background: rgba(255,255,255,.7);
      white-space: nowrap;
      font-size: .88rem;
    }
    .layout { display: grid; grid-template-columns: minmax(320px, 430px) 1fr; gap: 18px; align-items: start; }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }
    h2 { font-size: 1rem; margin: 0 0 12px; }
    label { display: block; color: var(--muted); font-size: .86rem; margin: 12px 0 5px; }
    textarea, input, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: transparent;
      color: inherit;
      font: inherit;
      padding: .68rem .75rem;
    }
    textarea { min-height: 170px; resize: vertical; }
    input[type=file] { padding: .58rem; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .actions { display: flex; gap: 10px; align-items: center; margin-top: 14px; }
    button {
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: white;
      cursor: pointer;
      font: inherit;
      padding: .72rem 1rem;
    }
    button.secondary { background: transparent; color: var(--ink); border: 1px solid var(--line); }
    button:disabled { opacity: .52; cursor: not-allowed; }
    .meta { color: var(--muted); font-size: .84rem; }
    .upload-state {
      border: 1px dashed var(--line);
      border-radius: 6px;
      padding: .75rem;
      margin-top: 8px;
      color: var(--muted);
      font-size: .88rem;
    }
    .console-wrap { padding: 0; overflow: hidden; }
    .console-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      background: var(--console);
      color: var(--console-ink);
      min-height: 430px;
      max-height: 66vh;
      padding: 1rem;
      overflow: auto;
      font: .9rem/1.5 ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
    }
    .reply-bar {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      padding: 12px;
      border-top: 1px solid var(--line);
      background: var(--panel);
    }
    .reply-bar input { margin: 0; }
    .tabs {
      display: flex;
      gap: 8px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    .tab-button {
      background: transparent;
      border: 1px solid var(--line);
      color: var(--ink);
      padding: .45rem .72rem;
    }
    .tab-button.active {
      background: var(--accent);
      border-color: var(--accent);
      color: white;
    }
    .pane { display: none; }
    .pane.active { display: block; }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #111412;
        --panel: #191d1b;
        --ink: #edf3ef;
        --muted: #a2ada7;
        --line: #323a36;
        --accent: #26a98f;
        --accent-strong: #7ce0cb;
        --console: #080a09;
        --console-ink: #d7f8ee;
      }
      .status-pill { background: rgba(25,29,27,.7); }
    }
    @media (max-width: 860px) {
      header { display: block; }
      .status-pill { display: inline-block; margin-top: 12px; }
      .layout { grid-template-columns: 1fr; }
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>GlassBox · IronClaw</h1>
        <p class="sub">A local browser console for IronClaw plus the GlassBox MCP server. Upload a CSV once, describe the target, then continue with follow-up turns without leaving the page.</p>
      </div>
      <div class="status-pill" id="connection">idle</div>
    </header>

    <div class="layout">
      <section>
        <h2>Run setup</h2>
        <form id="askForm">
          <label for="csvInput">CSV context</label>
          <input id="csvInput" type="file" name="csv" accept=".csv,text/csv">
          <div class="upload-state" id="uploadState">No CSV attached. If you choose one, IronClaw will be told its saved path automatically.</div>

          <label for="targetInput">Target column</label>
          <input id="targetInput" name="target" placeholder="e.g. purchased" value="purchased">

          <div class="grid">
            <div>
              <label for="taskInput">Task</label>
              <select id="taskInput" name="task">
                <option value="auto">auto</option>
                <option value="classification" selected>classification</option>
                <option value="regression">regression</option>
              </select>
            </div>
            <div>
              <label for="searchInput">Search</label>
              <select id="searchInput" name="search">
                <option value="random" selected>random</option>
                <option value="grid">grid</option>
              </select>
            </div>
          </div>

          <label for="budgetInput">Time budget, seconds</label>
          <input id="budgetInput" type="number" min="1" max="300" value="15">

          <label for="promptInput">Prompt</label>
          <textarea id="promptInput" name="prompt" required>Use the auto_fit tool with the provided CSV, target column, task, search strategy, and time budget. Summarize the best model, cross-validation score, and top features.</textarea>

          <div class="actions">
            <button type="submit" id="submitBtn">Ask IronClaw</button>
            <button type="button" class="secondary" id="stopBtn" disabled>Stop</button>
          </div>
          <p class="meta">Each turn runs IronClaw with the CSV context and the prior transcript, so replies continue the conversation cleanly.</p>
        </form>
      </section>

      <section class="console-wrap">
        <div class="console-head">
          <strong>Conversation</strong>
          <span id="liveMeta" class="meta">ready</span>
        </div>
        <div class="tabs" role="tablist" aria-label="Conversation views">
          <button type="button" class="tab-button active" data-tab="answer">Answer</button>
          <button type="button" class="tab-button" data-tab="details">Details</button>
        </div>
        <pre id="answerOut" class="pane active">(no answer yet)</pre>
        <pre id="detailsOut" class="pane">(no session yet)</pre>
        <form class="reply-bar" id="replyForm">
          <input id="replyInput" placeholder="Reply to IronClaw..." disabled>
          <button type="submit" id="replyBtn" disabled>Send</button>
        </form>
      </section>
    </div>
  </main>

  <script>
    const form = document.getElementById('askForm');
    const answerOut = document.getElementById('answerOut');
    const detailsOut = document.getElementById('detailsOut');
    const btn = document.getElementById('submitBtn');
    const stopBtn = document.getElementById('stopBtn');
    const csv = document.getElementById('csvInput');
    const uploadState = document.getElementById('uploadState');
    const live = document.getElementById('liveMeta');
    const connection = document.getElementById('connection');
    const replyForm = document.getElementById('replyForm');
    const replyInput = document.getElementById('replyInput');
    const replyBtn = document.getElementById('replyBtn');
    let socket = null;
    let uploadInfo = null;
    let timer = null;
    let sessionActive = false;
    let turnRunning = false;
    let answerTurns = [];
    let activeIronClawTurn = null;

    function setConnected(active) {
      sessionActive = active;
      btn.disabled = active;
      stopBtn.disabled = !active;
      connection.textContent = active ? 'connected' : 'idle';
      setTurnRunning(false);
    }

    function setTurnRunning(active) {
      turnRunning = active;
      replyInput.disabled = !sessionActive || active;
      replyBtn.disabled = !sessionActive || active;
      if (active) {
        live.textContent = 'streaming';
      } else if (sessionActive) {
        live.textContent = 'waiting for reply';
      }
    }

    document.querySelectorAll('.tab-button').forEach((button) => {
      button.addEventListener('click', () => {
        const tab = button.dataset.tab;
        document.querySelectorAll('.tab-button').forEach((b) => b.classList.toggle('active', b.dataset.tab === tab));
        document.querySelectorAll('.pane').forEach((pane) => pane.classList.toggle('active', pane.id === tab + 'Out'));
      });
    });

    function appendDetails(text) {
      const nearBottom = (detailsOut.scrollHeight - detailsOut.scrollTop - detailsOut.clientHeight) < 40;
      detailsOut.textContent += text;
      if (nearBottom) detailsOut.scrollTop = detailsOut.scrollHeight;
    }

    function renderAnswer(statusText = '') {
      const text = answerTurns.map((turn) => `${turn.role}:\\n${turn.text}`).join('\\n\\n');
      answerOut.textContent = text ? (statusText ? text + '\\n\\n' + statusText : text) : (statusText || '(no answer yet)');
      answerOut.scrollTop = answerOut.scrollHeight;
    }

    function addAnswerTurn(role, text) {
      answerTurns.push({ role, text });
      renderAnswer();
    }

    function updateIronClawTurn(text) {
      if (!text) return;
      if (activeIronClawTurn === null) {
        answerTurns.push({ role: 'IronClaw', text });
        activeIronClawTurn = answerTurns.length - 1;
      } else {
        answerTurns[activeIronClawTurn].text = text;
      }
      renderAnswer();
    }

    csv.addEventListener('change', () => {
      uploadInfo = null;
      const file = csv.files[0];
      uploadState.textContent = file ? `Ready to upload ${file.name} when the session starts.` : 'No CSV attached. If you choose one, IronClaw will be told its saved path automatically.';
    });

    async function uploadCsvIfNeeded() {
      if (!csv.files.length) return null;
      if (uploadInfo) return uploadInfo;
      const fd = new FormData();
      fd.append('csv', csv.files[0]);
      uploadState.textContent = 'Uploading CSV...';
      const r = await fetch('/upload', { method: 'POST', body: fd });
      if (!r.ok) throw new Error(await r.text());
      uploadInfo = await r.json();
      const cols = uploadInfo.columns.length ? ` Columns: ${uploadInfo.columns.join(', ')}${uploadInfo.truncated_columns ? ', ...' : ''}` : '';
      uploadState.textContent = `Uploaded to ${uploadInfo.path}.${cols}`;
      return uploadInfo;
    }

    function openSocket(payload) {
      socket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/ask`);
      socket.addEventListener('open', () => {
        socket.send(JSON.stringify({ type: 'start', ...payload }));
      });
      socket.addEventListener('message', (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === 'turn_start') {
          setTurnRunning(true);
          activeIronClawTurn = null;
          renderAnswer('IronClaw is working...');
        }
        if (msg.type === 'output') appendDetails(msg.text);
        if (msg.type === 'answer_update') updateIronClawTurn(msg.text);
        if (msg.type === 'answer') updateIronClawTurn(msg.text || '(no clean answer found; see Details)');
        if (msg.type === 'error') {
          appendDetails('\\n[web error] ' + msg.text + '\\n');
          renderAnswer('[web error] ' + msg.text);
        }
        if (msg.type === 'turn_done') {
          setTurnRunning(false);
          activeIronClawTurn = null;
          const reason = msg.reason === 'answer_idle' ? 'answer captured' : `ironclaw exited with code ${msg.code}`;
          appendDetails(`\\n[turn complete: ${reason}]\\n`);
        }
        if (msg.type === 'session_closed') {
          socket.close();
        }
      });
      socket.addEventListener('close', () => {
        setConnected(false);
        clearInterval(timer);
        live.textContent = 'session closed';
      });
      socket.addEventListener('error', () => {
        appendDetails('\\n[websocket failed]\\n');
        renderAnswer('[websocket failed]');
      });
    }

    stopBtn.addEventListener('click', () => {
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: 'stop' }));
      }
    });

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      answerTurns = [];
      activeIronClawTurn = null;
      answerOut.textContent = '(waiting for IronClaw answer...)';
      detailsOut.textContent = '';
      const t0 = Date.now();
      try {
        const uploaded = await uploadCsvIfNeeded();
        addAnswerTurn('You', document.getElementById('promptInput').value);
        setConnected(true);
        timer = setInterval(() => {
          const mode = turnRunning ? 'streaming' : 'waiting';
          live.textContent = mode + ' · ' + ((Date.now() - t0) / 1000).toFixed(1) + 's';
        }, 200);
        openSocket({
          prompt: document.getElementById('promptInput').value,
          target: document.getElementById('targetInput').value,
          task: document.getElementById('taskInput').value,
          search: document.getElementById('searchInput').value,
          budget: document.getElementById('budgetInput').value,
          upload: uploaded
        });
      } catch (err) {
        appendDetails('\\n[start failed: ' + err.message + ']\\n');
        renderAnswer('[start failed: ' + err.message + ']');
        setConnected(false);
      }
    });

    replyForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const text = replyInput.value;
      if (!text || !socket || socket.readyState !== WebSocket.OPEN) return;
      appendDetails('\\n> ' + text + '\\n');
      addAnswerTurn('You', text);
      socket.send(JSON.stringify({ type: 'input', text }));
      replyInput.value = '';
      replyInput.focus();
      setTurnRunning(true);
    });
  </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


def _save_upload(upload: UploadFile) -> str:
    """Persist an uploaded CSV to a fixed, predictable repo-relative path."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target = UPLOAD_DIR / UPLOAD_NAME
    total = 0
    with target.open("wb") as fh:
        while True:
            chunk = upload.file.read(64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                target.unlink(missing_ok=True)
                raise ValueError(
                    f"Upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."
                )
            fh.write(chunk)
    return str(target.relative_to(REPO_ROOT)).replace("\\", "/")


def _uploaded_columns(path: str) -> tuple[list[str], bool]:
    target = REPO_ROOT / path
    try:
        with target.open("r", encoding="utf-8-sig", newline="") as fh:
            header = next(csv_lib.reader(fh), [])
    except (OSError, UnicodeDecodeError, StopIteration, csv_lib.Error):
        return [], False
    columns = [col.strip() for col in header if col.strip()]
    return columns[:MAX_CONTEXT_COLUMNS], len(columns) > MAX_CONTEXT_COLUMNS


def _build_prompt(
    prompt: str,
    upload: dict | None,
    target: str,
    task: str,
    search: str,
    budget: str,
) -> str:
    context = _build_context_block(upload, target, task, search, budget)
    if not context:
        return prompt
    return prompt.rstrip() + "\n\n" + context


def _build_context_block(
    upload: dict | None,
    target: str,
    task: str,
    search: str,
    budget: str,
) -> str:
    notes = []
    tool_args = {}
    if upload and upload.get("path"):
        column_text = ""
        columns = upload.get("columns") or []
        if columns:
            suffix = ", ..." if upload.get("truncated_columns") else ""
            column_text = f" Its detected columns include: {', '.join(columns)}{suffix}."
        notes.append(
            "A CSV was uploaded by the user and saved at "
            f"{upload['path']}.{column_text} Use this path when calling GlassBox tools."
        )
        tool_args["csv_path"] = upload["path"]
    if target.strip():
        notes.append(f"The intended target column is {target.strip()}.")
        tool_args["target_column"] = target.strip()
    if task.strip() or search.strip() or budget.strip():
        clean_task = task.strip() or "auto"
        clean_search = search.strip() or "random"
        clean_budget = budget.strip() or "15"
        notes.append(
            "Preferred auto_fit settings: "
            f"task={clean_task}, "
            f"search={clean_search}, "
            f"time_budget={clean_budget}."
        )
        tool_args["task"] = clean_task
        tool_args["search"] = clean_search
        tool_args["time_budget"] = _safe_int(clean_budget, 15)
    if not notes:
        return ""
    lines = ["Context for this run:", "- " + "\n- ".join(notes)]
    if tool_args:
        lines.extend(
            [
                "",
                "When calling the GlassBox auto_fit MCP tool, use these exact argument names and values:",
                _format_tool_args(tool_args),
                "",
                "Do not call auto_fit with an empty argument object. Do not ask the user to confirm values already listed above.",
            ]
        )
    return "\n".join(lines)


def _format_tool_args(args: dict) -> str:
    lines = ["{"]
    for index, (key, value) in enumerate(args.items()):
        comma = "," if index < len(args) - 1 else ""
        if isinstance(value, int):
            rendered = str(value)
        else:
            rendered = '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'
        lines.append(f'  "{key}": {rendered}{comma}')
    lines.append("}")
    return "\n".join(lines)


def _safe_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _build_followup_prompt(
    original_prompt: str,
    context_block: str,
    transcript: list[tuple[str, str]],
) -> str:
    transcript_text = _format_transcript(transcript)
    return (
        "Continue this GlassBox/IronClaw conversation. Preserve the user's uploaded CSV context and tool settings.\n\n"
        f"Original user request:\n{original_prompt.strip()}\n\n"
        f"{context_block}\n\n"
        "Conversation so far:\n"
        f"{transcript_text}\n\n"
        "Respond to the latest user message. If enough information is now available, call the GlassBox auto_fit MCP tool with the exact arguments from the context block."
    )


def _format_transcript(transcript: list[tuple[str, str]]) -> str:
    lines = []
    remaining = MAX_TRANSCRIPT_CHARS
    for role, text in reversed(transcript):
        entry = f"{role}: {text.strip()}\n\n"
        if len(entry) > remaining:
            entry = entry[-remaining:]
        lines.append(entry)
        remaining -= len(entry)
        if remaining <= 0:
            break
    return "".join(reversed(lines)).strip()


def _extract_visible_answer(raw_output: str) -> str:
    """Pull the user-facing assistant text out of IronClaw's verbose CLI trace."""
    normalized = _normalize_cli_text(raw_output)
    if not normalized.strip():
        return ""

    segments = FINAL_ANSWER_RE.split(normalized)
    candidates = [_filter_detail_lines(segment) for segment in segments]
    candidates = [candidate for candidate in candidates if _has_answer_text(candidate)]
    if candidates:
        return candidates[-1].strip()
    return _filter_detail_lines(normalized).strip()


def _extract_streaming_answer(raw_output: str) -> str:
    """Extract only once IronClaw has started printing its final answer block."""
    if not FINAL_ANSWER_RE.search(raw_output):
        return ""
    return _extract_visible_answer(raw_output)


def _normalize_cli_text(text: str) -> str:
    text = ANSI_RE.sub("", text)
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _filter_detail_lines(text: str) -> str:
    lines = []
    skip_json_block = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if lines and lines[-1]:
                lines.append("")
            continue
        if _is_detail_line(line):
            skip_json_block = line.startswith(("{", "["))
            continue
        if skip_json_block:
            if line.endswith(("}", "]")):
                skip_json_block = False
            continue
        lines.append(raw_line.strip())
    return _trim_blank_lines("\n".join(lines))


def _is_detail_line(line: str) -> bool:
    detail_patterns = [
        r"^\d{4}-\d\d-\d\dT.*\b(INFO|WARN|ERROR|DEBUG|TRACE)\b",
        r"^[\u25cb\u25cf\u25c8\u2717\u280b]\s+",
        r"^(Processing|Thinking|Reading memory|Searching memory|Running)\b",
        r"^(memory_read|memory_search|memory_tree|glassbox_auto_fit)\b",
        r"^(Loaded|Injecting|Starting heartbeat|Failed to connect)\b",
        r"^\[\s*\"",
        r"^\{\s*\"(content|query|result_count|error|path)",
    ]
    return any(re.search(pattern, line) for pattern in detail_patterns)


def _has_answer_text(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 12:
        return False
    letters = sum(ch.isalpha() for ch in stripped)
    return letters >= 8


def _trim_blank_lines(text: str) -> str:
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


@app.post("/upload", response_model=None)
def upload_csv(csv: UploadFile = File(...)) -> JSONResponse | PlainTextResponse:
    if not csv.filename:
        return PlainTextResponse("No CSV file selected.", status_code=400)
    try:
        saved_path = _save_upload(csv)
    except ValueError as exc:
        return PlainTextResponse(f"Error: {exc}", status_code=400)
    columns, truncated = _uploaded_columns(saved_path)
    return JSONResponse(
        {
            "path": saved_path,
            "filename": csv.filename,
            "columns": columns,
            "truncated_columns": truncated,
        }
    )


async def _run_ironclaw_turn(
    websocket: WebSocket,
    prompt: str,
    stop_event: asyncio.Event,
) -> tuple[int | None, str, str, str]:
    """Run one clean IronClaw message turn and stream its output."""
    await websocket.send_json({"type": "turn_start"})
    try:
        proc = await asyncio.create_subprocess_exec(
            "ironclaw",
            "--cli-only",
            "--no-onboard",
            "-m",
            prompt,
            "--auto-approve",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(REPO_ROOT),
        )
    except FileNotFoundError:
        await websocket.send_json(
            {"type": "error", "text": "'ironclaw' CLI not found on PATH. Install it first."}
        )
        stop_event.set()
        return None, "", "", "missing_cli"

    assert proc.stdout is not None
    chunks: list[str] = []
    last_streamed_answer = ""
    last_answer_at: float | None = None
    answer_seen = asyncio.Event()
    loop = asyncio.get_running_loop()

    async def pump_output() -> None:
        nonlocal last_answer_at, last_streamed_answer
        while True:
            chunk = await proc.stdout.read(2048)
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            cleaned = ANSI_RE.sub("", text)
            chunks.append(cleaned)
            await websocket.send_json({"type": "output", "text": cleaned})
            answer = _extract_streaming_answer("".join(chunks))
            if answer and answer != last_streamed_answer:
                last_streamed_answer = answer
                last_answer_at = loop.time()
                answer_seen.set()
                await websocket.send_json({"type": "answer_update", "text": answer})

    output_task = asyncio.create_task(pump_output())
    wait_task = asyncio.create_task(proc.wait())
    stop_task = asyncio.create_task(stop_event.wait())
    answer_task = asyncio.create_task(answer_seen.wait())
    try:
        deadline = loop.time() + TIMEOUT_SECONDS
        while True:
            timeout = max(0.0, deadline - loop.time())
            if timeout <= 0:
                await _terminate_process(proc)
                await websocket.send_json(
                    {
                        "type": "error",
                        "text": f"IronClaw timed out after {TIMEOUT_SECONDS} seconds.",
                    }
                )
                raw = "".join(chunks)
                answer = _extract_visible_answer(raw)
                await websocket.send_json({"type": "answer", "text": answer})
                return proc.returncode, raw, answer, "timeout"

            done, _ = await asyncio.wait(
                {wait_task, stop_task, answer_task},
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stop_task in done and stop_event.is_set():
                await _terminate_process(proc)
                raw = "".join(chunks)
                return proc.returncode, raw, _extract_visible_answer(raw), "stopped"
            if wait_task in done:
                raw = "".join(chunks)
                answer = _extract_visible_answer(raw)
                if answer != last_streamed_answer:
                    await websocket.send_json({"type": "answer", "text": answer})
                return wait_task.result(), raw, answer, "process_exit"
            if answer_task in done:
                while proc.returncode is None and not stop_event.is_set():
                    if last_answer_at is not None and loop.time() - last_answer_at >= ANSWER_IDLE_SECONDS:
                        await _terminate_process(proc)
                        raw = "".join(chunks)
                        answer = _extract_visible_answer(raw)
                        if answer != last_streamed_answer:
                            await websocket.send_json({"type": "answer", "text": answer})
                        return proc.returncode, raw, answer, "answer_idle"
                    await asyncio.sleep(0.2)

                raw = "".join(chunks)
                answer = _extract_visible_answer(raw)
                if answer != last_streamed_answer:
                    await websocket.send_json({"type": "answer", "text": answer})
                reason = "stopped" if stop_event.is_set() else "process_exit"
                return proc.returncode, raw, answer, reason
    finally:
        await output_task
        for task in (wait_task, stop_task, answer_task):
            task.cancel()
        if proc.returncode is None:
            await _terminate_process(proc)


async def _terminate_process(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=3)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()


@app.websocket("/ws/ask")
async def ask_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        first = await websocket.receive_json()
    except (WebSocketDisconnect, ValueError):
        return
    if first.get("type") != "start":
        await websocket.send_json({"type": "error", "text": "First message must be type=start."})
        return

    cleaned = str(first.get("prompt", "")).strip()
    if not cleaned:
        await websocket.send_json({"type": "error", "text": "Empty prompt."})
        return

    upload = first.get("upload")
    target = str(first.get("target", ""))
    task = str(first.get("task", ""))
    search = str(first.get("search", ""))
    budget = str(first.get("budget", ""))
    context_block = _build_context_block(upload, target, task, search, budget)
    prompt = cleaned if not context_block else cleaned + "\n\n" + context_block

    transcript: list[tuple[str, str]] = [("User", cleaned)]
    control_queue: asyncio.Queue[dict] = asyncio.Queue()
    stop_event = asyncio.Event()

    async def receive_controls() -> None:
        while not stop_event.is_set():
            try:
                msg = await websocket.receive_json()
            except WebSocketDisconnect:
                stop_event.set()
                break
            except ValueError:
                continue
            msg_type = msg.get("type")
            if msg_type == "stop":
                stop_event.set()
                break
            if msg_type == "input":
                await control_queue.put(msg)

    receiver_task = asyncio.create_task(receive_controls())
    try:
        current_prompt = prompt
        while not stop_event.is_set():
            rc, output, answer, reason = await _run_ironclaw_turn(
                websocket, current_prompt, stop_event
            )
            transcript_text = answer.strip() or output.strip()
            if transcript_text:
                transcript.append(("IronClaw", transcript_text))
            await websocket.send_json({"type": "turn_done", "code": rc, "reason": reason})
            if stop_event.is_set():
                break

            get_task = asyncio.create_task(control_queue.get())
            stop_task = asyncio.create_task(stop_event.wait())
            done, _ = await asyncio.wait(
                {get_task, stop_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stop_task in done:
                get_task.cancel()
                break
            stop_task.cancel()
            msg = get_task.result()
            reply = str(msg.get("text", "")).strip()
            if not reply:
                continue
            transcript.append(("User", reply))
            current_prompt = _build_followup_prompt(cleaned, context_block, transcript)
    finally:
        stop_event.set()
        receiver_task.cancel()
        try:
            await websocket.send_json({"type": "session_closed"})
        except WebSocketDisconnect:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
