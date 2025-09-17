import os, re, json, time
from flask import Flask, request, session, redirect, url_for, render_template_string
from dotenv import load_dotenv

# --------------- Config ---------------
load_dotenv()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "sk-local")
MODEL_NAME      = os.getenv("MODEL_NAME", "qwen2.5:32b-instruct")
FLASK_SECRET    = os.getenv("FLASK_SECRET", "dev-secret")

POSTGRES_DSN    = os.getenv("POSTGRES_DSN", "postgresql://user:pass@localhost:5432/dbname")
SQL_ROW_LIMIT   = int(os.getenv("SQL_ROW_LIMIT", "500"))
HTTP_TIMEOUT_S  = float(os.getenv("HTTP_TIMEOUT_S", "8.0"))

# OpenAI-compatible client (OpenWebUI)
try:
    from openai import OpenAI
    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
except Exception:
    client = None

# DB + HTTP
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except Exception:
    psycopg2 = None
try:
    import requests
except Exception:
    requests = None

app = Flask(__name__)
app.secret_key = FLASK_SECRET

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

# --------------- Utils ---------------
def read_prompt(filename: str) -> str:
    path = os.path.join(PROMPTS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def call_llm(system_text: str, user_text: str, temperature: float = 0.2, max_tokens: int = 1200) -> str:
    if client is None:
        return "[LLM client not initialized]"
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user",   "content": user_text}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()

def get_state():
    if "state" not in session:
        session["state"] = {
            "question": "",
            "step1_router_json": "",
            "step2_norm_json": "",
            "step3_sql_text": "",
            "step4_api_json": "",
            "step5_sql_result_json": "",
            "step6_api_result_json": ""
        }
    return session["state"]

def set_state(updates: dict):
    st = get_state()
    st.update(updates)
    session["state"] = st

# --------------- Guards ---------------
_SQL_ALLOW_RE = re.compile(r'^(\\s*(with|select)\\b)', re.IGNORECASE | re.DOTALL)
_SQL_FORBID_RE = re.compile(r'\\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|commit|rollback)\\b', re.IGNORECASE)
_SQL_VIEW_RE = re.compile(r'from\\s+public\\s*\\.\\s*\"?PurchaseAllView\"?', re.IGNORECASE)

def guard_sql(sql_text: str) -> str:
    """Return sanitized SQL or raise ValueError."""
    if not _SQL_ALLOW_RE.search(sql_text or ""):
        raise ValueError("SQL должен начинаться с SELECT/WITH.")
    if _SQL_FORBID_RE.search(sql_text):
        raise ValueError("Запрещены DML/DDL операции.")
    if not _SQL_VIEW_RE.search(sql_text):
        raise ValueError("Разрешены запросы только к public.\"PurchaseAllView\".")
    # единичный запрос, без многокомандных пачек
    if ';' in sql_text.strip()[:-1]:
        raise ValueError("Несколько команд запрещены (найдена точка с запятой в середине).")
    return sql_text.strip()

def run_sql(sql_text: str, limit: int = SQL_ROW_LIMIT):
    if psycopg2 is None:
        return {"error": "psycopg2 не установлен"}
    sql = guard_sql(sql_text)
    # Принудительный лимит (если не указан)
    if re.search(r'\\blimit\\b', sql, re.IGNORECASE) is None:
        sql = f"{sql.rstrip(';')}\\nLIMIT {int(limit)};"
    conn = None
    t0 = time.time()
    try:
        conn = psycopg2.connect(POSTGRES_DSN)
        conn.autocommit = False
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchall()
        elapsed = round((time.time() - t0)*1000)
        # ограничим размер ответа
        preview = rows[:limit]
        return {
            "ok": True,
            "elapsed_ms": elapsed,
            "row_count": len(rows),
            "preview_limit": limit,
            "columns": list(preview[0].keys()) if preview else [],
            "rows": preview
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            if conn: conn.close()
        except Exception:
            pass

def run_api_plan(plan_json: str, timeout: float = HTTP_TIMEOUT_S):
    if requests is None:
        return {"error": "requests не установлен"}
    try:
        plan = json.loads(plan_json or "{}")
    except Exception as e:
        return {"error": f"Плохой JSON плана API: {e}"}
    need_api = bool(plan.get("need_api"))
    calls = plan.get("calls") or []
    merge_key = plan.get("merge_key") or "PurchaseCardId"
    extract = plan.get("extract") or []
    if not need_api:
        return {"ok": True, "need_api": False, "results": []}
    results = []
    for c in calls:
        cid = c.get("id")
        url = c.get("url")
        if not cid or not url:
            continue
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            results.append({"id": cid, "error": str(e)})
            continue
        # Попробуем вытащить стандартные поля, иначе отдадим весь JSON
        out = {"id": cid, "raw": data}
        if "current_status" in extract:
            out["current_status"] = data.get("current_status")
        if "last_change_at" in extract:
            out["last_change_at"] = data.get("last_change_at")
        if "history" in extract:
            out["history"] = data.get("history")
        results.append(out)
    return {"ok": True, "need_api": True, "merge_key": merge_key, "results": results}

# --------------- Steps ---------------
STEPS = {
    1: {
        "kind": "llm",
        "name": "Router (намерение и роутинг)",
        "sysfile": "01_router_system.txt",
        "desc": "Определяет A/B/A+B, need_api, фильтры и выдаёт JSON по схеме.",
        "build_user": lambda st: f"Вопрос: {st['question']}\\n"
    },
    2: {
        "kind": "llm",
        "name": "Normalizer (опечатки → ILIKE-шаблоны)",
        "sysfile": "02_normalizer_system.txt",
        "desc": "Нормализует search_terms из шага 1 и возвращает corrected + patterns (JSON).",
        "build_user": lambda st: st["step1_router_json"] or "{ }"
    },
    3: {
        "kind": "llm",
        "name": "SQL Composer (только SQL-черновик)",
        "sysfile": "03_sql_composer_system.txt",
        "desc": "Собирает полный SQL по правилам, опираясь на Router JSON + patterns.",
        "build_user": lambda st: json.dumps({
            "router": json.loads(st["step1_router_json"] or "{}"),
            "patterns": json.loads(st["step2_norm_json"] or '{"corrected":[],"patterns":[]}')
        }, ensure_ascii=False, indent=2)
    },
    4: {
        "kind": "llm",
        "name": "API Planner (JSON-only)",
        "sysfile": "04_api_planner_system.txt",
        "desc": "Планирует вызовы API статусов (JSON-only), без выполнения HTTP.",
        "build_user": lambda st: json.dumps({
            "need_api": (json.loads(st["step1_router_json"] or "{}").get("need_api", False)),
            "purchase_card_ids": []  # можешь заполнить вручную перед запуском
        }, ensure_ascii=False, indent=2)
    },
    5: {
        "kind": "exec",
        "name": "SQL Runner (исполнение SELECT)",
        "desc": "Выполняет сгенерированный SQL против Postgres (только SELECT) и сохраняет превью результата.",
        "build_user": lambda st: (st["step3_sql_text"] or "").strip()
    },
    6: {
        "kind": "exec",
        "name": "API Runner (вызовы статусов)",
        "desc": "Выполняет HTTP GET по calls из шага 4 и собирает статусы/таймлайн.",
        "build_user": lambda st: (st["step4_api_json"] or "").strip()
    }
}

# --------------- Templates ---------------
BASE_HTML = """
<!doctype html>
<html lang=\\"ru\\"><head>
<meta charset=\\"utf-8\\"><meta name=\\"viewport\\" content=\\"width=device-width, initial-scale=1\\">
<title>SGR Demo</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Arial,sans-serif;margin:24px;line-height:1.4}
h1{margin:0 0 12px} .muted{color:#666} pre{background:#f6f6f6;padding:12px;border-radius:8px;overflow:auto}
textarea{width:100%;min-height:160px;font-family:ui-monospace,Consolas,Monaco,monospace}
input[type=text]{width:100%} .row{display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap}
.card{border:1px solid #e5e5e5;border-radius:10px;padding:16px;margin:12px 0;flex:1}
.btn{background:#111;color:#fff;border:0;border-radius:8px;padding:10px 14px;cursor:pointer}
.btn.alt{background:#e0e0e0;color:#111}
label{font-weight:600} .kv{display:grid;grid-template-columns:160px 1fr;gap:8px}
.small{font-size:12px;color:#666}
</style>
</head><body>
{{ body|safe }}
</body></html>
"""

INDEX_HTML = """
<h1>SGR Demo — 6 шагов (Qwen 2.5 32B)</h1>
<form method=\\"post\\" action=\\"{{ url_for('start') }}\\" class=\\"card\\">
  <label>Вопрос руководителя / пользователя</label>
  <input type=\\"text\\" name=\\"question\\" value=\\"{{ question }}\\" placeholder=\\"Например: Найди все позиции линолеума за май по объекту МК-12 у Петрова\\">
  <div style=\\"margin-top:12px\\">
    <button class=\\"btn\\">Сохранить вопрос</button>
    {% if question %}
      <a class=\\"btn alt\\" href=\\"{{ url_for('step', n=1) }}\\">Перейти к шагу 1 →</a>
    {% endif %}
  </div>
</form>
<div class=\\"muted\\">Модель: {{ model }} | API: {{ base }} | DSN: {{ dsn }}</div>
"""

STEP_LLM_HTML = """
<h1>Шаг {{ n }} — {{ step['name'] }}</h1>
<div class=\\"card\\"><div class=\\"muted\\">{{ step['desc'] }}</div></div>

{% if prev_out %}
<div class=\\"card\\">
  <label>Результат предыдущего шага</label>
  <pre>{{ prev_out }}</pre>
</div>
{% endif %}

<div class=\\"row\\">
  <div class=\\"card\\">
    <label>System prompt (из файла: {{ step['sysfile'] }})</label>
    <pre>{{ system_text }}</pre>
  </div>
  <div class=\\"card\\">
    <form method=\\"post\\">
      <label>Параметры запроса (User)</label>
      <textarea name=\\"user_text\\">{{ user_text }}</textarea>
      <div style=\\"margin-top:12px\\">
        <button class=\\"btn\\" name=\\"action\\" value=\\"run\\">Запустить шаг</button>
        <a class=\\"btn alt\\" href=\\"{{ url_for('index') }}\\">На главную</a>
        {% if n < 6 %}
          {% if out_text %}<a class=\\"btn alt\\" href=\\"{{ url_for('step', n=n+1) }}\\">Далее →</a>{% endif %}
        {% else %}
          {% if out_text %}<a class=\\"btn alt\\" href=\\"{{ url_for('summary') }}\\">Сводка →</a>{% endif %}
        {% endif %}
      </div>
    </form>
  </div>
</div>

{% if out_text %}
<div class=\\"card\\">
  <label>Ответ шага {{ n }}</label>
  <pre>{{ out_text }}</pre>
</div>
{% endif %}
"""

STEP_EXEC_SQL_HTML = """
<h1>Шаг {{ n }} — {{ step['name'] }}</h1>
<div class=\\"card\\"><div class=\\"muted\\">{{ step['desc'] }}</div></div>

<div class=\\"row\\">
  <div class=\\"card\\">
    <label>SQL к исполнению (из шага 3, редактируемый)</label>
    <form method=\\"post\\">
      <textarea name=\\"sql_text\\">{{ sql_text }}</textarea>
      <div class=\\"kv\\" style=\\"margin-top:8px\\">
        <div class=\\"small\\">ROW LIMIT (принудительно)</div><input type=\\"text\\" name=\\"row_limit\\" value=\\"{{ row_limit }}\\">
      </div>
      <div style=\\"margin-top:12px\\">
        <button class=\\"btn\\" name=\\"action\\" value=\\"run_sql\\">Выполнить SQL</button>
        <a class=\\"btn alt\\" href=\\"{{ url_for('index') }}\\">На главную</a>
        {% if out_text %}<a class=\\"btn alt\\" href=\\"{{ url_for('step', n=n+1) }}\\">Далее →</a>{% endif %}
      </div>
    </form>
  </div>
  <div class=\\"card\\">
    <label>Подключение</label>
    <pre>POSTGRES_DSN = {{ dsn }}\\nПримечание: выполняется только SELECT к public.\\"PurchaseAllView\\"</pre>
  </div>
</div>

{% if out_text %}
<div class=\\"card\\">
  <label>Результат SQL</label>
  <pre>{{ out_text }}</pre>
</div>
{% endif %}
"""

STEP_EXEC_API_HTML = """
<h1>Шаг {{ n }} — {{ step['name'] }}</h1>
<div class=\\"card\\"><div class=\\"muted\\">{{ step['desc'] }}</div></div>

<div class=\\"row\\">
  <div class=\\"card\\">
    <label>План вызовов (JSON из шага 4, редактируемый)</label>
    <form method=\\"post\\">
      <textarea name=\\"api_plan\\">{{ api_plan }}</textarea>
      <div class=\\"kv\\" style=\\"margin-top:8px\\">
        <div class=\\"small\\">HTTP TIMEOUT (сек)</div><input type=\\"text\\" name=\\"timeout\\" value=\\"{{ timeout }}\\">
      </div>
      <div style=\\"margin-top:12px\\">
        <button class=\\"btn\\" name=\\"action\\" value=\\"run_api\\">Выполнить вызовы</button>
        <a class=\\"btn alt\\" href=\\"{{ url_for('index') }}\\">На главную</a>
        {% if out_text %}<a class=\\"btn alt\\" href=\\"{{ url_for('summary') }}\\">Сводка →</a>{% endif %}
      </div>
    </form>
  </div>
  <div class=\\"card\\">
    <label>Базовый URL</label>
    <pre>https://sigma.lgss-spb.ru/webhook/relay?id=&lt;PurchaseCardId&gt;</pre>
  </div>
</div>

{% if out_text %}
<div class=\\"card\\">
  <label>Результат API</label>
  <pre>{{ out_text }}</pre>
</div>
{% endif %}
"""

SUMMARY_HTML = """
<h1>Сводка</h1>
<div class=\\"card\\"><label>Изначальный вопрос</label><pre>{{ question }}</pre></div>
<div class=\\"card\\"><label>Router (шаг 1)</label><pre>{{ s1 }}</pre></div>
<div class=\\"card\\"><label>Normalizer (шаг 2)</label><pre>{{ s2 }}</pre></div>
<div class=\\"card\\"><label>SQL (шаг 3)</label><pre>{{ s3 }}</pre></div>
<div class=\\"card\\"><label>SQL результат (шаг 5)</label><pre>{{ s5 }}</pre></div>
<div class=\\"card\\"><label>API Planner (шаг 4)</label><pre>{{ s4 }}</pre></div>
<div class=\\"card\\"><label>API результат (шаг 6)</label><pre>{{ s6 }}</pre></div>
<div style=\\"margin-top:12px\\">
  <a class=\\"btn alt\\" href=\\"{{ url_for('index') }}\\">← На главную</a>
</div>
"""

# --------------- Routes ---------------
@app.route("/", methods=["GET"])
def index():
    st = get_state()
    body = render_template_string(INDEX_HTML, question=st["question"], model=MODEL_NAME, base=OPENAI_BASE_URL, dsn=POSTGRES_DSN)
    return render_template_string(BASE_HTML, body=body)

@app.route("/start", methods=["POST"])
def start():
    q = (request.form.get("question") or "").strip()
    set_state({
        "question": q,
        "step1_router_json": "",
        "step2_norm_json": "",
        "step3_sql_text": "",
        "step4_api_json": "",
        "step5_sql_result_json": "",
        "step6_api_result_json": ""
    })
    return redirect(url_for("step", n=1))

@app.route("/step/<int:n>", methods=["GET", "POST"])
def step(n: int):
    st = get_state()
    if n not in (1,2,3,4,5,6):
        return redirect(url_for("index"))

    # Prev output
    prev_out = ""
    if n == 1:
        prev_out = st["question"]
    elif n == 2:
        prev_out = st["step1_router_json"]
    elif n == 3:
        prev_out = st["step2_norm_json"] or "(Нет результата шага 2)"
    elif n == 4:
        prev_out = st["step3_sql_text"]
    elif n == 5:
        prev_out = st["step3_sql_text"]
    elif n == 6:
        prev_out = st["step4_api_json"]

    step_cfg = STEPS[n]
    kind = step_cfg["kind"]

    if kind == "llm":
        system_text = read_prompt(step_cfg["sysfile"])
        default_user = step_cfg["build_user"](st)
        out_key = {1:"step1_router_json", 2:"step2_norm_json", 3:"step3_sql_text", 4:"step4_api_json"}[n]
        out_text = st[out_key]

        if request.method == "POST" and request.form.get("action") == "run":
            user_text = request.form.get("user_text") or default_user
            result = call_llm(system_text, user_text)
            set_state({out_key: result})
            return redirect(url_for("step", n=n))

        body = render_template_string(
            STEP_LLM_HTML,
            n=n, step=step_cfg, system_text=system_text,
            user_text=default_user, out_text=out_text, prev_out=prev_out
        )
        return render_template_string(BASE_HTML, body=body)

    elif kind == "exec" and n == 5:
        sql_text = (st["step3_sql_text"] or "").strip()
        out_text = st["step5_sql_result_json"]
        if request.method == "POST" and request.form.get("action") == "run_sql":
            sql_text = request.form.get("sql_text") or sql_text
            row_limit = int(request.form.get("row_limit") or SQL_ROW_LIMIT)
            res = run_sql(sql_text, limit=row_limit)
            set_state({"step5_sql_result_json": json.dumps(res, ensure_ascii=False, indent=2)})
            set_state({"step3_sql_text": sql_text})
            return redirect(url_for("step", n=n))
        body = render_template_string(
            STEP_EXEC_SQL_HTML,
            n=n, step=step_cfg, sql_text=sql_text, row_limit=SQL_ROW_LIMIT,
            out_text=out_text, dsn=POSTGRES_DSN
        )
        return render_template_string(BASE_HTML, body=body)

    elif kind == "exec" and n == 6:
        api_plan = (st["step4_api_json"] or "").strip()
        out_text = st["step6_api_result_json"]
        if request.method == "POST" and request.form.get("action") == "run_api":
            api_plan = request.form.get("api_plan") or api_plan
            timeout = float(request.form.get("timeout") or HTTP_TIMEOUT_S)
            res = run_api_plan(api_plan, timeout=timeout)
            set_state({"step6_api_result_json": json.dumps(res, ensure_ascii=False, indent=2)})
            set_state({"step4_api_json": api_plan})
            return redirect(url_for("step", n=n))
        body = render_template_string(
            STEP_EXEC_API_HTML,
            n=n, step=step_cfg, api_plan=api_plan, timeout=HTTP_TIMEOUT_S, out_text=out_text
        )
        return render_template_string(BASE_HTML, body=body)

    return redirect(url_for("index"))

@app.route("/summary")
def summary():
    st = get_state()
    body = render_template_string(
        SUMMARY_HTML,
        question=st["question"],
        s1=st["step1_router_json"],
        s2=st["step2_norm_json"],
        s3=st["step3_sql_text"],
        s4=st["step4_api_json"],
        s5=st["step5_sql_result_json"],
        s6=st["step6_api_result_json"]
    )
    return render_template_string(BASE_HTML, body=body)

if __name__ == "__main__":
    debug = (os.getenv("FLASK_DEBUG", "true").lower() in {"1", "true", "yes", "on"})
    port = int(os.getenv("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=debug)
