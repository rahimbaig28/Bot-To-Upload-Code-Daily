import os, json, pathlib, re, datetime, requests, sys, random, unicodedata, tempfile, ast, html
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_URL = "https://api.perplexity.ai/chat/completions"
MODEL = os.getenv("PPLX_MODEL", "sonar")
TIMEOUT = int(os.getenv("PPLX_TIMEOUT", "300"))

# Folders: versioned artifacts + "latest" convenience copies
DIST = pathlib.Path("dist")
PROMPTS_DIR = DIST / "prompts"
APPS_DIR = DIST / "apps"
LATEST_DIR = DIST / "latest"
LOG_OUT = DIST / "log.txt"

# --- Tunables ---
PROMPT_TEMPERATURE = 0.8
BUILD_TEMPERATURE  = 0.3
MAX_RESPONSE_CHARS = 800_000  # hard cap to avoid runaway responses

THEMES = [
    "personal productivity", "learning & study tools", "health & wellness",
    "small business utilities", "finance & budgeting", "data visualization",
    "environment & sustainability", "accessibility helpers", "micro games",
    "research utilities"
]
FEATURE_EXTRAS = [
    "keyboard-first UX", "offline-first with localStorage", "export/import JSON",
    "drag-and-drop", "ARIA-compliant components", "responsive grid",
    "print-friendly view", "dark/light auto theme", "share via URL hash",
    "undo/redo"
]

SYSTEM_PROMPT_GEN = """You are a senior product spec writer.
Task: Produce a concise, buildable prompt for a small single-file app.
Requirements:
- The output MUST be PLAIN TEXT only (no code fences or commentary).
- Title on the first line. Then a clear bulleted spec.
- The app may be EITHER:
  (a) a SINGLE-FILE web app (one HTML file with inline CSS and JS, no external libraries), OR
  (b) a SINGLE-FILE Python 3 script (no external files; standard library only).
- Include concrete features, accessibility/keyboard support when relevant, and persistence (e.g., localStorage or JSON file for Python).
- Keep it < 250 lines of text. Avoid vague language.
- End result must be implementable in under ~1000 lines of code.
"""

USER_PROMPT_GEN_TEMPLATE = """Generate a brand new single-file app specification.
Inspiration seed (date/time & randoms):
- utc_now: {utc}
- seed: {seed}
- theme hint: {theme}
- extra features to consider: {extras}

Constraints:
- Return ONLY the prompt (title then spec). No explanations or markdown fences.
- Keep it distinct from previous days by varying purpose/features.
- Include a testable feature set that can be implemented in one file (HTML or Python).
Example titles: "Smart To-Do", "Budget Buddy", "Focus Timer+" (do NOT reuse these).
"""

SYSTEM_BUILD = """You are a strict code generator.
Return EITHER:
(A) ONLY a complete, valid single-file HTML document starting with <!DOCTYPE html>
    - Inline all CSS and JS (no external links, no modules, no CDNs).
    - Ensure accessibility (labels, roles, aria-*), keyboard navigation, responsiveness.
    - Persist data with localStorage when relevant.
    - Minimal, clean styling.
    - NO markdown fences. NO commentary.
OR
(B) ONLY a single-file Python 3 script
    - Standard library only. No external files required.
    - Persist data if relevant (e.g., local JSON).
    - Provide a simple CLI or minimal TUI if appropriate.
    - NO markdown fences. NO commentary.
Implement exactly the user's spec. Keep the code concise and readable.
"""

SYSTEM_FIX = """You are a corrective code editor.
Given BROKEN content and a TARGET_KIND ('html' or 'py'), return ONLY a fixed version that:
- For html: starts with <!DOCTYPE html>, contains <html>, <head>, <body>, inline CSS/JS only, no external refs.
- For py: valid Python 3 single-file script using stdlib only; compiles under ast.parse().
Do not explain anything. Output ONLY the corrected file content.
"""

def _session_with_retries() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5, connect=5, read=5,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("POST", "GET"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

def write_text_atomic(path: pathlib.Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=str(path.parent)) as tf:
        tf.write(content)
        tmp_name = tf.name
    os.replace(tmp_name, path)

def read_text_safe(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""

def write_log(line: str):
    DIST.mkdir(parents=True, exist_ok=True)
    prev = read_text_safe(LOG_OUT)
    write_text_atomic(LOG_OUT, prev + line + "\n")

def pplx_chat(messages, api_key, temperature, timeout=TIMEOUT) -> str:
    sess = _session_with_retries()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": MODEL, "temperature": float(temperature), "messages": messages}
    resp = sess.post(API_URL, headers=headers, data=json.dumps(payload), timeout=(15, timeout))
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:600]}") from e
    try:
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"Non-JSON response: {resp.text[:600]}") from e
    if "choices" not in data or not data["choices"]:
        raise RuntimeError(f"Malformed API reply (missing choices): {data}")
    content = data["choices"][0]["message"].get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Empty content from API.")
    if len(content) > MAX_RESPONSE_CHARS:
        content = content[:MAX_RESPONSE_CHARS]
    return content

def strip_code_fences(text: str) -> str:
    fence = re.search(r"```(?:\w+)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    return fence.group(1) if fence else text

def extract_html_or_python(raw: str) -> tuple[str, str]:
    """
    Return (content, kind) where kind is 'html' or 'py'.
    """
    text = strip_code_fences(raw).strip()
    lower = text.lower()
    if "<!doctype html>" in lower or "<html" in lower:
        return text, "html"
    py_markers = (
        r"\bimport\b", r"\bfrom\b\s+\w+\s+\bimport\b",
        r"\bdef\s+\w+\(", r"\bclass\s+\w+\(", r"if\s+__name__\s*==\s*['\"]__main__['\"]"
    )
    if any(re.search(p, text) for p in py_markers):
        return text, "py"
    if text.startswith("<"):
        return text, "html"
    return text, "py"

def enforce_single_file_html(html_text: str) -> str:
    t = html_text
    if "<!doctype html>" not in t.lower():
        if "<html" not in t.lower():
            t = f"<!DOCTYPE html>\n<html><head><meta charset='utf-8'><title>App</title></head><body>\n{t}\n</body></html>"
        else:
            t = "<!DOCTYPE html>\n" + t
    # viewport + charset + minimal title
    if "<head" in t.lower():
        t = re.sub(r"(?i)<head>", "<head>\n<meta charset='utf-8'>\n<meta name='viewport' content='width=device-width,initial-scale=1'>\n", t, count=1)
    # Remove external refs
    t = re.sub(r'<script[^>]+src=["\'][^"\']+["\'][^>]*>\s*</script>', "", t, flags=re.IGNORECASE)
    t = re.sub(r'<link[^>]+rel=["\']stylesheet["\'][^>]*>', "", t, flags=re.IGNORECASE)
    # Stamp (comment)
    stamp = f"<!-- Auto-generated via Perplexity on {datetime.datetime.utcnow().isoformat()}Z -->"
    if stamp not in t:
        if "<head" in t.lower():
            t = re.sub(r"(?i)<head>", f"<head>\n{stamp}\n", t, count=1)
        else:
            t = stamp + "\n" + t
    return t

def add_python_stamp(py_text: str) -> str:
    stamp = f"# Auto-generated via Perplexity on {datetime.datetime.utcnow().isoformat()}Z"
    if not py_text.lstrip().startswith("# Auto-generated via Perplexity"):
        return stamp + "\n" + py_text
    return py_text

def validate_html(text: str) -> tuple[bool, str]:
    lower = text.lower()
    problems = []
    if not lower.startswith("<!doctype html"):
        problems.append("Missing <!DOCTYPE html> at top.")
    for tag in ("<html", "<head", "<body"):
        if tag not in lower:
            problems.append(f"Missing required tag: {tag}>")
    # Quick sanity: no <script src=...> or <link rel=stylesheet>
    if re.search(r'<script[^>]+src=', lower):
        problems.append("External <script src> found (must be inline).")
    if re.search(r'<link[^>]+rel=["\']stylesheet', lower):
        problems.append("External <link rel=stylesheet> found (must be inline).")
    return (len(problems) == 0, "; ".join(problems))

def validate_python(text: str) -> tuple[bool, str]:
    try:
        ast.parse(text)
        return True, ""
    except SyntaxError as e:
        return False, f"Python syntax error: {e}"

def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    text = re.sub(r"[\s_-]+", "-", text)
    return text or "app"

def first_line_title(prompt_text: str) -> str:
    for line in prompt_text.splitlines():
        if line.strip():
            return line.strip()
    return "app"

def unique_path(base: pathlib.Path) -> pathlib.Path:
    if not base.exists():
        return base
    stem, suffix = base.stem, base.suffix
    i = 2
    while True:
        candidate = base.with_name(f"{stem}-{i}{suffix}")
        if not candidate.exists():
            return candidate
        i += 1

def save_latest(kind: str, content: str):
    if kind == "html":
        write_text_atomic(LATEST_DIR / "app.html", content)
        # Remove stale other type if present
        py = LATEST_DIR / "app.py"
        if py.exists(): py.unlink()
    else:
        write_text_atomic(LATEST_DIR / "app.py", content)
        h = LATEST_DIR / "app.html"
        if h.exists(): h.unlink()

def build_once(prompt_text: str, api_key: str, temperature: float):
    raw = pplx_chat(
        messages=[{"role": "system", "content": SYSTEM_BUILD},
                  {"role": "user", "content": prompt_text}],
        api_key=api_key,
        temperature=temperature,
    )
    content, kind = extract_html_or_python(raw)
    if kind == "html":
        content = enforce_single_file_html(content)
        ok, err = validate_html(content)
    else:
        content = add_python_stamp(content)
        ok, err = validate_python(content)
    return ok, err, content, kind, raw

def attempt_fix(broken: str, target_kind: str, api_key: str):
    """Ask model once to fix the broken content."""
    fixed = pplx_chat(
        messages=[
            {"role": "system", "content": SYSTEM_FIX},
            {"role": "user", "content": f"TARGET_KIND={target_kind}\n\nBROKEN:\n{broken}"}
        ],
        api_key=api_key,
        temperature=0.1,
    )
    content, kind_detected = extract_html_or_python(fixed)
    # Force expected kind when needed
    if target_kind == "html":
        content = enforce_single_file_html(content)
        ok, err = validate_html(content)
    else:
        content = add_python_stamp(content)
        ok, err = validate_python(content)
    return ok, err, content, target_kind

def main():
    api_key = os.getenv("PPLX_API_KEY")
    if not api_key:
        print("Missing PPLX_API_KEY secret", file=sys.stderr)
        sys.exit(1)

    now = datetime.datetime.utcnow()
    utc_now = now.isoformat() + "Z"

    # Randomization inputs for prompt diversity
    seed = random.randint(10_000, 99_999)
    theme = random.choice(THEMES)
    extras = ", ".join(random.sample(FEATURE_EXTRAS, k=3))

    # ----- Stage 1: Generate the prompt -----
    user_prompt_gen = USER_PROMPT_GEN_TEMPLATE.format(
        utc=utc_now, seed=seed, theme=theme, extras=extras
    )

    try:
        prompt_text = pplx_chat(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_GEN},
                {"role": "user", "content": user_prompt_gen},
            ],
            api_key=api_key,
            temperature=PROMPT_TEMPERATURE,
        ).strip()
        # Basic sanity: must be plain text, no code fences
        prompt_text = strip_code_fences(prompt_text).strip()
        if not prompt_text or "\n" not in prompt_text:
            raise RuntimeError("Prompt looked empty or lacked a title + spec lines.")
        title = first_line_title(prompt_text)
        slug = slugify(title)
        prompt_path = unique_path(PROMPTS_DIR / f"{slug}.txt")
        write_text_atomic(prompt_path, prompt_text + f"\n\n(Generated: {utc_now}, seed={seed}, theme={theme}, extras={extras})\n")
        write_text_atomic(LATEST_DIR / "prompt.txt", prompt_text)
        print(f"Wrote {prompt_path}")
        write_log(f"[{utc_now}] Prompt OK  seed={seed}  theme={theme}  extras={extras}  title='{title}'  slug={slug}")
    except Exception as e:
        err = f"[{utc_now}] Prompt generation FAILED: {type(e).__name__}: {e}"
        print(err, file=sys.stderr)
        write_log(err)
        sys.exit(1)

    # ----- Stage 2: Build the app; validate; auto-fix once if needed -----
    try:
        ok, err, content, kind, raw = build_once(prompt_text, api_key, BUILD_TEMPERATURE)
        tried_fix = False
        if not ok:
            write_log(f"[{utc_now}] Build validation failed ({kind}): {err}. Attempting auto-fix.")
            ok, err, content, kind = attempt_fix(raw, kind, api_key)
            tried_fix = True

        if not ok:
            raise RuntimeError(f"Final {kind} validation failed after {'fix ' if tried_fix else ''}attempt: {err}")

        # Save versioned + latest
        if kind == "html":
            base = APPS_DIR / f"{slug}.html"
        else:
            base = APPS_DIR / f"{slug}.py"
        app_path = unique_path(base)
        write_text_atomic(app_path, content)
        save_latest(kind, content)

        print(f"Wrote {app_path}")
        write_log(f"[{utc_now}] Build OK  file={app_path.name}  kind={kind}  {'(after auto-fix)' if tried_fix else ''}")
        print(f"Done.\nSaved:\n  - {prompt_path}\n  - {app_path}\nLatest copies updated in {LATEST_DIR}")
    except Exception as e:
        err = f"[{utc_now}] Build FAILED: {type(e).__name__}: {e}"
        print(err, file=sys.stderr)
        write_log(err)
        sys.exit(1)

if __name__ == "__main__":
    main()
