import os, json, pathlib, re, datetime, requests, sys, random, unicodedata
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
Task: Produce a concise, *buildable* prompt for a SINGLE-FILE web app.
Requirements:
- The output MUST be PLAIN TEXT only (no code fences or commentary).
- Title on the first line. Then a clear bulleted spec.
- Must demand ONE HTML file with inline CSS and JS, no external libraries.
- Include concrete features, accessibility, keyboard support, and persistence.
- Keep it < 250 lines of text. Avoid vague language.
- End result must be implementable in under ~1000 lines of HTML+CSS+JS.
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
- Include a testable feature set that can be implemented in one HTML file.
Example titles: "Smart To-Do", "Budget Buddy", "Focus Timer+" (do NOT reuse these).
"""

SYSTEM_BUILD = """You are a strict code generator.
Output ONLY a complete, valid single-file HTML document starting with <!DOCTYPE html>.
- Inline all CSS and JS (no external links, no modules, no CDNs).
- Implement exactly the user's spec.
- Ensure accessibility (labels, roles, aria-*), keyboard navigation, responsiveness.
- Persist data with localStorage when relevant.
- Include minimal, clean styling.
- Keep the code concise and readable; avoid over-engineering.
- NO markdown fences. NO extra commentary.
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

def pplx_chat(messages, api_key, temperature, timeout=TIMEOUT):
    sess = _session_with_retries()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": MODEL, "temperature": float(temperature), "messages": messages}
    resp = sess.post(API_URL, headers=headers, data=json.dumps(payload), timeout=(15, timeout))
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]

def extract_html(raw: str) -> str:
    # If code fences present, extract; else return as-is.
    fence = re.search(r"```(?:html)?\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE)
    html = fence.group(1) if fence else raw

    # Ensure <!DOCTYPE html>
    if "<!doctype html>" not in html.lower():
        if "<html" not in html.lower():
            html = f"<!DOCTYPE html>\n<html><head><meta charset='utf-8'><title>App</title></head><body>\n{html}\n</body></html>"
        else:
            html = "<!DOCTYPE html>\n" + html

    # Remove external refs to enforce single-file
    html = re.sub(r'<script[^>]+src=["\'][^"\']+["\'][^>]*></script>', "", html, flags=re.IGNORECASE)
    html = re.sub(r'<link[^>]+rel=["\']stylesheet["\'][^>]*>', "", html, flags=re.IGNORECASE)

    # Stamp time for guaranteed diffs
    stamp = f"<!-- Auto-generated via Perplexity on {datetime.datetime.utcnow().isoformat()}Z -->"
    html = re.sub(r"(?i)<head>", f"<head>\n{stamp}\n", html, count=1)
    if stamp not in html:
        html = stamp + "\n" + html
    return html

def slugify(text: str) -> str:
    # Normalize, remove accents, keep alphanum and dashes, collapse spaces
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    text = re.sub(r"[\s_-]+", "-", text)
    return text or "app"

def first_line_title(prompt_text: str) -> str:
    for line in prompt_text.splitlines():
        if line.strip():
            return line.strip()
    return "app"

def write_text(path: pathlib.Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

def write_log(line: str):
    DIST.mkdir(parents=True, exist_ok=True)
    prev = LOG_OUT.read_text(encoding="utf-8") if LOG_OUT.exists() else ""
    LOG_OUT.write_text(prev + line + "\n", encoding="utf-8")

def main():
    api_key = os.getenv("PPLX_API_KEY")
    if not api_key:
        print("Missing PPLX_API_KEY secret", file=sys.stderr)
        sys.exit(1)

    # Timestamps
    now = datetime.datetime.utcnow()
    utc_now = now.isoformat() + "Z"
    # Filename-safe timestamp
    ts = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Randomization seeds for prompt diversity
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
        )
        title = first_line_title(prompt_text)
        slug = slugify(title)
        # Versioned + latest prompt paths
        prompt_versioned = PROMPTS_DIR / f"{ts}_{slug}.txt"
        write_text(prompt_versioned, prompt_text.strip() + f"\n\n(Generated: {utc_now}, seed={seed}, theme={theme}, extras={extras})\n")
        write_text(LATEST_DIR / "prompt.txt", prompt_text.strip())
        print(f"Wrote {prompt_versioned}")
        write_log(f"[{utc_now}] Prompt OK  seed={seed}  theme={theme}  extras={extras}  title='{title}'  slug={slug}")
    except Exception as e:
        err = f"[{utc_now}] Prompt generation FAILED: {type(e).__name__}: {e}"
        print(err, file=sys.stderr)
        write_log(err)
        sys.exit(1)

    # ----- Stage 2: Build the single-file app from the prompt -----
    try:
        raw_html = pplx_chat(
            messages=[
                {"role": "system", "content": SYSTEM_BUILD},
                {"role": "user", "content": prompt_text},
            ],
            api_key=api_key,
            temperature=BUILD_TEMPERATURE,
        )
        html = extract_html(raw_html)
        # Versioned + latest app paths
        app_versioned = APPS_DIR / f"{ts}_{slug}.html"
        write_text(app_versioned, html)
        write_text(LATEST_DIR / "app.html", html)
        print(f"Wrote {app_versioned}")
        write_log(f"[{utc_now}] Build OK  file={app_versioned.name}")
    except Exception as e:
        err = f"[{utc_now}] Build FAILED: {type(e).__name__}: {e}"
        print(err, file=sys.stderr)
        write_log(err)
        sys.exit(1)

    print(f"Done. Versioned files:\n  - {app_versioned}\n  - {prompt_versioned}\nLatest copies:\n  - {LATEST_DIR/'app.html'}\n  - {LATEST_DIR/'prompt.txt'}")

if __name__ == "__main__":
    main()
