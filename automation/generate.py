import os, json, pathlib, re, datetime, requests, sys, random

API_URL = "https://api.perplexity.ai/chat/completions"
MODEL = "sonar"
TIMEOUT = 120

OUT_DIR = pathlib.Path("dist")
PROMPT_OUT = OUT_DIR / "prompt.txt"
APP_OUT = OUT_DIR / "app.html"

# --- Tunables ---
PROMPT_TEMPERATURE = 0.8     # more creativity for prompt
BUILD_TEMPERATURE = 0.3      # more determinism for code

# Rotating “themes” and “features” for variety; the prompt-generator can use these.
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
"""

USER_PROMPT_GEN_TEMPLATE = """Generate a brand new single-file app specification.
Inspiration seed (date/time & randoms):
- utc_now: {utc}
- seed: {seed}
- theme hint: {theme}
- extra features to consider: {extras}

Constraints:
- Return ONLY the prompt (title then spec). No explanations.
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
- NO markdown fences. NO extra commentary.
"""

def pplx_chat(messages, api_key, temperature, timeout=TIMEOUT):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": MODEL, "temperature": temperature, "messages": messages}
    r = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

def extract_html(raw: str) -> str:
    # If code fences present, extract; else return as-is.
    fence = re.search(r"```(?:html)?\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE)
    html = fence.group(1) if fence else raw

    # Ensure <!DOCTYPE html>
    if "<!doctype html>" not in html.lower():
        # Add minimal wrapper if needed
        if "<html" not in html.lower():
            html = f"<!DOCTYPE html>\n<html><head><meta charset='utf-8'><title>App</title></head><body>\n{html}\n</body></html>"
        else:
            html = "<!DOCTYPE html>\n" + html

    # Remove external refs to enforce single-file
    html = re.sub(r'<script[^>]+src=["\'][^"\']+["\'][^>]*></script>', '', html, flags=re.IGNORECASE)
    html = re.sub(r'<link[^>]+rel=["\']stylesheet["\'][^>]*>', '', html, flags=re.IGNORECASE)

    # Stamp time for guaranteed diffs
    stamp = f"<!-- Auto-generated via Perplexity on {datetime.datetime.utcnow().isoformat()}Z -->"
    html = re.sub(r"(?i)<head>", f"<head>\n{stamp}\n", html, count=1)
    if stamp not in html:
        html = stamp + "\n" + html
    return html

def main():
    api_key = os.getenv("PPLX_API_KEY")
    if not api_key:
        print("Missing PPLX_API_KEY secret", file=sys.stderr)
        sys.exit(1)

    # ----- Stage 1: Generate the prompt -----
    utc_now = datetime.datetime.utcnow().isoformat() + "Z"
    seed = random.randint(10_000, 99_999)
    theme = random.choice(THEMES)
    extras = ", ".join(random.sample(FEATURE_EXTRAS, k=3))

    user_prompt_gen = USER_PROMPT_GEN_TEMPLATE.format(
        utc=utc_now, seed=seed, theme=theme, extras=extras
    )

    prompt_text = pplx_chat(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_GEN},
            {"role": "user", "content": user_prompt_gen}
        ],
        api_key=api_key,
        temperature=PROMPT_TEMPERATURE
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROMPT_OUT.write_text(prompt_text.strip() + f"\n\n(Generated: {utc_now}, seed={seed})\n", encoding="utf-8")

    # ----- Stage 2: Build the single-file app from the prompt -----
    raw_html = pplx_chat(
        messages=[
            {"role": "system", "content": SYSTEM_BUILD},
            {"role": "user", "content": prompt_text}
        ],
        api_key=api_key,
        temperature=BUILD_TEMPERATURE
    )
    html = extract_html(raw_html)
    APP_OUT.write_text(html, encoding="utf-8")

    print(f"Wrote {PROMPT_OUT} and {APP_OUT}")

if __name__ == "__main__":
    main()
