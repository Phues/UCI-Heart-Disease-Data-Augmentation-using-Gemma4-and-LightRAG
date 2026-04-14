import sys
import re
import urllib.request
import io

# Try to import optional dependencies
try:
    import pdfplumber
except ImportError:
    print("Installing pdfplumber...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "pdfplumber"], check=True)
    import pdfplumber

try:
    import ollama
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "ollama"], check=True)
    import ollama

# ── Sources ────────────────────────────────────────────────────────────────────
SOURCES = {
    "chol_bp": {
        "name": "Joint dependence of risk of coronary heart disease on serum cholesterol and systolic blood pressure: a discriminant function analysis",
        "url": "https://jhanley.biostat.mcgill.ca/c678/cornfield.pdf",
        "keywords": [
            "blood pressure", "hypertension", "systolic", "diastolic",
            "mmhg", "antihypertensive", "cardiovascular risk",
        ],
    },
    "who_guidelines": {
        "name": "CLINICAL GUIDELINES FOR THE MANAGEMENT OF CORONARY HEART DISEASE",
        "url": "https://extranet.who.int/ncdccs/Data/MUS_D1_chd.pdf",
        "keywords": [
            "blood pressure", "hypertension", "systolic", "diastolic",
            "mmhg", "antihypertensive", "cardiovascular risk",
        ],
    },

}

MODEL = "gemma4:e4b"

# ── Token counting ─────────────────────────────────────────────────────────────
def count_tokens_approx(text: str) -> int:
    """Rough approximation: 1 token ≈ 4 characters (standard for English text)."""
    return len(text) // 4

def get_model_context_length(model_name: str) -> int:
    """Query Ollama for the model's context window size."""
    try:
        info = ollama.show(model_name)
        # Try modelinfo first (newer Ollama versions)
        if hasattr(info, "modelinfo"):
            for key, val in info.modelinfo.items():
                if "context" in key.lower():
                    return int(val)
        # Fall back to parameters string
        if hasattr(info, "parameters") and info.parameters:
            for line in info.parameters.splitlines():
                if "num_ctx" in line.lower():
                    parts = line.split()
                    for p in parts:
                        if p.isdigit():
                            return int(p)
    except Exception as e:
        print(f"  Could not query model info: {e}")
    return 8192  # safe default for gemma4:e4b

# ── PDF fetching ───────────────────────────────────────────────────────────────
def fetch_pdf_bytes(url: str) -> bytes:
    print(f"  Downloading: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()

# ── Section extraction ─────────────────────────────────────────────────────────
def is_relevant(text: str, keywords: list[str]) -> bool:
    t = text.lower()
    return any(kw in t for kw in keywords)

def extract_relevant_sections(pdf_bytes: bytes, keywords: list[str]) -> str:
    sections = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        total_pages = len(pdf.pages)
        print(f"  PDF has {total_pages} pages — scanning for relevant sections...")

        buffer = []
        in_relevant = False

        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            text = re.sub(r"\n{3,}", "\n\n", text).strip()

            if not text:
                continue

            if is_relevant(text, keywords):
                in_relevant = True
                buffer.append(f"[Page {i+1}]\n{text}")
            else:
                if in_relevant and buffer:
                    # Keep a 1-page lookahead to avoid cutting mid-section
                    buffer.append(f"[Page {i+1} — context]\n{text}")
                in_relevant = False
                if buffer:
                    sections.append("\n\n".join(buffer))
                    buffer = []

        if buffer:
            sections.append("\n\n".join(buffer))

    return "\n\n" + ("=" * 60) + "\n\n".join(sections)

# ── Compatibility check ────────────────────────────────────────────────────────
def check_compatibility(text: str, model: str, source_name: str):
    ctx_len = get_model_context_length(model)
    tokens = count_tokens_approx(text)
    chars = len(text)

    print(f"\n{'='*60}")
    print(f"  Source  : {source_name}")
    print(f"  Model   : {model}")
    print(f"  Context : {ctx_len:,} tokens")
    print(f"  Chars   : {chars:,}")
    print(f"  ~Tokens : {tokens:,}")
    pct = (tokens / ctx_len) * 100
    print(f"  Usage   : {pct:.1f}% of context window")

    if tokens <= ctx_len * 0.75:
        print(f"  Status  : OK — fits comfortably (under 75%)")
    elif tokens <= ctx_len:
        print(f"  Status  : WARNING — fits but tight ({pct:.0f}%), consider trimming")
    else:
        over = tokens - ctx_len
        print(f"  Status  : EXCEEDS context by ~{over:,} tokens — must chunk or trim")
    print(f"{'='*60}\n")

    return tokens, ctx_len

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract relevant sections from clinical PDFs")
    parser.add_argument("--model", default=MODEL, help="Ollama model name")
    parser.add_argument("--sources", nargs="+", choices=list(SOURCES.keys()),
                        default=list(SOURCES.keys()), help="Which sources to process")
    parser.add_argument("--out-dir", default="rag/guidelines", help="Output directory")
    args = parser.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    for key in args.sources:
        src = SOURCES[key]
        print(f"\nProcessing: {src['name']}")
        print("-" * 60)

        try:
            pdf_bytes = fetch_pdf_bytes(src["url"])
            text = extract_relevant_sections(pdf_bytes, src["keywords"])

            if not text.strip():
                print("  No relevant sections found — try adding more keywords.")
                continue

            tokens, ctx = check_compatibility(text, args.model, src["name"])

            # If it exceeds context, truncate with a warning
            if tokens > ctx:
                print(f"  Truncating to fit context window...")
                max_chars = ctx * 4
                text = text[:max_chars] + "\n\n[TRUNCATED TO FIT CONTEXT WINDOW]"

            out_path = os.path.join(args.out_dir, f"{key}_extracted.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"Source: {src['name']}\nURL: {src['url']}\n\n")
                f.write(text)

            print(f"  Saved: {out_path} ({len(text):,} chars)")

        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nDone. Re-run rag_setup.py to ingest the new guidelines.")

if __name__ == "__main__":
    main()
