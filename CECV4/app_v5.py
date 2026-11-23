import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory

# ============================================================
# PATHS & ENVIRONMENT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
SPYDER_DIR = BASE_DIR / "spyder"
TXT_DIR = SPYDER_DIR / "CM_COMBINED_TEXT"
DB_PATH = SPYDER_DIR / "dental_fts.sqlite"

load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Allow importing kb_search.py from ./spyder
if str(SPYDER_DIR) not in sys.path:
    sys.path.insert(0, str(SPYDER_DIR))

from kb_search import (  # noqa: E402
    build_fts_index,
    format_two_stage_context,
    search_kb_two_stage,
)

# ============================================================
# FLASK APP SETUP
# ============================================================

app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")


def ensure_index_ready():
    """Build the SQLite FTS5 index from CM_COMBINED_TEXT if it does not exist yet."""
    if not DB_PATH.exists():
        if not TXT_DIR.exists():
            raise FileNotFoundError(f"Text directory not found: {TXT_DIR}")
        build_fts_index(txt_dir=TXT_DIR, db_path=DB_PATH)


def ask_groq(user_query: str, context_text: str) -> str:
    """Send the retrieved chunks and user question to Groq and return the answer."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")

    system_prompt = (
        "You are a helpful, accurate educational counselling assistant specializing in "
        "BDS, MDS, dental colleges, NEET UG, and NEET MDS admissions. "
        "Answer the user's question using ONLY the provided reference documents below. "
        "Keep facts attributed to the specific college or document they come from—do not "
        "blend fees, seat counts, or rules across different colleges unless comparing them clearly. "
        "If a specific detail (such as a cutoff, seat count, or college) is not present in the "
        "provided context, state clearly what is missing instead of guessing."
    )

    user_prompt = (
        f"User Question:\n{user_query}\n\n"
        f"Retrieved Reference Context:\n{context_text if context_text.strip() else '[No matching sections found]'}"
    )

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_completion_tokens": 1024,
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=45)
    response.raise_for_status()
    data = response.json()

    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("Groq returned an empty response.")

    return choices[0].get("message", {}).get("content", "").strip()


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def serve_index():
    """Serve index.html from the parent folder."""
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """Receive a query from index.html, run two-stage search, and query Groq."""
    payload = request.get_json(silent=True) or {}
    query = (payload.get("query") or "").strip()

    if not query:
        return jsonify({"error": "Please enter a query."}), 400

    try:
        ensure_index_ready()

        selected_files, chunks = search_kb_two_stage(
            query=query,
            db_path=DB_PATH,
            top_files=4,
            top_k_per_file=3,
            max_total_chunks=10,
        )

        context_text = format_two_stage_context(chunks, max_chars=14000)
        answer = ask_groq(user_query=query, context_text=context_text)

        return jsonify({
            "answer": answer,
            "selected_files": [
                {
                    "page_title": f["page_title"],
                    "filename": f["filename"],
                    "matched_chunks": f["matched_chunks"],
                }
                for f in selected_files
            ],
            "chunks_used": len(chunks),
            "model": GROQ_MODEL,
        })

    except requests.HTTPError as http_err:
        status_code = http_err.response.status_code if http_err.response is not None else 502
        detail = http_err.response.text if http_err.response is not None else str(http_err)
        return jsonify({"error": f"Groq API error ({status_code}): {detail}"}), 502

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/reindex", methods=["POST"])
def reindex():
    """Optional endpoint to rebuild dental_fts.sqlite from CM_COMBINED_TEXT."""
    try:
        build_fts_index(txt_dir=TXT_DIR, db_path=DB_PATH)
        return jsonify({"status": "ok", "message": "Index rebuilt successfully."})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    ensure_index_ready()
    app.run(host="127.0.0.1", port=5000, debug=True)