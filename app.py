# app.py
from flask import Flask, render_template, request, jsonify
from LLM_QA_CLI import preprocess, make_prompt, ask_llm  # reuse CLI functions
import os

app = Flask(__name__)

# ---------------- Home Page ----------------
@app.route("/", methods=["GET"])
def index():
    # Show empty form initially
    return render_template(
        "index.html",
        processed=None,
        raw_response=None,
        answer=None,
        original=None,
        error=None,
        provider=os.environ.get("PROVIDER", "openai")
    )

# ---------------- Ask Route ----------------
@app.route("/ask", methods=["GET", "POST"])
def ask():
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if not question:
            return render_template(
                "index.html",
                error="Please enter a question.",
                processed=None,
                raw_response=None,
                answer=None,
                original=None,
                provider=os.environ.get("PROVIDER", "openai")
            )

        provider = request.form.get("provider") or os.environ.get("PROVIDER", "openai")
        # Call shared CLI function
        result = ask_llm(question, provider=provider)
        processed = result.get("processed", {})
        return render_template(
            "index.html",
            original=processed.get("original"),
            processed=processed,
            raw_response=result.get("raw_response"),
            answer=result.get("answer"),
            error=None,
            provider=provider
        )
    # If GET request → show empty form
    return render_template(
        "index.html",
        processed=None,
        raw_response=None,
        answer=None,
        original=None,
        error=None,
        provider=os.environ.get("PROVIDER", "openai")
    )

# ---------------- JSON API Endpoint ----------------
@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.json or {}
    question = data.get("question", "").strip()
    provider = data.get("provider") or os.environ.get("PROVIDER", "openai")
    if not question:
        return jsonify({"error": "No question provided."}), 400
    result = ask_llm(question, provider=provider)
    return jsonify(result)

# ---------------- Run Server ----------------
if __name__ == "__main__":
    # For local development
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
