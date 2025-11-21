# app.py
from flask import Flask, render_template, request, jsonify
from LLM_QA_CLI import preprocess, make_prompt, ask_llm  # reuse functions
import os

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", processed=None, raw_response=None, answer=None, original=None)

@app.route("/ask", methods=["POST"])
def ask():
    q = request.form.get("question", "").strip()
    if not q:
        return render_template("index.html", error="Please enter a question.", processed=None, raw_response=None, answer=None, original=None)
    # provider can come from env or form (for demo)
    provider = request.form.get("provider") or os.environ.get("PROVIDER", "openai")
    # get result using shared ask_llm
    result = ask_llm(q, provider=provider)
    processed = result.get("processed")
    raw = result.get("raw_response")
    answer = result.get("answer")
    return render_template("index.html",
                           original=processed.get("original"),
                           processed=processed,
                           raw_response=raw,
                           answer=answer,
                           provider=provider)

# simple JSON endpoint (for debug)
@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.json or {}
    q = data.get("question", "")
    provider = data.get("provider") or os.environ.get("PROVIDER", "openai")
    if not q:
        return jsonify({"error": "No question provided."}), 400
    res = ask_llm(q, provider=provider)
    return jsonify(res)

if __name__ == "__main__":
    # For local development
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
