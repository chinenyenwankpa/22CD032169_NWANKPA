@app.route("/ask", methods=["GET", "POST"])
def ask():
    if request.method == "GET":
        return render_template("index.html",
                               processed=None,
                               raw_response=None,
                               answer=None,
                               original=None)

    q = request.form.get("question", "").strip()
    if not q:
        return render_template("index.html",
                               error="Please enter a question.",
                               processed=None,
                               raw_response=None,
                               answer=None,
                               original=None)

    provider = request.form.get("provider") or os.environ.get("PROVIDER", "openai")
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
