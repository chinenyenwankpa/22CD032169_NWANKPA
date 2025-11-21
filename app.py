from flask import Flask, render_template, request
from LLM_QA_CLI import ask_llm  # reuse your CLI function
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    processed_question = None
    answer = None

    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if question:
            provider = os.environ.get("PROVIDER", "openai")
            result = ask_llm(question, provider=provider)
            processed_question = result.get("processed", {}).get("cleaned")
            answer = result.get("answer")
        else:
            answer = "Please enter a question."

    return render_template(
        "index.html",
        processed_question=processed_question,
        answer=answer
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
