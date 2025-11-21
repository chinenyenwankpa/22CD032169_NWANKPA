from flask import Flask, render_template, request
import os

# Create the Flask app
app = Flask(__name__)

# ---------- Routes ----------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/ask", methods=["GET", "POST"])
def ask():
    if request.method == "POST":
        # your logic here
        return "Message received"
    return render_template("ask.html")

# ---------- Run app ----------
if __name__ == "__main__":
    app.run(debug=True)
