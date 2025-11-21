# LLM_QA_CLI.py
"""
LLM_QA_CLI.py
- Provides:
  - preprocess(question): basic preprocessing (lowercase, remove punctuation, tokenize)
  - ask_llm(question, provider="openai"): sends prompt to an LLM provider (default OpenAI)
  - CLI entrypoint: accept a question, show processed question, ask LLM, print final answer
Notes:
- Requires environment variable OPENAI_API_KEY for OpenAI.
- Optionally set PROVIDER=hf and HF_API_KEY to use HuggingFace Inference API.
"""
import os
import re
import sys
import json
import argparse
import requests

try:
    import openai
except ImportError:
    openai = None

# ---------- Preprocessing ----------
def preprocess(text: str) -> dict:
    """Return a dict with processed form:
       - original: original text
       - cleaned: lowercase + punctuation removed
       - tokens: split on whitespace
    """
    original = text.strip()
    lower = original.lower()
    cleaned = re.sub(r'[^0-9a-z\s]', '', lower)
    tokens = [t for t in cleaned.split() if t]
    return {"original": original, "cleaned": cleaned, "tokens": tokens}

# ---------- Prompt construction ----------
def make_prompt(preprocessed: dict) -> str:
    """Create a prompt to send to the LLM."""
    return (
        "You are a helpful assistant. Use the processed question below to produce a clear, concise answer.\n\n"
        f"Original question: {preprocessed['original']}\n"
        f"Processed (cleaned): {preprocessed['cleaned']}\n"
        f"Tokens: {preprocessed['tokens']}\n\n"
        "Answer the user's question directly. If you need to ask for clarification, ask a single concise question."
    )

# ---------- LLM callers ----------
def ask_openai(prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 300) -> str:
    """Ask OpenAI ChatCompletion (new API, requires OPENAI_API_KEY)."""
    if openai is None:
        raise RuntimeError("openai package not installed. Install with `pip install openai`.")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")
    openai.api_key = api_key

    resp = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.2
    )
    return resp.choices[0].message.content.strip()

def ask_huggingface(prompt: str, hf_token: str, model: str = "google/flan-t5-large") -> str:
    """Ask HuggingFace Inference API (text-generation or text2text models)."""
    if not hf_token:
        raise RuntimeError("HuggingFace API key required for provider 'hf' (set HF_API_KEY).")
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": prompt, "options": {"wait_for_model": True}}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"HuggingFace API error {r.status_code}: {r.text}")
    data = r.json()
    # Extract text depending on model response
    if isinstance(data, list) and "generated_text" in data[0]:
        return data[0]["generated_text"].strip()
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"].strip()
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"HuggingFace error: {data['error']}")
    return json.dumps(data)

def ask_llm(question: str, provider: str = None) -> dict:
    """
    Preprocess question, construct prompt, call LLM, return a dict:
    { processed, prompt, raw_response, answer }
    provider: None (default=OPENAI), or "openai", or "hf"
    """
    provider = (provider or os.environ.get("PROVIDER", "openai")).lower()
    pre = preprocess(question)
    prompt = make_prompt(pre)
    raw = ""
    try:
        if provider == "openai":
            raw = ask_openai(prompt)
        elif provider == "hf":
            hf_key = os.environ.get("HF_API_KEY")
            hf_model = os.environ.get("HF_MODEL", "google/flan-t5-large")
            raw = ask_huggingface(prompt, hf_key, model=hf_model)
        else:
            raise RuntimeError(f"Unsupported provider '{provider}'. Use 'openai' or 'hf'.")
    except Exception as e:
        raw = f"ERROR: {str(e)}"
    return {"processed": pre, "prompt": prompt, "raw_response": raw, "answer": raw}

# ---------- CLI entrypoint ----------
def main():
    parser = argparse.ArgumentParser(description="LLM Q&A CLI")
    parser.add_argument("-q", "--question", help="Question to ask. If omitted, interactive prompt will open.")
    parser.add_argument("--provider", help="LLM provider: openai (default) or hf", default=None)
    args = parser.parse_args()

    if args.question:
        question = args.question
    else:
        print("Enter your question (single line). Press Enter when done:")
        try:
            question = input("> ").strip()
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)

    if not question:
        print("No question provided. Exiting.")
        sys.exit(0)

    result = ask_llm(question, provider=args.provider)
    print("\n--- Results ---")
    print("Original question:", result["processed"]["original"])
    print("Processed (cleaned):", result["processed"]["cleaned"])
    print("Tokens:", result["processed"]["tokens"])
    print("Prompt sent to LLM (truncated):", result["prompt"][:1000])
    print("LLM response:", result["raw_response"])
    print("Final Answer:", result["answer"])

if __name__ == "__main__":
    main()
