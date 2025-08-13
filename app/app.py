# app.py
import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from intent_classifier_optimized import clean  # reuse cleaner
from answers import get_answer  # your answer templates

# ---- Optional translator (no-op fallback if not installed)
try:
    from deep_translator import GoogleTranslator
except Exception:  # library missing or runtime issue
    GoogleTranslator = None

def translate(text, src="en", tgt="en"):
    """Translate text using deep_translator if available; otherwise return text."""
    if not text or src == tgt:
        return text
    if GoogleTranslator is None:
        return text
    try:
        return GoogleTranslator(source=src, target=tgt).translate(text)
    except Exception:
        return text

app = Flask(__name__, static_folder="static", static_url_path="/static")

# ---- Paths
MODEL_PATH       = "../models/intent_pipeline.pkl"
ENCODER_PATH     = "../models/label_encoder.pkl"
THRESHOLDS_PATH  = "../models/thresholds.pkl"
KB_INDEX_PATH    = "../models/kb_index.pkl"   # optional

# ---- Load model assets
pipe = joblib.load(MODEL_PATH)
le   = joblib.load(ENCODER_PATH)
thr  = joblib.load(THRESHOLDS_PATH)

# KB is optional: if missing, we’ll skip KB routing gracefully
kb = None
if os.path.exists(KB_INDEX_PATH):
    kb = joblib.load(KB_INDEX_PATH)

def kb_search(query, k=3):
    """Search knowledge base using TF-IDF similarity (if KB available)."""
    if kb is None:
        return []
    v = kb["vec"].transform([query])
    sim = (v @ kb["M"].T).toarray()[0]  # cosine proxy for L2-normalized TF-IDF
    idx = np.argsort(sim)[::-1][:k]
    return [(kb["meta"][i], float(sim[i])) for i in idx]

def answer_from_kb(hits):
    """Return formatted KB answer from top search hit (English)."""
    top = hits[0]
    return {
        "reply": f"This page should help: {top[0]['title']}",
        "links": [{"title": top[0]['title'], "url": top[0]['url']}],
    }

@app.route("/", methods=["GET"])
def home():
    # Serve the chat UI
    return send_from_directory(app.static_folder, "index.html")

@app.route("/health", methods=["GET"])
def health():
    return {"ok": True, "kb_loaded": bool(kb)}

@app.post("/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    msg = (payload.get("message") or "").strip()
    lang = (payload.get("lang") or "en").lower()  # "en" | "it" | "ti"
    if lang not in {"en", "it", "ti"}:
        lang = "en"

    if not msg:
        return jsonify({"reply": "Please type a message.", "meta": {"routed_to": "error"}}), 400

    # 1) Translate incoming user text to English for classification
    msg_en = translate(msg, src=lang, tgt="en")

    # 2) Classify (unchanged logic)
    x = clean(msg_en)
    probs = pipe.predict_proba([x])[0]
    ids = np.argsort(probs)
    t1, t2 = ids[-1], ids[-2]
    p1, p2 = float(probs[t1]), float(probs[t2])
    intent = le.inverse_transform([t1])[0]

    # Thresholding
    tau = thr.get("per_class_tau", {}).get(int(t1), 0.55)
    margin = thr.get("margin", 0.10)

    # Routing
    if p1 >= tau and (p1 - p2) >= margin:
        ans = get_answer(intent)                # English templates
        routed = "intent"
    else:
        hits = kb_search(x, k=3)
        if hits and hits[0][1] >= 0.25:
            ans = answer_from_kb(hits)         # English KB title
            routed = "kb"
        else:
            ans = {
                "reply": "Did you mean shipping, returns, or payments?",
                "suggestions": ["Shipping info", "Return policy", "Payment methods"],
                "links": [],
            }
            routed = "clarify"

    # 3) Translate response to user's chosen language (reply + suggestions + link titles)
    if lang != "en":
        ans["reply"] = translate(ans.get("reply", ""), src="en", tgt=lang)
        if "suggestions" in ans and isinstance(ans["suggestions"], list):
            ans["suggestions"] = [translate(s, src="en", tgt=lang) for s in ans["suggestions"]]
        if "links" in ans and isinstance(ans["links"], list):
            for link in ans["links"]:
                if "title" in link:
                    link["title"] = translate(link["title"], src="en", tgt=lang)

    meta = {"intent": intent, "confidence": p1, "routed_to": routed, "lang": lang}
    return jsonify({**ans, "meta": meta})

if __name__ == "__main__":
    # Single-command run: python app.py
    app.run(host="0.0.0.0", port=5000, debug=True)
# To run the app, use: python app.py
# Access it at: http://localhost:5000/