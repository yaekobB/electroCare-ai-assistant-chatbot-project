# intent_classifier_optimized.py
import argparse, json, os, re, joblib, numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import FeatureUnion
import numpy as np

SEED = 42
TRAIN_PATH = "../data/intents_optimized.json"
TEST_PATH = "../data/test_data_optimized.json"
MODEL_PATH = "../models/intent_pipeline.pkl"
ENCODER_PATH = "../models/label_encoder.pkl"
THRESH_PATH = "../models/thresholds.pkl"


# ---------------- Text cleaning ----------------
_url = re.compile(r'https?://\S+|www\.\S+')
_email = re.compile(r'\S+@\S+')
_ids = re.compile(r'\b([a-z]{2}\d{6,}|[A-Z]{2}\d{6,}|\d{8,})\b')
_ws = re.compile(r'\s+')

def clean(t: str) -> str:
    t = t.lower()
    t = _url.sub(" ", t)
    t = _email.sub(" ", t)
    t = _ids.sub(" ", t)
    return _ws.sub(" ", t).strip()

# ---------------- Light augmentation (optional) ----------------
SYN = {"refund": ["return", "money back"], "delivery": ["shipping", "shipment"], "support": ["help", "assistance"]}
abc = "abcdefghijklmnopqrstuvwxyz"

def aug_syn(t: str) -> str:
    for k, vs in SYN.items():
        if k in t and np.random.rand() < 0.4:
            t = t.replace(k, np.random.choice(vs))
    return t

def aug_typos(t: str, p=0.03) -> str:
    out = []
    for c in t:
        if c.isalpha() and np.random.rand() < p:
            out.append(np.random.choice(list(abc)))
        else:
            out.append(c)
    return "".join(out)

def augment(t: str) -> str:
    return aug_typos(aug_syn(t))

# ---------------- Data I/O ----------------
def load_xy(path: str, do_augment=False) -> Tuple[List[str], List[str]]:
    data = json.load(open(path, "r", encoding="utf-8"))
    X, y = [], []
    for item in data:
        txt = clean(item["text"])
        X.append(txt); y.append(item["intent"])
        # train-only: add light variants
        if do_augment:
            X.append(clean(augment(item["text"])))
            y.append(item["intent"])
    return X, y

# ---------------- Build model ----------------
def build_pipeline() -> Pipeline:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english", sublinear_tf=True)
    base = LinearSVC(C=1.0, class_weight="balanced")
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=5)  # gives predict_proba
    return Pipeline([("tfidf", vec), ("clf", clf)])

# ---------------- Train ----------------
def train(tune=False, augment_train=True):
    X_all, y_all = load_xy(TRAIN_PATH, do_augment=augment_train)
    le = LabelEncoder()
    y_enc = le.fit_transform(y_all)

    Xtr, Xva, ytr, yva = train_test_split(X_all, y_enc, test_size=0.2, stratify=y_enc, random_state=SEED)

    pipe = build_pipeline()
    if tune:
        grid = {
            "tfidf__ngram_range": [(1,1), (1,2)],
            "tfidf__min_df": [1, 2, 3],
            "clf__estimator__C": [0.5, 1.0, 2.0]        # <-- correct key for sklearn 1.x
}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=0)
        gs.fit(Xtr, ytr)
        pipe = gs.best_estimator_
        print(f"Best params: {gs.best_params_}")

    pipe.fit(Xtr, ytr)

    # Validation report
    yhat = pipe.predict(Xva)
    print("\n🧪 Validation on held-out split:")
    print(classification_report(yva, yhat, target_names=le.classes_, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):\n", confusion_matrix(yva, yhat))

    # ---- Per-class thresholds from validation (10th percentile of correct probs; floor=0.45) ----
    proba = pipe.predict_proba(Xva)
    correct_probs: Dict[int, List[float]] = defaultdict(list)
    for i, (yt, yp) in enumerate(zip(yva, yhat)):
        if yt == yp:
            correct_probs[yp].append(proba[i][yp])

    per_class_tau = {}
    for cls in np.unique(y_enc):
        vals = correct_probs.get(cls, [])
        tau = 0.55 if not vals else max(0.45, float(np.percentile(vals, 10)))
        per_class_tau[int(cls)] = tau
    thresholds = {"per_class_tau": per_class_tau, "margin": 0.15}
    joblib.dump(thresholds, THRESH_PATH)

    # Save artifacts
    joblib.dump(pipe, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"\n💾 Saved: {MODEL_PATH}, {ENCODER_PATH}, {THRESH_PATH}")

# ---------------- Evaluate on external test ----------------
def eval_external():
    import os, json
    from collections import Counter
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    if not os.path.exists(TEST_PATH):
        print("⚠️ test_data.json not found. Skipping external evaluation.")
        return

    pipe = joblib.load(MODEL_PATH)
    le   = joblib.load(ENCODER_PATH)

    # Load external test (no augmentation)
    X, y = load_xy(TEST_PATH, do_augment=False)

    # Optional: map legacy labels -> current schema
    alias = {
        "product_condition_info": "faq_product_condition",
        "product_quality_info": "faq_product_quality",
        "order_confirmation": "faq_order_confirmation",
        "price_negotiation": "faq_price_negotiation",
        # add more mappings here if you spot them
    }
    y = [alias.get(lbl, lbl) for lbl in y]

    # Warn & drop labels that were never seen in training
    known = set(le.classes_)
    unknown = sorted(set(y) - known)
    if unknown:
        print("⚠️ Unknown intents in test (not in training):", unknown)
        print("   These examples will be excluded from metrics.\n")

    keep = [i for i, lbl in enumerate(y) if lbl in known]
    if not keep:
        print("⚠️ After filtering unknown labels, no examples remain to evaluate.")
        return

    Xk = [X[i] for i in keep]
    yk = [y[i] for i in keep]

    # Predict (keep labels as strings)
    yhat = pipe.predict(Xk)
    
    # If predictions are numeric encodings, convert to string labels
    if np.issubdtype(np.array(yhat).dtype, np.number):
        yhat = le.inverse_transform(yhat)


    # Build report over labels that actually appear
    present_labels = sorted(set(yk) | set(yhat))
    print("\n🧪 Evaluation on external test_data.json (known intents only):")
    print(classification_report(
        yk, yhat,
        labels=present_labels, target_names=present_labels, zero_division=0
    ))

    acc = accuracy_score(yk, yhat)
    print(f"Accuracy: {acc:.3f}")

    cm = confusion_matrix(yk, yhat, labels=present_labels)
    print("\nConfusion matrix (rows=true, cols=pred, present labels only):")
    print(cm)

    # Show a few mismatches to inspect
    shown = 0
    for txt, yt, yp in zip(Xk, yk, yhat):
        if yt != yp:
            print(f"❌ '{txt}'")
            print(f"   Expected: {yt} | Predicted: {yp}\n")
            shown += 1
            if shown >= 10:
                break


# ---------------- Predict (with abstain) ----------------
def predict(text: str) -> Dict:
    pipe: Pipeline = joblib.load(MODEL_PATH)
    le: LabelEncoder = joblib.load(ENCODER_PATH)
    th = joblib.load(THRESH_PATH)

    probs = pipe.predict_proba([clean(text)])[0]
    idxs = np.argsort(probs)
    top1, top2 = idxs[-1], idxs[-2] if len(idxs) > 1 else idxs[-1]
    p1, p2 = float(probs[top1]), float(probs[top2])

    intent = le.inverse_transform([top1])[0]
    tau = th["per_class_tau"].get(int(top1), 0.55)
    if p1 < tau or (p1 - p2) < th["margin"]:
        return {"intent": "fallback", "confidence": p1, "top2": [[intent, p1], [le.inverse_transform([top2])[0], p2]]}
    return {"intent": intent, "confidence": p1, "top2": [[intent, p1], [le.inverse_transform([top2])[0], p2]]}

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="TechNest Intent Classifier (optimized TF-IDF + LinearSVC)")
    ap.add_argument("--train", action="store_true", help="Train and save artifacts")
    ap.add_argument("--tune", action="store_true", help="Grid search before training")
    ap.add_argument("--no-aug", action="store_true", help="Disable light augmentation during training")
    ap.add_argument("--eval", action="store_true", help="Evaluate on external test_data.json")
    ap.add_argument("--chat", action="store_true", help="Interactive mode")
    args = ap.parse_args()

    if args.train:
        train(tune=args.tune, augment_train=not args.no_aug)
    if args.eval:
        eval_external()
    if args.chat:
        print("\n🤖 TechNest NLP — type 'exit' to quit\n")
        while True:
            msg = input("You: ").strip()
            if msg.lower() == "exit":
                break
            res = predict(msg)
            print(f"→ intent: {res['intent']} | conf: {res['confidence']:.2f} | top2: {res['top2']}\n")

if __name__ == "__main__":
    main()
