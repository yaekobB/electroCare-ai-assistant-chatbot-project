# kb_build.py
import json, joblib
from sklearn.feature_extraction.text import TfidfVectorizer

kb = json.load(open("../data/kb_articles.json","r",encoding="utf-8"))

docs, meta = [], []
for a in kb:
    docs.append(f"{a['title']} {a.get('summary','')}")
    meta.append({"id":a["id"],"title":a["title"],"url":a["url"]})
    for i, faq in enumerate(a.get("faqs",[])):
        docs.append(f"FAQ: {faq['q']} {faq['a']}")
        meta.append({"id":f"{a['id']}#faq{i}","title":a["title"],"url":a["url"]})

vec = TfidfVectorizer(ngram_range=(1,2), stop_words="english", min_df=1, sublinear_tf=True)
M = vec.fit_transform(docs)
joblib.dump({"vec": vec, "M": M, "meta": meta}, "../models/kb_index.pkl")
print("Saved kb_index.pkl with", len(docs), "chunks")
