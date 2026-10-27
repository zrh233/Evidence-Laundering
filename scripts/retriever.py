"""Deterministic BM25 retrieval over local corpus pages."""
import json
import math
import re
from collections import Counter
from pathlib import Path

_TOK = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return _TOK.findall(text.lower())


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        # docs: list of {"doc_id","title","body",...}; index on title+body
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.doc_ids = [d["doc_id"] for d in docs]
        self.toks = [tokenize(d["title"] + " \n " + d["body"]) for d in docs]
        self.dl = [len(t) for t in self.toks]
        self.avgdl = (sum(self.dl) / len(self.dl)) if self.dl else 0.0
        self.tf = [Counter(t) for t in self.toks]
        df = Counter()
        for t in self.toks:
            for w in set(t):
                df[w] += 1
        N = len(docs)
        self.idf = {
            w: math.log(1 + (N - n + 0.5) / (n + 0.5)) for w, n in df.items()
        }

    def search(self, query, top_k=5):
        q = tokenize(query)
        scores = []
        for i, _ in enumerate(self.docs):
            tf = self.tf[i]
            dl = self.dl[i]
            s = 0.0
            for w in q:
                if w not in tf:
                    continue
                idf = self.idf[w]
                num = tf[w] * (self.k1 + 1)
                den = tf[w] + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += idf * num / den
            scores.append((s, i))
        scores.sort(key=lambda x: (-x[0], x[1]))  # deterministic tie-break by index
        out = []
        for s, i in scores[:top_k]:
            d = self.docs[i]
            body = d["body"]
            out.append(
                {
                    "doc_id": d["doc_id"],
                    "title": d["title"],
                    "outlet": d.get("outlet", ""),
                    "snippet": body[:240].replace("\n", " "),
                    "score": round(s, 4),
                }
            )
        return out


def load_corpus(corpus_dir):
    corpus_dir = Path(corpus_dir)
    docs = []
    for p in sorted(corpus_dir.glob("*.json")):
        if p.name.startswith("."):  # skip macOS AppleDouble ._* sidecars etc.
            continue
        docs.append(json.loads(p.read_text(encoding="utf-8")))
    return docs
