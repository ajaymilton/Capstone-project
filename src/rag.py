"""Part D tasks 1-3: chunking, TF-IDF retrieval and the grounded answer() call. The refund agent reuses retrieve()."""
import glob
import os
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mock_llm  # noqa: E402

POLICY_DIR = "data/policies"
CHUNK_WORDS = 40

SYSTEM_PROMPT = ("You are a POLICY ASSISTANT for the support desk. Answer the question using only the context supplied in "
                 "the user message. If the context does not contain the answer, say that you could not find the "
                 "information in the policy documents.")


def chunk_text(text, size=CHUNK_WORDS, overlap=10):
    words = text.split()
    step = size - overlap
    chunks, start = [], 0
    while True:
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):          # this chunk reaches (or passes) the last word
            break
        start += step
    return chunks


class PolicyIndex:
    def __init__(self, overlap=10, policy_dir=POLICY_DIR):
        self.chunks = []
        for path in sorted(glob.glob(os.path.join(policy_dir, "*.txt"))):
            with open(path, encoding="utf-8") as f:
                text = f.read()
            for ch in chunk_text(text, CHUNK_WORDS, overlap):
                self.chunks.append({"source": os.path.basename(path), "text": ch})
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        sims = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        top = sims.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": round(float(sims[i]), 4), "text": self.chunks[i]["text"]} for i in top]


DEFAULT_INDEX = PolicyIndex(overlap=10)


def retrieve(query, k=3, index=None):
    return (index or DEFAULT_INDEX).retrieve(query, k)


def answer(question, index=None):
    hits = retrieve(question, 3, index)
    context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"}]
    resp = mock_llm.chat(messages)
    return resp.choices[0].message.content, [h["source"] for h in hits]
