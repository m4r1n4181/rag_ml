"""Keyword search (BM25) with a small normalizer for Serbian."""
import re
import unicodedata

# Common case/number endings, longest first. Stripping one of them is a crude stemmer,
# but it puts "model", "modela", "modelom" on the same token.
SUFFIXES = ("ijama", "ama", "ima", "om", "em", "og", "ih", "im", "a", "e", "i", "u", "o")
MIN_STEM = 4
MAX_STEM = 6


def strip_diacritics(text: str) -> str:
    text = text.replace("đ", "dj").replace("Đ", "Dj")
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def stem(word: str) -> str:
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= MIN_STEM:
            word = word[: -len(suffix)]
            break
    return word[:MAX_STEM]


def tokenize(text: str) -> list[str]:
    words = re.findall(r"\w+", strip_diacritics(text.lower()))
    return [stem(w) for w in words if len(w) > 1]


class BM25Index:
    def __init__(self, ids: list[str], documents: list[str], metadatas: list[dict]):
        from rank_bm25 import BM25Okapi

        self.ids = ids
        self.documents = documents
        self.metadatas = metadatas
        self.bm25 = BM25Okapi([tokenize(d) for d in documents])

    def search(self, query: str, n: int) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [
            {"id": self.ids[i], "content": self.documents[i], "metadata": self.metadatas[i], "bm25": float(scores[i])}
            for i in top
        ]
