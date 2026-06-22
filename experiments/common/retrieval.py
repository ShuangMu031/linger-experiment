"""Dependency-free BM25 retriever for the A2 (RAG) baseline.

Pure-Python BM25 over character unigrams + bigrams (no jieba/rank_bm25; works on
Chinese with no segmenter). A2 stores every past user message and injects the
top-k hits — the honest "what if the bare LLM just had a memory lookup?" control.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import List, Tuple


def _tokenize(text: str) -> List[str]:
    chars = [c for c in (text or "") if not c.isspace()]
    bigrams = [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
    return chars + bigrams


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._docs: List[Tuple[str, List[str]]] = []
        self._doc_freq: Counter = Counter()
        self.k1 = k1
        self.b = b

    def add(self, text: str) -> None:
        tokens = _tokenize(text)
        if not tokens:
            return
        self._docs.append((text, tokens))
        for term in set(tokens):
            self._doc_freq[term] += 1

    def search(self, query: str, top_k: int = 3) -> List[str]:
        if not self._docs:
            return []
        n_docs = len(self._docs)
        avg_len = sum(len(tokens) for _, tokens in self._docs) / n_docs
        query_terms = _tokenize(query)
        scored: List[Tuple[float, str]] = []
        for text, tokens in self._docs:
            term_freq = Counter(tokens)
            doc_len = len(tokens)
            score = 0.0
            for term in query_terms:
                if term not in term_freq:
                    continue
                idf = math.log(1 + (n_docs - self._doc_freq[term] + 0.5) / (self._doc_freq[term] + 0.5))
                tf = term_freq[term]
                norm = tf + self.k1 * (1 - self.b + self.b * doc_len / avg_len)
                score += idf * tf * (self.k1 + 1) / norm
            scored.append((score, text))
        scored.sort(key=lambda pair: -pair[0])
        return [text for score, text in scored[:top_k] if score > 0]


if __name__ == "__main__":
    r = BM25Retriever()
    for s in ["我养了只橘猫叫团子", "我老家是西安的", "我最爱吃螺蛳粉", "今天好困"]:
        r.add(s)
    print("query cat ->", r.search("你那只猫叫啥来着", top_k=2))
    print("BM25 self-test done")
