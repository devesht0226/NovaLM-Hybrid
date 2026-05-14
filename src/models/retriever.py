from dataclasses import dataclass
from typing import Any, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievalResult:
    text: str
    score: float


def _as_tuple2(val: Any) -> tuple[int, int]:
    if isinstance(val, (list, tuple)) and len(val) == 2:
        return int(val[0]), int(val[1])
    return 1, 2


def _retrieval_defaults() -> dict[str, Any]:
    return {
        "max_features": 50000,
        "ngram_range": (1, 2),
        "min_df": 1,
        "max_df": 0.98,
        "sublinear_tf": True,
        "strip_accents": "unicode",
        "stop_words": None,
        "min_context_score": None,
    }


class TfidfRetriever:
    """TF-IDF chunk retriever with optional minimum-score gating for hybrid quality."""

    def __init__(self, retrieval_config: dict[str, Any] | None = None):
        rc = {**_retrieval_defaults(), **(retrieval_config or {})}
        self.min_context_score: float | None = rc.get("min_context_score")
        if self.min_context_score is not None:
            self.min_context_score = float(self.min_context_score)
        ng = _as_tuple2(rc.get("ngram_range", (1, 2)))
        sw = rc.get("stop_words")
        st = rc.get("strip_accents")
        if st == "" or st is False:
            st = None
        self.vectorizer = TfidfVectorizer(
            max_features=int(rc.get("max_features", 50000)),
            ngram_range=ng,
            min_df=rc.get("min_df", 1),
            max_df=rc.get("max_df", 0.98),
            sublinear_tf=bool(rc.get("sublinear_tf", True)),
            strip_accents=st,
            stop_words=sw,
        )
        self.corpus_chunks: List[str] = []
        self.corpus_matrix = None

    def fit(self, chunks: List[str]):
        self.corpus_chunks = chunks
        self.corpus_matrix = self.vectorizer.fit_transform(chunks)

    def top_k(self, query: str, k: int = 3) -> List[RetrievalResult]:
        if self.corpus_matrix is None or not self.corpus_chunks:
            return []
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.corpus_matrix).flatten()
        mx = float(sims.max()) if sims.size else 0.0
        if self.min_context_score is not None and mx < self.min_context_score:
            return []
        top_idx = sims.argsort()[::-1][:k]
        return [RetrievalResult(text=self.corpus_chunks[i], score=float(sims[i])) for i in top_idx]
