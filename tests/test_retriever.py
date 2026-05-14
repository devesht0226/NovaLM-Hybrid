from src.models.retriever import TfidfRetriever


def test_retriever_returns_top_k():
    r = TfidfRetriever({"min_context_score": None, "stop_words": None})
    r.fit(["alpha beta gamma delta", "epsilon zeta eta theta"])
    hits = r.top_k("beta gamma", k=1)
    assert len(hits) == 1
    assert "beta" in hits[0].text.lower()


def test_min_context_score_returns_empty_when_no_good_match():
    r = TfidfRetriever({"min_context_score": 0.99})
    r.fit(["alpha beta gamma delta", "epsilon zeta eta theta"])
    assert r.top_k("completely unrelated query xyz", k=2) == []


def test_min_context_score_none_always_returns_k():
    r = TfidfRetriever({"min_context_score": None})
    r.fit(["aa bb cc", "dd ee ff"])
    out = r.top_k("qqq zzz", k=2)
    assert len(out) == 2
