"""Tests for LLM query rewriting with safe fallback."""

from __future__ import annotations

from findoc_rag.query_rewriting import LLMQueryRewriter


class _FakeRewriter(LLMQueryRewriter):
    """No-network subclass for deterministic tests."""

    def __init__(self, mapping: dict[str, str]) -> None:
        super().__init__(api_key="")
        self.mapping = mapping

    def _rewrite_remote(self, query: str) -> str | None:
        return self.mapping.get(query)


def test_llm_rewrite_used_when_available() -> None:
    rewriter = _FakeRewriter({"去年赚了多少": "2024年营业收入是多少"})
    assert rewriter.rewrite("去年赚了多少") == "2024年营业收入是多少"


def test_falls_back_to_deterministic_expansion_when_llm_unavailable() -> None:
    rewriter = _FakeRewriter({})
    assert rewriter.rewrite("去年赚了多少") == "去年赚了多少"


def test_rewrites_alias_when_llm_returns_nothing() -> None:
    rewriter = _FakeRewriter({"茅台酒毛利水平": None})
    assert rewriter.rewrite("茅台酒毛利水平") == "茅台酒毛利率"


def test_cache_prevents_duplicate_calls() -> None:
    calls: list[str] = []

    class CountingRewriter(LLMQueryRewriter):
        def _rewrite_remote(self, query: str) -> str | None:
            calls.append(query)
            return query + "改写"

    rewriter = CountingRewriter(api_key="")
    rewriter.rewrite("问题")
    rewriter.rewrite("问题")
    assert calls == ["问题"]
