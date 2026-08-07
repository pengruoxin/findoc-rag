"""LLM-based query rewriting for out-of-vocabulary paraphrases.

This replaces the "add a synonym every time" maintenance loop with model-based
term normalization: the LLM rewrites user wording into filing wording. The
deterministic synonym table stays as a zero-cost fast path and as a fallback
when the LLM is unavailable, but it is no longer the primary mechanism.
"""

from __future__ import annotations

import os
import time
from typing import Protocol

import httpx

from findoc_rag.query_expansion import expand_query

REWRITE_SYSTEM_PROMPT = (
    "你是中文年报问答的查询改写助手。把用户的问题改写成年报里会使用的标准措辞，"
    "只做术语归一化：\n"
    "- 口语 / 简称 → 年报术语（例如 营收→营业收入、净利→净利润、"
    "毛利水平→毛利率、回报率→净资产收益率、赚了多少→营业收入）\n"
    "- 保留公司名、年份、指标、时间范围、排序与限额，不改变语义\n"
    "- 不要添加问题里没有的信息，不要回答原问题\n"
    "- 只输出一条改写后的查询，不要解释"
)


class QueryRewriter(Protocol):
    def rewrite(self, query: str) -> str: ...


class LLMQueryRewriter:
    """Rewrite queries with an OpenAI-compatible model, falling back safely."""

    def __init__(
        self,
        model: str = "",
        endpoint: str = "",
        api_key: str | None = None,
    ) -> None:
        self.model = model or os.getenv("FINDOC_RAG_ANSWER_MODEL", "deepseek-chat")
        self.endpoint = endpoint or os.getenv(
            "FINDOC_RAG_ANSWER_ENDPOINT", "https://api.deepseek.com/chat/completions"
        )
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "") or os.getenv(
            "OPENAI_API_KEY", ""
        )
        self._cache: dict[str, str] = {}

    def rewrite(self, query: str) -> str:
        """Return an LLM-rewritten query, or the deterministic expansion on failure."""
        cached = self._cache.get(query)
        if cached is not None:
            return cached
        rewritten = self._rewrite_remote(query)
        if rewritten is None:
            rewritten = expand_query(query)
        self._cache[query] = rewritten
        return rewritten

    def _rewrite_remote(self, query: str) -> str | None:
        if not self.api_key:
            return None
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        }
        for attempt in range(3):
            try:
                response = httpx.post(
                    self.endpoint,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                    timeout=httpx.Timeout(60.0, connect=20.0),
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"].strip()
                if not content or content == query:
                    return expand_query(query)
                return content
            except (httpx.TransportError, httpx.HTTPStatusError):
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
        return None
