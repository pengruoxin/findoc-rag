import os
import re
from typing import Protocol

import httpx
from pydantic import BaseModel

from findoc_rag.indexing import SearchHit

MAX_GENERATION_CONTEXTS = 5


class Citation(BaseModel):
    chunk_id: str
    page_start: int
    page_end: int
    section_path: list[str]
    excerpt: str = ""


class GeneratedAnswer(BaseModel):
    answer: str
    citations: list[Citation]
    provider: str
    grounded: bool = True


class AnswerGenerator(Protocol):
    def generate(self, query: str, hits: list[SearchHit]) -> GeneratedAnswer: ...


class GroundedAnswerGenerator:
    """Evidence-first answer generator with an optional OpenAI-compatible backend."""

    def __init__(self, model: str = "", endpoint: str = "", enabled: bool = False) -> None:
        self.enabled = enabled
        self.model = model or os.getenv("FINDOC_RAG_ANSWER_MODEL", "")
        self.endpoint = endpoint or os.getenv(
            "FINDOC_RAG_ANSWER_ENDPOINT", "https://api.deepseek.com/chat/completions"
        )
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")

    @staticmethod
    def _citations(hits: list[SearchHit]) -> list[Citation]:
        return [
            Citation(
                chunk_id=h.chunk.chunk_id,
                page_start=h.chunk.page_start,
                page_end=h.chunk.page_end,
                section_path=h.chunk.section_path,
                excerpt=h.chunk.text[:1200],
            )
            for h in hits[:MAX_GENERATION_CONTEXTS]
        ]

    def generate(self, query: str, hits: list[SearchHit]) -> GeneratedAnswer:
        clarification = self._clarification_prompt(query)
        if clarification:
            return GeneratedAnswer(
                answer=clarification,
                citations=[],
                provider="clarification",
                grounded=False,
            )
        selected = hits[:MAX_GENERATION_CONTEXTS]
        citations = self._citations(selected)
        if not selected:
            return GeneratedAnswer(answer="当前证据不足，无法可靠回答。", citations=[], provider="abstention", grounded=False)
        if not self._evidence_supports_query(query, selected):
            return GeneratedAnswer(
                answer="当前检索证据与问题中的公司、年份或指标不一致，系统拒绝生成答案。",
                citations=citations,
                provider="guardrail-abstention",
                grounded=False,
            )
        structured = self._extract_quarterly_metric(query, selected[0].chunk.text)
        if structured and not (self.enabled and self.api_key and self.model):
            return GeneratedAnswer(answer=structured, citations=citations, provider="deterministic-table")
        if self.enabled and self.api_key and self.model:
            return self._generate_remote(query, selected, citations)
        return GeneratedAnswer(
            answer="已找到相关证据，请展开“查看证据”核验原文。",
            citations=citations,
            provider="extractive",
        )

    @staticmethod
    def _clarification_prompt(query: str) -> str | None:
        compact = re.sub(r"\s+", "", query)
        if "成本" in compact and not any(
            scope in compact
            for scope in (
                "营业成本",
                "主营业务成本",
                "其他业务成本",
                "销售成本",
                "研发成本",
                "分产品成本",
                "酒类成本",
            )
        ):
            return "请明确成本口径，例如合并或母公司营业成本、主营业务成本、分产品成本或期间费用。"
        if "利润" in compact and not any(
            scope in compact
            for scope in ("净利润", "扣非", "利润总额", "营业利润", "毛利润", "毛利率")
        ):
            return "请明确报告年份及利润口径，例如归母净利润、扣非归母净利润、营业利润或利润总额。"
        return None

    @staticmethod
    def _evidence_supports_query(query: str, hits: list[SearchHit]) -> bool:
        evidence = "\n".join(hit.chunk.text for hit in hits)
        compact_evidence = re.sub(r"\s+", "", evidence)
        for company in ("贵州茅台", "伊利股份"):
            if company in query and not any(
                company == hit.chunk.company_name
                or company in re.sub(r"\s+", "", hit.chunk.text)
                for hit in hits
            ):
                return False
        year_match = re.search(r"20\d{2}", query)
        if year_match:
            year = int(year_match.group())
            if not any(hit.chunk.report_year == year or str(year) in hit.chunk.text for hit in hits):
                return False
        metric_groups = (
            ("营业收入", ("营业收入", "营业总收入")),
            ("营业成本", ("营业成本",)),
            ("现金流", ("现金流", "经营活动产生的现金流量净额")),
            ("净利润", ("净利润",)),
        )
        for query_term, evidence_terms in metric_groups:
            if query_term in query and not any(term in compact_evidence for term in evidence_terms):
                return False
        return True

    @staticmethod
    def _extract_quarterly_metric(query: str, text: str) -> str | None:
        if "季度" not in query:
            return None
        metric = None
        compact_text = re.sub(r"\s+", "", text)
        for query_metric, text_metrics in (
            ("经营活动现金流量净额", ("经营活动现金流量净额", "经营活动产生的现金流量净额")),
            ("归属于上市公司股东的净利润", ("归属于上市公司股东的净利润",)),
        ):
            if query_metric in query:
                metric = next((item for item in text_metrics if item in compact_text), None)
            if metric:
                break
        if metric is None:
            return None
        metric_pattern = r"\s*".join(re.escape(character) for character in metric)
        metric_match = re.search(metric_pattern, text)
        if metric_match is None:
            return None
        values = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text[metric_match.end() :])
        if len(values) < 4:
            return None
        return "\n".join(f"第{i}季度：{value}" for i, value in enumerate(values[:4], 1))

    def _generate_remote(self, query: str, hits: list[SearchHit], citations: list[Citation]) -> GeneratedAnswer:
        context = "\n\n".join(f"[{i + 1}] {hit.chunk.text[:1800]}" for i, hit in enumerate(hits))
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "你是严谨的财务报告问答助手。只能依据证据回答，并使用[1]、[2]格式引用；证据不足时明确拒答。"},
                {"role": "user", "content": f"问题：{query}\n\n证据：\n{context}"},
            ],
        }
        response = httpx.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        valid_citation = rf"\[(?:[1-{MAX_GENERATION_CONTEXTS}])\]"
        if not re.search(valid_citation, content):
            return GeneratedAnswer(answer="当前回答未提供可验证引用，系统拒绝展示。", citations=citations, provider="guardrail-abstention", grounded=False)
        return GeneratedAnswer(answer=content, citations=citations, provider="openai-compatible")
