import os
import re
import time
from decimal import Decimal, InvalidOperation
from typing import Literal, Protocol

import httpx
from pydantic import BaseModel, Field

from findoc_rag.documents.models import StructuredTableCell
from findoc_rag.indexing import SearchHit
from findoc_rag.provider_credentials import resolve_provider_api_key
from findoc_rag.table_extraction import (
    ExtractedCell,
    extract_annual_rows,
    extract_concentration,
    extract_note_cost,
    extract_quarterly,
    extract_segment,
)

MAX_GENERATION_CONTEXTS = 5
MAX_GENERATION_CONTEXT_CHARS = 1800
CITATION_ORDINAL_PATTERN = re.compile(r"\[(\d+)\]")

QUARTERLY_ROW_ALIASES = (
    ("经营活动现金流量净额", "经营活动产生的现金流量净额"),
    ("经营活动产生的现金流量净额", "经营活动产生的现金流量净额"),
    ("归母净利润", "归属于上市公司股东的净利润"),
    ("归属于上市公司股东的净利润", "归属于上市公司股东的净利润"),
    ("扣非净利润", "归属于上市公司股东的扣除非经常性损益后的净利润"),
    ("扣除非经常性损益后的净利润", "归属于上市公司股东的扣除非经常性损益后的净利润"),
    ("营业收入", "营业收入"),
)

NOTE_COST_DISPLAY = ("主营业务", "其他业务", "合计")

ABSTENTION_PATTERNS = (
    "无法回答",
    "无法确认",
    "无法确定",
    "无法给出",
    "无法判断",
    "无法核实",
    "无法提供",
    "无法准确回答",
    "无法准确判断",
    "不能回答",
    "不能确认",
    "不能确定",
    "不能给出",
    "证据不足",
    "信息不足",
    "没有足够信息",
    "未提供该数据",
    "无法从证据",
)


def _format_decimal(value: str) -> str:
    """Format a normalized decimal string with thousands separators."""
    try:
        return f"{Decimal(value):,.2f}"
    except InvalidOperation:
        return value


class Citation(BaseModel):
    ordinal: int
    chunk_id: str
    page_start: int
    page_end: int
    section_path: list[str]
    excerpt: str = ""


class ClaimCitation(BaseModel):
    """A machine-readable claim-to-evidence edge for Agent consumers."""

    claim: str
    citation_ordinals: list[int]


class GeneratedAnswer(BaseModel):
    answer: str
    citations: list[Citation]
    provider: str
    grounded: bool = True
    claim_citations: list[ClaimCitation] = Field(default_factory=list)


class AnswerGenerator(Protocol):
    def generate(self, query: str, hits: list[SearchHit]) -> GeneratedAnswer: ...


class GroundedAnswerGenerator:
    """Evidence-first answer generator with an optional OpenAI-compatible backend."""

    def __init__(
        self,
        model: str = "",
        endpoint: str = "",
        enabled: bool = False,
        api_key: str | None = None,
    ) -> None:
        self.enabled = enabled
        self.model = model or os.getenv("FINDOC_RAG_ANSWER_MODEL", "")
        self.endpoint = endpoint or os.getenv(
            "FINDOC_RAG_ANSWER_ENDPOINT", "https://api.deepseek.com/chat/completions"
        )
        self.api_key = resolve_provider_api_key(self.endpoint, api_key)

    @staticmethod
    def _citations(hits: list[SearchHit]) -> list[Citation]:
        return [
            Citation(
                ordinal=ordinal,
                chunk_id=h.chunk.chunk_id,
                page_start=h.chunk.page_start,
                page_end=h.chunk.page_end,
                section_path=h.chunk.section_path,
                excerpt=h.chunk.text[:MAX_GENERATION_CONTEXT_CHARS],
            )
            for ordinal, h in enumerate(hits[:MAX_GENERATION_CONTEXTS], start=1)
        ]

    @staticmethod
    def _referenced_citations(
        content: str, citations: list[Citation]
    ) -> list[Citation] | None:
        """Resolve every answer ordinal to the exact evidence item it names.

        ``None`` means the answer is not safely displayable: it either contains
        no citation or references an ordinal outside the supplied context.  A
        filtered list is returned so API callers never receive unrelated hits
        disguised as citations used by the answer.
        """
        ordinals = [int(value) for value in CITATION_ORDINAL_PATTERN.findall(content)]
        available = {citation.ordinal: citation for citation in citations}
        if not ordinals or any(ordinal not in available for ordinal in ordinals):
            return None
        return [available[ordinal] for ordinal in dict.fromkeys(ordinals)]

    @staticmethod
    def _claim_citations(content: str, citations: list[Citation]) -> list[ClaimCitation]:
        available = {citation.ordinal for citation in citations}
        claims: list[ClaimCitation] = []
        for raw in re.split(r"(?<=[。；;])", content):
            ordinals = [
                ordinal
                for ordinal in dict.fromkeys(
                    int(value) for value in CITATION_ORDINAL_PATTERN.findall(raw)
                )
                if ordinal in available
            ]
            claim = CITATION_ORDINAL_PATTERN.sub("", raw).strip(" ，,。；;")
            if claim and ordinals:
                claims.append(ClaimCitation(claim=claim, citation_ordinals=ordinals))
        return claims

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
        remote = bool(self.enabled and self.api_key and self.model)
        structured = (
            None
            if os.getenv("FINDOC_RAG_DISABLE_DETERMINISTIC_TABLES") == "1"
            else self._deterministic_table_answer(query, selected)
        )
        remote_deterministic = (
            os.getenv("FINDOC_RAG_REMOTE_DETERMINISTIC_TABLES") == "1"
        )
        if structured and (not remote or remote_deterministic):
            referenced = self._referenced_citations(structured, citations)
            if referenced is None:
                return GeneratedAnswer(
                    answer="当前结构化回答未提供可验证引用，系统拒绝展示。",
                    citations=[],
                    provider="guardrail-abstention",
                    grounded=False,
                )
            return GeneratedAnswer(
                answer=structured,
                citations=referenced,
                provider="deterministic-table",
                claim_citations=self._claim_citations(structured, referenced),
            )
        if remote:
            return self._generate_remote(query, selected, citations)
        return GeneratedAnswer(
            answer="已找到相关证据，请展开“查看证据”核验原文。",
            citations=citations,
            provider="evidence-only",
            grounded=False,
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
    def _is_abstention(content: str) -> bool:
        """Detect explicit refusal signals in a remote answer.

        Conservative by design: only clear refusal phrasing counts, so risk
        disclosures like "存在不确定性" are not mistaken for abstentions.
        """
        compact = re.sub(r"\s+", "", content)
        return any(pattern in compact for pattern in ABSTENTION_PATTERNS)

    @staticmethod
    def _extract_quarterly_metric(query: str, text: str) -> str | None:
        """Back-compatible structured quarterly output (used by legacy tests)."""
        if "季度" not in query:
            return None
        row = GroundedAnswerGenerator._quarterly_row(query)
        if row is None:
            return None
        cells = [cell for cell in extract_quarterly(text) if cell.row == row]
        if len(cells) != 4:
            return None
        return "\n".join(
            f"第{i + 1}季度：{_format_decimal(cell.value)}"
            for i, cell in enumerate(cells)
        )

    @staticmethod
    def _quarterly_row(query: str) -> str | None:
        return next(
            (row for alias, row in QUARTERLY_ROW_ALIASES if alias in query),
            None,
        )

    @staticmethod
    def _annual_table_text(hit: SearchHit) -> str | None:
        """Return the chunk text only when it looks like the annual table."""
        text = hit.chunk.text
        compact = re.sub(r"\s+", "", text)
        return text if "主要会计数据" in compact else None

    @staticmethod
    def _table_cells(
        hit: SearchHit,
        table_type: Literal[
            "quarterly", "note_cost", "segment", "annual_data", "concentration"
        ],
    ) -> list[StructuredTableCell | ExtractedCell]:
        """Prefer verified index sidecars and retain the legacy text fallback."""
        structured = [
            cell
            for table in getattr(hit.chunk, "structured_tables", [])
            if table.table_type == table_type
            for cell in table.cells
        ]
        if structured:
            return structured
        extractors = {
            "quarterly": extract_quarterly,
            "note_cost": extract_note_cost,
            "segment": extract_segment,
            "concentration": extract_concentration,
        }
        extractor = extractors.get(table_type)
        return extractor(hit.chunk.text) if extractor is not None else []

    @classmethod
    def _annual_value(
        cls, hit: SearchHit, row: str, year: int | None
    ) -> str | None:
        cells = cls._table_cells(hit, "annual_data")
        matching = [cell for cell in cells if cell.row == row]
        if matching:
            if year is not None:
                year_pattern = re.compile(rf"^{year}年(?:末)?$")
                value = next(
                    (cell.value for cell in matching if year_pattern.match(cell.column)),
                    None,
                )
                if value is not None:
                    return value
            return matching[0].value
        annual_text = cls._annual_table_text(hit)
        if annual_text is None:
            return None
        rows = [item for item in extract_annual_rows(annual_text) if item.label == row]
        if not rows:
            return None
        return rows[0].value_for_year(year) if year is not None else rows[0].values[0]

    @staticmethod
    def _quarterly_table_answer(query: str, hits: list[SearchHit]) -> str | None:
        row = GroundedAnswerGenerator._quarterly_row(query)
        if row is None:
            return None
        query_company = GroundedAnswerGenerator._query_company(query)
        query_year = GroundedAnswerGenerator._query_year(query)
        quarterly: tuple[int, list[str]] | None = None
        annual: tuple[int, str] | None = None
        for index, hit in enumerate(hits, start=1):
            if not GroundedAnswerGenerator._hit_matches_identity(
                hit, company=query_company, year=query_year
            ):
                continue
            cells = [cell for cell in GroundedAnswerGenerator._table_cells(
                hit, "quarterly"
            ) if cell.row == row]
            if len(cells) == 4 and quarterly is None:
                quarterly = (index, [cell.value for cell in cells])
            if annual is None:
                annual_year = query_year or hit.chunk.report_year
                annual_value = GroundedAnswerGenerator._annual_value(
                    hit, row, annual_year
                )
                if annual_value is not None:
                    annual = (index, annual_value)
        if quarterly is None:
            return None
        values = [_format_decimal(value) for value in quarterly[1]]
        quote = (
            "第一至第四季度分别为"
            + "、".join(f"{value}元" for value in values)
            + f"[{quarterly[0]}]"
        )
        if "合计" in query or "全年" in query:
            if annual is None:
                return None
            try:
                total = sum(Decimal(value) for value in quarterly[1])
            except InvalidOperation:
                return None
            annual_value = Decimal(annual[1])
            relation = "一致" if total == annual_value else (
                f"不一致，差额{_format_decimal(f'{total - annual_value}')}元"
            )
            quote += (
                f"；合计{_format_decimal(f'{total}')}元，"
                f"与全年披露值{_format_decimal(annual[1])}元{relation}[{annual[0]}]"
            )
        return quote

    @staticmethod
    def _query_company(query: str) -> str | None:
        companies = GroundedAnswerGenerator._query_companies(query)
        return companies[0] if len(companies) == 1 else None

    @staticmethod
    def _query_companies(query: str) -> list[str]:
        companies = []
        if any(alias in query for alias in ("贵州茅台", "茅台", "600519")):
            companies.append("贵州茅台")
        if any(alias in query for alias in ("伊利股份", "伊利", "600887")):
            companies.append("伊利股份")
        return companies

    @staticmethod
    def _query_year(query: str) -> int | None:
        match = re.search(r"20\d{2}", query)
        return int(match.group()) if match else None

    @staticmethod
    def _hit_company(hit: SearchHit) -> str:
        company = hit.chunk.company_name or ""
        compact = re.sub(r"\s+", "", hit.chunk.text)
        if company:
            return company
        if "贵州茅台" in compact:
            return "贵州茅台"
        if "伊利股份" in compact or "伊利集团" in compact:
            return "伊利股份"
        return ""

    @staticmethod
    def _hit_matches_identity(
        hit: SearchHit, *, company: str | None, year: int | None
    ) -> bool:
        if company and GroundedAnswerGenerator._hit_company(hit) not in {"", company}:
            return False
        return not (year and hit.chunk.report_year not in {None, year})

    @staticmethod
    def _note_cost_answer(query: str, hits: list[SearchHit]) -> str | None:
        want_cost = "成本" in query
        want_revenue = "收入" in query and "成本" not in query
        if not (want_cost or want_revenue):
            return None
        query_company = GroundedAnswerGenerator._query_company(query)
        query_year = GroundedAnswerGenerator._query_year(query)
        requested_scope = "parent" if "母公司" in query else "consolidated"
        for index, hit in enumerate(hits, start=1):
            if not GroundedAnswerGenerator._hit_matches_identity(
                hit, company=query_company, year=query_year
            ):
                continue
            hit_scope = GroundedAnswerGenerator._statement_scope(hit)
            if hit_scope in {"consolidated", "parent"} and hit_scope != requested_scope:
                continue
            cells = GroundedAnswerGenerator._table_cells(hit, "note_cost")
            if not cells:
                continue
            column = "本期成本" if want_cost else "本期收入"
            values = {
                cell.row: cell.value for cell in cells if cell.column == column
            }
            if not all(row in values for row in NOTE_COST_DISPLAY):
                continue
            if "差额" in query or "不同" in query or "为什么" in query:
                main = Decimal(values["主营业务"])
                total = Decimal(values["合计"])
                diff = total - main
                label = "营业成本" if want_cost else "营业收入"
                return (
                    f"主营业务{label[-2:]}为{_format_decimal(values['主营业务'])}元，"
                    f"{label}合计为{_format_decimal(values['合计'])}元，"
                    f"差额{_format_decimal(f'{diff}')}元，来自其他业务{label[-2:]}[{index}]"
                )
            return (
                f"主营业务成本{_format_decimal(values['主营业务'])}元，"
                f"其他业务成本{_format_decimal(values['其他业务'])}元，"
                f"营业成本合计{_format_decimal(values['合计'])}元[{index}]"
            )
        return None

    @classmethod
    def _deducted_profit_answer(
        cls, query: str, hits: list[SearchHit]
    ) -> str | None:
        if not (
            "归母净利润" in query
            and "扣非" in query
            and "非经常性损益" in query
        ):
            return None
        query_company = cls._query_company(query)
        query_year = cls._query_year(query)
        annual: tuple[int, Decimal, Decimal] | None = None
        deducted_rows = (
            "归属于上市公司股东的扣除非经常性损益的净利润",
            "归属于上市公司股东的扣除非经常性损益后的净利润",
        )
        for index, hit in enumerate(hits, start=1):
            if not cls._hit_matches_identity(
                hit, company=query_company, year=query_year
            ):
                continue
            requested_year = query_year or hit.chunk.report_year
            reported = cls._annual_value(
                hit, "归属于上市公司股东的净利润", requested_year
            )
            deducted = next(
                (
                    value
                    for row in deducted_rows
                    if (value := cls._annual_value(hit, row, requested_year))
                    is not None
                ),
                None,
            )
            if reported is not None and deducted is not None:
                annual = (index, Decimal(reported), Decimal(deducted))
                break
        if annual is None:
            return None
        annual_index, reported, deducted = annual
        nonrecurring = reported - deducted
        formatted_nonrecurring = _format_decimal(f"{nonrecurring}")
        evidence_index = next(
            (
                index
                for index, hit in enumerate(hits, start=1)
                if formatted_nonrecurring in re.sub(r"\s+", "", hit.chunk.text)
            ),
            None,
        )
        if evidence_index is None:
            return None
        difference = abs(nonrecurring)
        higher = "归母口径" if nonrecurring >= 0 else "扣非口径"
        return (
            f"归母净利润为{_format_decimal(f'{reported}')}元，"
            f"扣非归母净利润为{_format_decimal(f'{deducted}')}元[{annual_index}]；"
            f"{higher}高{_format_decimal(f'{difference}')}元，"
            f"与非经常性损益合计{formatted_nonrecurring}元的方向和金额一致"
            f"[{evidence_index}]"
        )

    @staticmethod
    def _forecast_commitment_answer(
        query: str, hits: list[SearchHit]
    ) -> str | None:
        compact_query = re.sub(r"\s+", "", query)
        if not any(cue in compact_query for cue in ("保证", "承诺")):
            return None
        for index, hit in enumerate(hits, start=1):
            compact = re.sub(r"\s+", "", hit.chunk.text)
            if not (
                "不构成对投资者的业绩承诺" in compact
                or "不构成业绩承诺" in compact
            ):
                continue
            target = re.search(
                r"计划实现营业总收入(?P<value>\d[\d,]*(?:\.\d+)?)亿元",
                compact,
            )
            if target is None:
                continue
            year = re.search(r"20\d{2}", compact)
            year_text = f"{year.group()}年" if year else ""
            return (
                f"不能保证。{year_text}营业总收入{target.group('value')}亿元"
                "是经营计划目标，受未来经营环境影响存在不确定性，"
                f"并不构成对投资者的业绩承诺[{index}]"
            )
        return None

    @staticmethod
    def _segment_answer(query: str, hits: list[SearchHit]) -> str | None:
        if "毛利率" not in query:
            return None
        query_company = GroundedAnswerGenerator._query_company(query)
        query_year = GroundedAnswerGenerator._query_year(query)
        for index, hit in enumerate(hits, start=1):
            if not GroundedAnswerGenerator._hit_matches_identity(
                hit, company=query_company, year=query_year
            ):
                continue
            cells = GroundedAnswerGenerator._table_cells(hit, "segment")
            if not cells:
                continue
            if "直销" in query or "批发代理" in query or "销售模式" in query:
                section = "主营业务分销售模式情况"
            else:
                section = "主营业务分产品情况"
            margins = [
                (cell.row, cell.value)
                for cell in cells
                if cell.section == section and cell.column == "毛利率"
            ]
            if not margins:
                continue
            rows: list[tuple[str, str]] = []
            for row, value in margins:
                if row not in {existing for existing, _ in rows}:
                    rows.append((row, value))
            if "直销" in query or "批发代理" in query or "销售模式" in query:
                direct = dict(rows).get("直销")
                wholesale = dict(rows).get("批发代理")
                if direct is None or wholesale is None:
                    continue
                diff = Decimal(direct) - Decimal(wholesale)
                higher = "直销" if diff > 0 else "批发代理"
                return (
                    f"直销毛利率{_format_decimal(direct)}%，"
                    f"批发代理{_format_decimal(wholesale)}%；"
                    f"{higher}高{_format_decimal(f'{abs(diff)}')}个百分点[{index}]"
                )
            joined = "、".join(
                f"{row}{_format_decimal(value)}%" for row, value in rows
            )
            highest = max(rows, key=lambda item: Decimal(item[1]))
            return f"{joined}；其中{highest[0]}最高[{index}]"
        return None

    @staticmethod
    def _annual_revenue_answer(query: str, hits: list[SearchHit]) -> str | None:
        if "营业收入" not in query:
            return None
        query_year = GroundedAnswerGenerator._query_year(query)
        company_values: dict[str, tuple[int, str, str | None]] = {}
        for index, hit in enumerate(hits, start=1):
            if not GroundedAnswerGenerator._hit_matches_identity(
                hit, company=None, year=query_year
            ):
                continue
            company = GroundedAnswerGenerator._hit_company(hit)
            if company not in ("贵州茅台", "伊利股份"):
                continue
            requested_year = query_year or hit.chunk.report_year
            value = GroundedAnswerGenerator._annual_value(
                hit, "营业收入", requested_year
            )
            if value is None:
                continue
            annual_text = GroundedAnswerGenerator._annual_table_text(hit)
            text_rows = (
                [row for row in extract_annual_rows(annual_text) if row.label == "营业收入"]
                if annual_text is not None
                else []
            )
            company_values[company] = (
                index,
                value,
                text_rows[0].yoy if text_rows else None,
            )
        query_companies = GroundedAnswerGenerator._query_companies(query)
        if set(query_companies) == {"贵州茅台", "伊利股份"}:
            if "贵州茅台" not in company_values or "伊利股份" not in company_values:
                return None
            moutai_index, moutai_value, _ = company_values["贵州茅台"]
            yili_index, yili_value, _ = company_values["伊利股份"]
            diff = Decimal(moutai_value) - Decimal(yili_value)
            higher = "贵州茅台" if diff > 0 else "伊利股份"
            return (
                f"贵州茅台为{_format_decimal(moutai_value)}元[{moutai_index}]，"
                f"伊利股份为{_format_decimal(yili_value)}元[{yili_index}]；"
                f"{higher}高{_format_decimal(f'{abs(diff)}')}元"
            )
        query_company = GroundedAnswerGenerator._query_company(query)
        if query_company is not None:
            selected = company_values.get(query_company)
            if selected is None:
                return None
            company_values = {query_company: selected}
        for company, (index, value, yoy) in company_values.items():
            year = query_year or 2024
            parts = [f"{company}{year}年营业收入为{_format_decimal(value)}元"]
            if yoy is not None and ("同比" in query or "增幅" in query):
                parts.append(f"同比增长{_format_decimal(yoy)}%")
            return "，".join(parts) + f"[{index}]"
        return None

    @staticmethod
    def _consolidated_parent_revenue_answer(
        query: str, hits: list[SearchHit]
    ) -> str | None:
        if "合并" not in query or "母公司" not in query or "营业收入" not in query:
            return None
        query_company = GroundedAnswerGenerator._query_company(query)
        query_year = GroundedAnswerGenerator._query_year(query)
        values: dict[str, tuple[int, Decimal]] = {}
        for index, hit in enumerate(hits, start=1):
            if not GroundedAnswerGenerator._hit_matches_identity(
                hit, company=query_company, year=query_year
            ):
                continue
            cells = GroundedAnswerGenerator._table_cells(hit, "note_cost")
            # Text fallback can hallucinate a four-column note table from an
            # unrelated debt table.  This scope-sensitive answer therefore
            # accepts only the coordinate-verified sidecar representation.
            has_verified_note_cost = any(
                table.table_type == "note_cost"
                for table in getattr(hit.chunk, "structured_tables", [])
            )
            text_declares_yuan = bool(
                re.search(r"单位\s*[：:]\s*元(?![万亿])", hit.chunk.text)
            )
            if not has_verified_note_cost and not text_declares_yuan:
                continue
            total = next(
                (
                    cell.value
                    for cell in cells
                    if cell.row == "合计" and cell.column == "本期收入"
                ),
                None,
            )
            if total is not None:
                scope = GroundedAnswerGenerator._statement_scope(hit)
                if scope and scope not in values:
                    values[scope] = (index, Decimal(total))
        if "consolidated" not in values or "parent" not in values:
            return None
        consolidated = values["consolidated"]
        parent = values["parent"]
        diff = consolidated[1] - parent[1]
        higher = "合并口径" if diff >= 0 else "母公司口径"
        return (
            f"合并口径营业收入为{_format_decimal(f'{consolidated[1]}')}元"
            f"[{consolidated[0]}]，母公司口径为{_format_decimal(f'{parent[1]}')}元"
            f"[{parent[0]}]；{higher}高{_format_decimal(f'{abs(diff)}')}元"
        )

    @staticmethod
    def _statement_scope(hit: SearchHit) -> Literal["consolidated", "parent"] | None:
        explicit = getattr(hit.chunk, "statement_scope", None)
        if explicit in {"consolidated", "parent"}:
            return explicit
        context = normalize_scope_text(" ".join(hit.chunk.section_path) + "\n" + hit.chunk.text)
        if "母公司财务报表主要项目注释" in context or "母公司利润表" in context:
            return "parent"
        if "合并财务报表项目注释" in context or "合并利润表" in context:
            return "consolidated"
        return None

    @staticmethod
    def _concentration_answer(query: str, hits: list[SearchHit]) -> str | None:
        compact = re.sub(r"\s+", "", query)
        want_customer = "客户" in compact and (
            "占比" in compact or "集中度" in compact
        )
        want_supplier = "供应商" in compact and (
            "占比" in compact or "集中度" in compact
        )
        if not (want_customer or want_supplier):
            return None
        by_company: dict[str, tuple[int, str, str]] = {}
        query_year = GroundedAnswerGenerator._query_year(query)
        for index, hit in enumerate(hits, start=1):
            if not GroundedAnswerGenerator._hit_matches_identity(
                hit, company=None, year=query_year
            ):
                continue
            cells = GroundedAnswerGenerator._table_cells(hit, "concentration")
            values = {cell.column: cell.value for cell in cells}
            if "销售占比(%)" not in values or "采购占比(%)" not in values:
                continue
            company = hit.chunk.company_name or ""
            if not company:
                text = re.sub(r"\s+", "", hit.chunk.text)
                company = (
                    "贵州茅台"
                    if "贵州茅台" in text
                    else "伊利股份"
                    if "伊利股份" in text
                    else company
                )
            if company:
                by_company[company] = (
                    index,
                    values["销售占比(%)"],
                    values["采购占比(%)"],
                )
        if "贵州茅台" in compact and "伊利股份" in compact:
            if "贵州茅台" not in by_company or "伊利股份" not in by_company:
                return None
            moutai_index, moutai_customer, moutai_supplier = by_company["贵州茅台"]
            yili_index, yili_customer, yili_supplier = by_company["伊利股份"]
            if want_customer:
                diff = Decimal(moutai_customer) - Decimal(yili_customer)
                higher = "贵州茅台" if diff > 0 else "伊利股份"
                first = _format_decimal(moutai_customer)
                second = _format_decimal(yili_customer)
            else:
                diff = Decimal(moutai_supplier) - Decimal(yili_supplier)
                higher = "贵州茅台" if diff > 0 else "伊利股份"
                first = _format_decimal(moutai_supplier)
                second = _format_decimal(yili_supplier)
            return (
                f"贵州茅台为{first}%[{moutai_index}]，"
                f"伊利股份为{second}%[{yili_index}]；"
                f"{higher}高{_format_decimal(f'{abs(diff)}')}个百分点"
            )
        query_company = (
            "贵州茅台"
            if "贵州茅台" in compact
            else "伊利股份"
            if "伊利股份" in compact
            else None
        )
        if query_company is not None:
            if query_company not in by_company:
                return None
            index, customer, supplier = by_company[query_company]
            parts = []
            if want_customer:
                parts.append(f"前五名客户销售占比为{_format_decimal(customer)}%")
            if want_supplier:
                parts.append(
                    f"前五名供应商采购占比为{_format_decimal(supplier)}%"
                )
            return "，".join(parts) + f"[{index}]"
        for company, (index, customer, supplier) in by_company.items():
            parts = []
            if want_customer:
                parts.append(f"前五名客户销售占比为{_format_decimal(customer)}%")
            if want_supplier:
                parts.append(
                    f"前五名供应商采购占比为{_format_decimal(supplier)}%"
                )
            return "，".join(parts) + f"[{index}]"
        return None

    @classmethod
    def _deterministic_table_answer(
        cls, query: str, hits: list[SearchHit]
    ) -> str | None:
        """Return a citation-formatted deterministic answer from table cells."""
        compact = re.sub(r"\s+", "", query)
        forecast_commitment = cls._forecast_commitment_answer(query, hits)
        if forecast_commitment is not None:
            return forecast_commitment
        deducted_profit = cls._deducted_profit_answer(query, hits)
        if deducted_profit is not None:
            return deducted_profit
        if "季度" in compact:
            return cls._quarterly_table_answer(query, hits)
        if "合并" in compact and "母公司" in compact and "营业收入" in compact:
            return cls._consolidated_parent_revenue_answer(query, hits)
        if "附注" in compact and "成本" in compact:
            return cls._note_cost_answer(query, hits)
        if "成本" in compact and (
            "差额" in compact or "为什么" in compact or "不同" in compact
        ):
            return cls._note_cost_answer(query, hits)
        if "毛利率" in compact:
            return cls._segment_answer(query, hits)
        if "营业收入" in compact:
            return cls._annual_revenue_answer(query, hits)
        if (
            "前五名客户" in compact
            or "前五名供应商" in compact
            or "客户集中度" in compact
            or "供应商集中度" in compact
            or ("客户" in compact and "占比" in compact)
            or ("供应商" in compact and "占比" in compact)
        ):
            return cls._concentration_answer(query, hits)
        return None

    def _generate_remote(self, query: str, hits: list[SearchHit], citations: list[Citation]) -> GeneratedAnswer:
        context = "\n\n".join(
            f"[{i + 1}] {hit.chunk.text[:MAX_GENERATION_CONTEXT_CHARS]}"
            for i, hit in enumerate(hits)
        )
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "你是严谨的财务报告问答助手。只能依据证据回答，并使用[1]、[2]格式引用；证据不足时明确拒答。"},
                {"role": "user", "content": f"问题：{query}\n\n证据：\n{context}"},
            ],
        }
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = httpx.post(
                    self.endpoint,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                    timeout=httpx.Timeout(120.0, connect=30.0),
                )
                response.raise_for_status()
                break
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
        else:
            raise last_error  # type: ignore[misc]
        content = response.json()["choices"][0]["message"]["content"].strip()
        if self._is_abstention(content):
            referenced = self._referenced_citations(content, citations)
            return GeneratedAnswer(
                answer=content,
                citations=referenced or [],
                provider="remote-abstention",
                grounded=False,
                claim_citations=self._claim_citations(content, referenced or []),
            )
        referenced = self._referenced_citations(content, citations)
        if referenced is None:
            return GeneratedAnswer(
                answer="当前回答包含缺失或越界引用，系统拒绝展示。",
                citations=[],
                provider="guardrail-abstention",
                grounded=False,
            )
        return GeneratedAnswer(
            answer=content,
            citations=referenced,
            provider="openai-compatible",
            claim_citations=self._claim_citations(content, referenced),
        )


def normalize_scope_text(text: str) -> str:
    return re.sub(r"\s+", "", text)
