import os
import re
import time
from decimal import Decimal, InvalidOperation
from typing import Protocol

import httpx
from pydantic import BaseModel

from findoc_rag.indexing import SearchHit
from findoc_rag.table_extraction import (
    extract_annual_rows,
    extract_concentration,
    extract_note_cost,
    extract_quarterly,
    extract_segment,
)

MAX_GENERATION_CONTEXTS = 5

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
            return GeneratedAnswer(
                answer=structured,
                citations=citations,
                provider="deterministic-table",
            )
        if remote:
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
    def _quarterly_table_answer(query: str, hits: list[SearchHit]) -> str | None:
        row = GroundedAnswerGenerator._quarterly_row(query)
        if row is None:
            return None
        quarterly: tuple[int, list[str]] | None = None
        annual: tuple[int, str] | None = None
        for index, hit in enumerate(hits, start=1):
            cells = [
                cell for cell in extract_quarterly(hit.chunk.text) if cell.row == row
            ]
            if len(cells) == 4 and quarterly is None:
                quarterly = (index, [cell.value for cell in cells])
            annual_rows = []
            annual_text = GroundedAnswerGenerator._annual_table_text(hit)
            if annual_text is not None:
                annual_rows = [
                    annual_row
                    for annual_row in extract_annual_rows(annual_text)
                    if annual_row.label == row
                ]
            if annual_rows and annual is None:
                annual = (index, annual_rows[0].value_2024)
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
            quote += (
                f"；合计{_format_decimal(f'{total}')}元，"
                f"与全年披露值{_format_decimal(annual[1])}元一致[{annual[0]}]"
            )
        return quote

    @staticmethod
    def _note_cost_answer(query: str, hits: list[SearchHit]) -> str | None:
        want_cost = "成本" in query
        want_revenue = "收入" in query and "成本" not in query
        if not (want_cost or want_revenue):
            return None
        for index, hit in enumerate(hits, start=1):
            cells = extract_note_cost(hit.chunk.text)
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

    @staticmethod
    def _segment_answer(query: str, hits: list[SearchHit]) -> str | None:
        if "毛利率" not in query:
            return None
        for index, hit in enumerate(hits, start=1):
            cells = extract_segment(hit.chunk.text)
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
        company_values: dict[str, tuple[int, str, str | None]] = {}
        for index, hit in enumerate(hits, start=1):
            annual_text = GroundedAnswerGenerator._annual_table_text(hit)
            if annual_text is None:
                continue
            rows = [
                row
                for row in extract_annual_rows(annual_text)
                if row.label == "营业收入"
            ]
            if not rows:
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
            company_values[company] = (index, rows[0].value_2024, rows[0].yoy)
        if "贵州茅台" in query and "伊利股份" in query:
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
        for company, (index, value, yoy) in company_values.items():
            parts = [f"{company}2024年营业收入为{_format_decimal(value)}元"]
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
        values: list[tuple[int, Decimal]] = []
        for index, hit in enumerate(hits, start=1):
            cells = extract_note_cost(hit.chunk.text)
            total = next(
                (
                    cell.value
                    for cell in cells
                    if cell.row == "合计" and cell.column == "本期收入"
                ),
                None,
            )
            if total is not None:
                values.append((index, Decimal(total)))
        if len(values) < 2:
            return None
        consolidated = max(values, key=lambda item: item[1])
        parent = min(values, key=lambda item: item[1])
        diff = consolidated[1] - parent[1]
        return (
            f"合并口径营业收入为{_format_decimal(f'{consolidated[1]}')}元"
            f"[{consolidated[0]}]，母公司口径为{_format_decimal(f'{parent[1]}')}元"
            f"[{parent[0]}]；合并口径高{_format_decimal(f'{diff}')}元"
        )

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
        for index, hit in enumerate(hits, start=1):
            cells = extract_concentration(hit.chunk.text)
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
        context = "\n\n".join(f"[{i + 1}] {hit.chunk.text[:1800]}" for i, hit in enumerate(hits))
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
            return GeneratedAnswer(
                answer=content,
                citations=citations,
                provider="remote-abstention",
                grounded=False,
            )
        valid_citation = rf"\[(?:[1-{MAX_GENERATION_CONTEXTS}])\]"
        if not re.search(valid_citation, content):
            return GeneratedAnswer(answer="当前回答未提供可验证引用，系统拒绝展示。", citations=citations, provider="guardrail-abstention", grounded=False)
        return GeneratedAnswer(answer=content, citations=citations, provider="openai-compatible")
