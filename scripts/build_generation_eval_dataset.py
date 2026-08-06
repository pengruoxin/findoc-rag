"""Build the assistant-curated FinDocRAG generation regression benchmark."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from findoc_rag.generation_evaluation import (
    AnnotationProvenance,
    AnswerContract,
    ExpectedFact,
    GenerationEvaluationDataset,
    GenerationEvaluationItem,
    GoldEvidence,
    HardNegative,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/evaluation/generation-eval-v1.json"
VERSION_DIRS = {
    "7961508deeffb5e66ae88808": ROOT / "data/catalog/versions/7961508deeffb5e66ae88808/chunks.jsonl",
    "e96cf669106c99e4e283ca45": ROOT / "data/catalog/versions/e96cf669106c99e4e283ca45/chunks.jsonl",
}
COMPANIES = {
    "moutai": ("600519", ["贵州茅台"], ["贵州茅台", "茅台", "Kweichow Moutai"]),
    "yili": ("600887", ["伊利股份"], ["伊利股份", "伊利", "Inner Mongolia Yili"]),
    "both": ("600519+600887", ["贵州茅台", "伊利股份"], ["贵州茅台", "茅台", "伊利股份", "伊利"]),
}
VISUALLY_REVIEWED = {
    ("7961508deeffb5e66ae88808", page)
    for page in (5, 6, 7, 8, 9, 10, 11, 22, 55, 56, 108)
} | {
    ("e96cf669106c99e4e283ca45", page)
    for page in (2, 7, 8, 9, 11, 18, 19, 20, 22, 23, 33, 34, 45, 82, 83, 84, 123, 124, 206, 267)
}


def load_chunks() -> dict[str, tuple[dict, str]]:
    chunks: dict[str, tuple[dict, str]] = {}
    for version_id, path in VERSION_DIRS.items():
        for line in path.read_text(encoding="utf-8").splitlines():
            chunk = json.loads(line)
            chunks[chunk["chunk_id"]] = (chunk, version_id)
    return chunks


def fact(
    fact_id: str,
    description: str,
    subject: str,
    predicate: str,
    value: str,
    value_type: str,
    period: str,
    scope: str,
    chunks: list[str],
    *,
    unit: str | None = None,
    currency: str | None = None,
    derivation: str | None = None,
    tolerance: str = "0",
    acceptable_values: list[str] | None = None,
) -> ExpectedFact:
    return ExpectedFact(
        fact_id=fact_id,
        description=description,
        subject=subject,
        predicate=predicate,
        canonical_value=value,
        value_type=value_type,
        acceptable_values=acceptable_values or [value],
        unit=unit,
        currency=currency,
        period=period,
        scope=scope,
        tolerance=tolerance,
        derivation=derivation,
        evidence_chunk_ids=chunks,
    )


def numeric_value_span(text: str, value: str) -> tuple[int, int] | None:
    """Locate a canonical number while tolerating PDF whitespace and thousands separators."""
    compact = re.sub(r"[\s,]", "", value)
    pattern = r"[\s,]*".join(re.escape(character) for character in compact)
    match = re.search(rf"(?<![\d.]){pattern}(?![\d.])", text)
    return match.span() if match else None


def expand_quote_to_cover_numeric_facts(
    text: str,
    quote: str,
    supported_facts: list[ExpectedFact],
) -> str:
    """Expand a human-selected anchor to cover every bound direct numeric fact."""
    quote_start = text.index(quote)
    starts = [quote_start]
    ends = [quote_start + len(quote)]
    for expected in supported_facts:
        if expected.value_type == "text" or expected.derivation:
            continue
        span = numeric_value_span(text, expected.canonical_value)
        if span is None:
            raise ValueError(
                f"Canonical value {expected.canonical_value!r} is absent from its source chunk"
            )
        starts.append(span[0])
        ends.append(span[1])
    return text[min(starts) : max(ends)]


def answerable(
    chunks: dict[str, tuple[dict, str]],
    *,
    query_id: str,
    family_id: str,
    split: str,
    query: str,
    company: str,
    category: str,
    difficulty: str,
    reference_answer: str,
    facts: list[ExpectedFact],
    quotes: list[tuple[str, str, list[str]]],
    required_format: str = "short",
    tags: list[str] | None = None,
) -> GenerationEvaluationItem:
    company_id, company_names, aliases = COMPANIES[company]
    evidence: list[GoldEvidence] = []
    source_hashes: set[str] = set()
    facts_by_id = {item.fact_id: item for item in facts}
    for index, (chunk_id, quote, supports) in enumerate(quotes, start=1):
        chunk, version_id = chunks[chunk_id]
        if quote not in chunk["text"]:
            raise ValueError(f"Quote mismatch for {query_id} in {chunk_id}: {quote!r}")
        quote = expand_quote_to_cover_numeric_facts(
            chunk["text"],
            quote,
            [facts_by_id[fact_id] for fact_id in supports],
        )
        source_hashes.add(chunk["document_id"].removeprefix("sha256:"))
        evidence.append(
            GoldEvidence(
                evidence_id=f"{query_id}:e{index}",
                chunk_id=chunk_id,
                document_version_id=version_id,
                page_start=chunk["page_start"],
                page_end=chunk["page_end"],
                section_path=chunk["section_path"],
                verbatim_quote=quote,
                supports_fact_ids=supports,
                pdf_visual_verified=all(
                    (version_id, page) in VISUALLY_REVIEWED
                    for page in range(chunk["page_start"], chunk["page_end"] + 1)
                ),
            )
        )
    return GenerationEvaluationItem(
        query_id=query_id,
        family_id=family_id,
        split=split,
        query=query,
        company_ids=[company_id],
        company_names=company_names,
        company_aliases=aliases,
        report_years=[2024],
        category=category,
        difficulty=difficulty,
        answerability="answerable",
        reference_answer=reference_answer,
        expected_facts=facts,
        gold_chunk_ids=list(dict.fromkeys(chunk_id for chunk_id, _, _ in quotes)),
        gold_evidence=evidence,
        tags=tags or [],
        answer_contract=AnswerContract(
            expected_behavior="answer",
            required_format=required_format,
            require_citations=True,
            require_units=True,
        ),
        required_citation_count=len({item.chunk_id for item in evidence}),
        annotation=AnnotationProvenance(
            created_by="assistant_curated",
            review_status="assistant_verified",
            confidence="high",
            source_pdf_sha256=sorted(source_hashes),
            notes="由年报文本证据核对；表格题仍需 PDF 视觉复核后升级为 human_verified。",
        ),
    )


def unanswerable(
    *,
    query_id: str,
    split: str,
    query: str,
    company: str,
    reason: str,
    behavior: str = "abstain",
    tags: list[str] | None = None,
) -> GenerationEvaluationItem:
    company_id, company_names, aliases = COMPANIES[company]
    report_years = [int(value) for value in re.findall(r"20\d{2}", query)]
    return GenerationEvaluationItem(
        query_id=query_id,
        family_id=query_id,
        split=split,
        query=query,
        company_ids=[company_id],
        company_names=company_names,
        company_aliases=aliases,
        report_years=report_years,
        category="unanswerable",
        difficulty="hard",
        answerability="unanswerable" if behavior == "abstain" else "needs_clarification",
        reference_answer=("应拒绝回答：" if behavior == "abstain" else "应先澄清：") + reason,
        answer_contract=AnswerContract(
            expected_behavior=behavior,
            required_format="abstention",
            require_citations=False,
            require_units=False,
        ),
        required_citation_count=0,
        abstention_reason=reason,
        annotation=AnnotationProvenance(
            created_by="assistant_curated",
            review_status="assistant_verified",
            confidence="high",
            source_pdf_sha256=[],
            notes="拒答/澄清压力测试，不提供伪造 gold evidence。",
        ),
        tags=tags or [],
    )


def build_items(chunks: dict[str, tuple[dict, str]]) -> list[GenerationEvaluationItem]:
    m8 = "5299f4940e2c:c8:81a5543345db6ac2"
    m9 = "5299f4940e2c:c9:b7e5f0d6a604b703"
    m11 = "5299f4940e2c:c11:4ee8dfe0e52fe716"
    m15 = "5299f4940e2c:c15:21ea8a01397a570e"
    m18 = "5299f4940e2c:c18:9e2aedcf9ff898fa"
    m19 = "5299f4940e2c:c19:807840659a321252"
    m20 = "5299f4940e2c:c20:dd96a623b5d39ea0"
    m22 = "5299f4940e2c:c22:79c3acfcbf3676ef"
    y10 = "a82a81e52f52:c10:0e6aa9a18586e694"
    y13 = "a82a81e52f52:c13:1557c17c79b6bb19"
    y49 = "a82a81e52f52:c49:5b0070d92a364736"
    y291 = "a82a81e52f52:c291:97e950a89bc304c9"

    items: list[GenerationEvaluationItem] = []

    f = [
        fact("revenue", "2024年营业收入", "贵州茅台", "营业收入", "170899152276.34", "number", "FY2024", "主要会计数据", [m8], unit="元", currency="CNY"),
        fact("yoy", "营业收入同比增幅", "贵州茅台", "同比增幅", "15.71", "percentage", "FY2024", "主要会计数据", [m8], unit="%"),
    ]
    items.append(answerable(chunks, query_id="moutai_revenue_yoy", family_id="moutai_annual_headline", split="calibration", query="贵州茅台2024年营业收入及同比增幅是多少？", company="moutai", category="single_fact", difficulty="easy", reference_answer="贵州茅台2024年营业收入为170,899,152,276.34元，同比增长15.71%[1]。", facts=f, quotes=[(m8, "营业收入\n170,899,152,276.34", ["revenue", "yoy"])], tags=["numeric", "annual"]))

    f = [fact("roe", "2024年加权平均净资产收益率", "贵州茅台", "加权平均净资产收益率", "36.02", "percentage", "FY2024", "主要财务指标", [m9], unit="%")]
    items.append(answerable(chunks, query_id="moutai_roe", family_id="moutai_annual_headline", split="calibration", query="贵州茅台2024年加权平均净资产收益率是多少？", company="moutai", category="single_fact", difficulty="easy", reference_answer="贵州茅台2024年加权平均净资产收益率为36.02%[1]。", facts=f, quotes=[(m9, "加权平均净资产收益率（%）\n36.02", ["roe"])], tags=["percentage", "annual"]))

    f = [
        fact("cashflow", "经营活动现金流量净额", "贵州茅台", "经营活动产生的现金流量净额", "92463692168.43", "number", "FY2024", "主营业务分析", [m18], unit="元", currency="CNY"),
        fact("cashflow_yoy", "经营现金流同比增幅", "贵州茅台", "同比增幅", "38.85", "percentage", "FY2024", "主营业务分析", [m18], unit="%"),
        fact("cashflow_reason", "经营现金流变动原因", "贵州茅台", "变动原因", "销售商品收到的现金增加及财务公司归集集团其他成员单位资金增加", "text", "FY2024", "管理层讨论与分析", [m18]),
    ]
    items.append(answerable(chunks, query_id="moutai_cashflow_change", family_id="moutai_cashflow_analysis", split="calibration", query="贵州茅台2024年经营活动现金流量净额、同比增幅及增长原因分别是什么？", company="moutai", category="narrative", difficulty="medium", reference_answer="经营活动现金流量净额为92,463,692,168.43元，同比增长38.85%；主要因销售商品收到的现金增加，以及财务公司归集集团其他成员单位资金增加[1]。", facts=f, quotes=[(m18, "经营活动产生的现金流量净额\n92,463,692,168.43", ["cashflow", "cashflow_yoy"]), (m18, "主要是本期公司销售商品收到的现金增加及控股子", ["cashflow_reason"])], required_format="list", tags=["numeric", "causal", "multi_fact"]))

    q_values = ["9187422415.09", "27434411397.54", "7799552404.82", "48042305950.98"]
    f = [fact(f"q{i}", f"第{i}季度经营现金流", "贵州茅台", "经营活动产生的现金流量净额", value, "number", f"2024Q{i}", "分季度主要财务数据", [m11], unit="元", currency="CNY") for i, value in enumerate(q_values, 1)]
    items.append(answerable(chunks, query_id="moutai_quarterly_cashflow", family_id="moutai_quarterly_table", split="calibration", query="贵州茅台2024年四个季度经营活动现金流量净额分别是多少？", company="moutai", category="multi_fact_table", difficulty="medium", reference_answer="第一至第四季度分别为9,187,422,415.09元、27,434,411,397.54元、7,799,552,404.82元和48,042,305,950.98元[1]。", facts=f, quotes=[(m11, "经营活动产生的现金\n流量净额", [f"q{i}" for i in range(1, 5)])], required_format="table", tags=["table_linearization", "quarterly", "negative_sensitive"]))

    yq = ["2267600204.82", "3057943456.58", "8544068622.02", "7870128109.96"]
    f = [fact(f"q{i}", f"第{i}季度经营现金流", "伊利股份", "经营活动产生的现金流量净额", value, "number", f"2024Q{i}", "分季度主要财务数据", [y13], unit="元", currency="CNY") for i, value in enumerate(yq, 1)]
    f.append(fact("annual", "全年经营现金流", "伊利股份", "经营活动产生的现金流量净额", "21739740393.38", "number", "FY2024", "主要会计数据", [y10], unit="元", currency="CNY", derivation="Q1+Q2+Q3+Q4"))
    items.append(answerable(chunks, query_id="yili_quarterly_cashflow_reconcile", family_id="yili_quarterly_table", split="calibration", query="伊利股份2024年各季度经营活动现金流量净额是多少，合计是否与全年一致？", company="yili", category="calculation", difficulty="hard", reference_answer="四季度分别为2,267,600,204.82元、3,057,943,456.58元、8,544,068,622.02元和7,870,128,109.96元；合计21,739,740,393.38元，与全年披露值一致[1][2]。", facts=f, quotes=[(y13, "经营活动产生的现金流量净额", [f"q{i}" for i in range(1, 5)]), (y10, "经营活动产生的现金流量净额\n21,739,740,393.38", ["annual"])], required_format="table", tags=["table_linearization", "calculation", "multi_evidence"]))

    profits = ["5922814507.71", "1608319620.91", "3337346216.72", "-2415620352.16"]
    f = [fact(f"q{i}", f"第{i}季度归母净利润", "伊利股份", "归属于上市公司股东的净利润", value, "number", f"2024Q{i}", "分季度主要财务数据", [y13], unit="元", currency="CNY") for i, value in enumerate(profits, 1)]
    items.append(answerable(chunks, query_id="yili_quarterly_net_profit", family_id="yili_quarterly_table", split="calibration", query="伊利股份2024年各季度归母净利润分别是多少？", company="yili", category="multi_fact_table", difficulty="medium", reference_answer="第一至第四季度分别为5,922,814,507.71元、1,608,319,620.91元、3,337,346,216.72元和-2,415,620,352.16元[1]，第四季度为负值。", facts=f, quotes=[(y13, "归属于上市公司股东的净利润", [f"q{i}" for i in range(1, 5)])], required_format="table", tags=["table_linearization", "negative_sensitive"]))

    f = [
        fact("moutai_revenue", "贵州茅台营业收入", "贵州茅台", "营业收入", "170899152276.34", "number", "FY2024", "主要会计数据", [m8], unit="元", currency="CNY"),
        fact("yili_revenue", "伊利股份营业收入", "伊利股份", "营业收入", "115393310976.69", "number", "FY2024", "主要会计数据", [y10], unit="元", currency="CNY"),
        fact("difference", "两家公司营业收入差额", "贵州茅台与伊利股份", "营业收入差额", "55505841299.65", "number", "FY2024", "跨公司比较", [m8, y10], unit="元", currency="CNY", derivation="170899152276.34-115393310976.69"),
    ]
    items.append(answerable(chunks, query_id="revenue_cross_company", family_id="annual_revenue_comparison", split="calibration", query="比较贵州茅台与伊利股份2024年营业收入，哪家更高，差额是多少？", company="both", category="comparison", difficulty="hard", reference_answer="贵州茅台为170,899,152,276.34元[1]，伊利股份为115,393,310,976.69元[2]；贵州茅台高55,505,841,299.65元。", facts=f, quotes=[(m8, "营业收入\n170,899,152,276.34", ["moutai_revenue", "difference"]), (y10, "营业收入\n115,393,310,976.69", ["yili_revenue", "difference"])], required_format="comparison", tags=["multi_document", "calculation"]))

    f = [
        fact("total", "研发投入合计", "贵州茅台", "研发投入合计", "695376735.81", "number", "FY2024", "研发投入", [m22], unit="元", currency="CNY"),
        fact("expensed", "费用化研发投入", "贵州茅台", "费用化研发投入", "593779816.96", "number", "FY2024", "研发投入", [m22], unit="元", currency="CNY"),
        fact("capitalized", "资本化研发投入", "贵州茅台", "资本化研发投入", "101596918.85", "number", "FY2024", "研发投入", [m22], unit="元", currency="CNY"),
        fact("ratio", "研发投入占营业收入比例", "贵州茅台", "研发投入占营业收入比例", "0.41", "percentage", "FY2024", "研发投入", [m22], unit="%"),
    ]
    items.append(answerable(chunks, query_id="moutai_rd_composition", family_id="moutai_rd", split="dev", query="贵州茅台2024年研发投入由哪些部分构成，合计及占营收比例是多少？", company="moutai", category="multi_fact_table", difficulty="medium", reference_answer="费用化研发投入593,779,816.96元，资本化研发投入101,596,918.85元，合计695,376,735.81元，占营业收入0.41%[1]。", facts=f, quotes=[(m22, "本期费用化研发投入\n593,779,816.96", ["expensed"]), (m22, "研发投入合计\n695,376,735.81", ["total", "capitalized", "ratio"])], required_format="list", tags=["numeric", "multi_fact"]))

    f = [
        fact("moutai", "茅台酒毛利率", "贵州茅台", "茅台酒毛利率", "94.06", "percentage", "FY2024", "主营业务分产品", [m19], unit="%"),
        fact("series", "其他系列酒毛利率", "贵州茅台", "其他系列酒毛利率", "79.87", "percentage", "FY2024", "主营业务分产品", [m19], unit="%"),
    ]
    items.append(answerable(chunks, query_id="moutai_product_margin", family_id="moutai_segment", split="dev", query="贵州茅台2024年茅台酒和其他系列酒的毛利率分别是多少？", company="moutai", category="multi_fact_table", difficulty="medium", reference_answer="茅台酒毛利率为94.06%，其他系列酒为79.87%[1]。", facts=f, quotes=[(m19, "茅台酒\n145,928,075,955.31", ["moutai", "series"])], required_format="table", tags=["table_linearization", "scope_disambiguation"]))

    f = [
        fact("direct", "直销毛利率", "贵州茅台", "直销毛利率", "95.33", "percentage", "FY2024", "主营业务分销售模式", [m19], unit="%"),
        fact("wholesale", "批发代理毛利率", "贵州茅台", "批发代理毛利率", "89.42", "percentage", "FY2024", "主营业务分销售模式", [m19], unit="%"),
        fact("difference", "毛利率差", "直销与批发代理", "毛利率差", "5.91", "percentage_point", "FY2024", "主营业务分销售模式", [m19], unit="个百分点", derivation="95.33-89.42"),
    ]
    items.append(answerable(chunks, query_id="moutai_channel_margin", family_id="moutai_segment", split="dev", query="贵州茅台2024年直销与批发代理哪种模式毛利率更高，高多少个百分点？", company="moutai", category="calculation", difficulty="hard", reference_answer="直销毛利率95.33%，批发代理89.42%；直销高5.91个百分点[1]。", facts=f, quotes=[(m19, "批发代理\n95,768,511,021.23", ["direct", "wholesale", "difference"])], required_format="comparison", tags=["table_linearization", "calculation"]))

    margins = [("liquid", "液体乳", "30.98"), ("powder", "奶粉及奶制品", "41.02"), ("cold", "冷饮产品", "37.42"), ("other", "其他产品", "23.55")]
    f = [fact(fid, f"{name}毛利率", "伊利股份", f"{name}毛利率", value, "percentage", "FY2024", "主营业务分产品", [y49], unit="%") for fid, name, value in margins]
    f.append(fact("highest", "毛利率最高产品", "伊利股份", "最高毛利率产品", "奶粉及奶制品", "text", "FY2024", "主营业务分产品", [y49]))
    items.append(answerable(chunks, query_id="yili_product_margin", family_id="yili_segment", split="dev", query="伊利股份2024年各产品毛利率分别是多少，哪类最高？", company="yili", category="multi_fact_table", difficulty="medium", reference_answer="液体乳30.98%、奶粉及奶制品41.02%、冷饮产品37.42%、其他产品23.55%；其中奶粉及奶制品最高[1]。", facts=f, quotes=[(y49, "液体乳\n75,002,672,990.21", [x[0] for x in margins] + ["highest"])], required_format="table", tags=["table_linearization", "multi_fact"]))

    f = [
        fact("business", "主要业务", "贵州茅台", "主要业务", "茅台酒及系列酒的生产与销售", "text", "FY2024", "业务情况", [m15]),
        fact("process", "生产工艺流程", "贵州茅台", "生产工艺流程", "制曲—制酒—贮存—勾兑—包装", "text", "FY2024", "业务情况", [m15]),
        fact("channels", "销售渠道", "贵州茅台", "销售模式", "直销和批发代理", "text", "FY2024", "业务情况", [m15]),
    ]
    items.append(answerable(chunks, query_id="moutai_business_chain", family_id="moutai_business", split="dev", query="贵州茅台的主要业务、生产工艺流程和销售渠道分别是什么？", company="moutai", category="narrative", difficulty="medium", reference_answer="公司主要从事茅台酒及系列酒的生产与销售；工艺流程为制曲—制酒—贮存—勾兑—包装；通过直销和批发代理渠道销售[1]。", facts=f, quotes=[(m15, "公司主要业务是茅台酒及系列酒的生产与销售", ["business"]), (m15, "制曲—制酒—贮存—勾兑—包装", ["process"]), (m15, "销售模式为：公司产品通过直销和批发代理", ["channels"])], required_format="list", tags=["narrative", "process"]))

    f = [
        fact("production", "酒类生产量", "贵州茅台", "生产量", "104384.50", "number", "FY2024", "产销量情况", [m20], unit="吨"),
        fact("sales", "酒类销售量", "贵州茅台", "销售量", "83332.76", "number", "FY2024", "产销量情况", [m20], unit="吨"),
        fact("inventory", "酒类库存量", "贵州茅台", "库存量", "310008.02", "number", "FY2024", "产销量情况", [m20], unit="吨"),
    ]
    items.append(answerable(chunks, query_id="moutai_production_sales_inventory", family_id="moutai_production", split="dev", query="贵州茅台2024年酒类生产量、销售量和库存量分别是多少？", company="moutai", category="multi_fact_table", difficulty="medium", reference_answer="酒类生产量104,384.50吨、销售量83,332.76吨、库存量310,008.02吨[1]。", facts=f, quotes=[(m20, "104,384.50 83,332.76 310,008.02", ["production", "sales", "inventory"])], required_format="table", tags=["table_linearization", "unit"]))

    f = [
        fact("valuation", "存货发出计价方法", "伊利股份", "存货计价", "加权平均法", "text", "FY2024", "重要会计政策", [y291]),
        fact("inventory_system", "存货盘存制度", "伊利股份", "存货盘存制度", "永续盘存制", "text", "FY2024", "重要会计政策", [y291]),
        fact("consumables", "低值易耗品摊销", "伊利股份", "低值易耗品摊销方法", "一次转销法", "text", "FY2024", "重要会计政策", [y291]),
    ]
    items.append(answerable(chunks, query_id="yili_inventory_policy", family_id="yili_accounting_policy", split="dev", query="伊利股份存货发出采用什么计价方法，盘存制度和低值易耗品摊销方法是什么？", company="yili", category="accounting_policy", difficulty="medium", reference_answer="存货发出按加权平均法计价，采用永续盘存制；低值易耗品采用一次转销法[1]。", facts=f, quotes=[(y291, "存货发出时按\n\n加权平均法计价", ["valuation"]), (y291, "采用永续盘存制", ["inventory_system"]), (y291, "低值易耗品采用一次转销法", ["consumables"])], required_format="list", tags=["accounting_policy", "narrative"]))

    # Additional high-value cases are defined from source-verified evidence below.
    items.extend(build_extended_items(chunks))
    items.extend(build_depth_items(chunks))

    items.extend(
        [
            unanswerable(query_id="u_moutai_2025_actual_revenue", split="calibration", query="贵州茅台2025年实际营业收入是多少？", company="moutai", reason="当前语料只有2024年报，无法确认2025年实际营业收入。", tags=["out_of_period"]),
            unanswerable(query_id="u_yili_top_customer_names", split="frozen_test", query="伊利股份2024年前五大客户分别叫什么名字？", company="yili", reason="年报仅披露前五名客户合计金额及占比，当前证据未披露具体名称。", tags=["missing_granularity"]),
            unanswerable(query_id="u_moutai_stock_cause", split="frozen_test", query="贵州茅台2024年股价上涨的主要原因是什么？", company="moutai", reason="年报语料没有足够市场价格与外部因果证据。", tags=["unsupported_causality"]),
            unanswerable(query_id="u_moutai_liquid_milk", split="dev", query="贵州茅台2024年液体乳销量是多少？", company="moutai", reason="液体乳是另一家公司业务，贵州茅台年报不支持该问题。", tags=["wrong_company", "cross_document_pollution"]),
            unanswerable(query_id="u_yili_moutai_wine", split="dev", query="伊利股份2024年茅台酒产量是多少？", company="yili", reason="茅台酒不属于伊利股份披露业务，语料不支持。", tags=["wrong_company", "cross_document_pollution"]),
            unanswerable(query_id="u_investment_advice", split="frozen_test", query="只根据这两份年报，哪家公司股票更值得买？", company="both", reason="投资建议需要估值、市场数据及个人风险偏好，年报证据不足以作规范性结论。", tags=["normative", "external_knowledge"]),
            unanswerable(query_id="u_yili_q4_loss_cause", split="dev", query="伊利股份2024年第四季度归母净亏损是因为春节备货吗？", company="yili", reason="春节备货证据关联库存或现金流，不能证明第四季度净亏损的因果关系。", tags=["unsupported_causality", "evidence_splicing"]),
            unanswerable(query_id="u_moutai_profit_ambiguous", split="frozen_test", query="茅台利润是多少？", company="moutai", reason="需要明确报告年份以及净利润、扣非净利润或其他利润口径。", behavior="clarify", tags=["ambiguous_scope", "missing_period"]),
        ]
    )
    return attach_hard_negatives(items, chunks)


def attach_hard_negatives(
    items: list[GenerationEvaluationItem],
    chunks: dict[str, tuple[dict, str]],
) -> list[GenerationEvaluationItem]:
    """Bind source-backed distractors used by the paired robustness lane."""
    m8 = "5299f4940e2c:c8:81a5543345db6ac2"
    m11 = "5299f4940e2c:c11:4ee8dfe0e52fe716"
    m18 = "5299f4940e2c:c18:9e2aedcf9ff898fa"
    m20 = "5299f4940e2c:c20:dd96a623b5d39ea0"
    m21 = "5299f4940e2c:c21:5f126f59fff9e0f8"
    m22 = "5299f4940e2c:c22:79c3acfcbf3676ef"
    m47 = "5299f4940e2c:c47:ec925bcc1be9f851"
    m131 = "5299f4940e2c:c131:f2471539423f408a"
    m133 = "5299f4940e2c:c133:a7cd2a96c54bb5de"
    m264 = "5299f4940e2c:c264:8c3ffd81c4fbcad1"
    m343 = "5299f4940e2c:c343:27674032a4ae8f15"
    y10 = "a82a81e52f52:c10:0e6aa9a18586e694"
    y13 = "a82a81e52f52:c13:1557c17c79b6bb19"
    y46 = "a82a81e52f52:c46:88b28456d076f92c"
    y49 = "a82a81e52f52:c49:5b0070d92a364736"
    y51 = "a82a81e52f52:c51:1e8ec3f24e198935"
    y54 = "a82a81e52f52:c54:4fd21da6c3f59495"
    y55 = "a82a81e52f52:c55:0f3bef67c37107bf"
    y78 = "a82a81e52f52:c78:cd65cd69e58f27e2"
    y79 = "a82a81e52f52:c79:a71230b53902b53b"
    y200 = "a82a81e52f52:c200:27ccd1f1492fcf0d"
    y202 = "a82a81e52f52:c202:8330cda8e76d7f13"
    y483 = "a82a81e52f52:c483:7456b44fce38cbcb"
    y484 = "a82a81e52f52:c484:9cf7b65b3b49fef2"
    y574 = "a82a81e52f52:c574:dccbe0c8c8f75b65"
    y603 = "a82a81e52f52:c603:73e1981521881f32"

    specs: dict[str, list[tuple[str, str, str]]] = {
        "moutai_revenue_yoy": [
            (m264, "wrong_scope", "同公司同年度的附注主营/其他业务口径，数值高度相似。"),
            (y10, "wrong_company", "另一家公司同年度主要会计数据中的营业收入。"),
        ],
        "moutai_annual_deducted_profit": [
            (m11, "wrong_period", "同公司季度表中的扣非归母净利润，不是年度值。"),
            (y10, "wrong_company", "另一家公司同年度扣非归母净利润。"),
        ],
        "yili_annual_deducted_profit": [
            (y13, "wrong_period", "同公司季度表中的扣非归母净利润，不是年度值。"),
            (m8, "wrong_company", "另一家公司同年度扣非归母净利润。"),
        ],
        "yili_quarterly_profit_reconcile": [
            (y483, "partial_evidence", "只给出全年值，无法独立完成季度加总核对。"),
            (m11, "wrong_company", "另一家公司结构相似的季度财务表。"),
        ],
        "moutai_cashflow_change": [
            (m11, "wrong_period", "同公司季度现金流表，缺少年度变动原因。"),
            (y46, "wrong_company", "另一家公司同年度现金流及原因分析。"),
        ],
        "yili_cashflow_change": [
            (y10, "partial_evidence", "主要会计数据只支持年度数值，不支持原因。"),
            (m18, "wrong_company", "另一家公司同年度现金流及原因分析。"),
        ],
        "yili_dividend_timing": [
            (y483, "partial_evidence", "只披露已实施股利总额，缺每股金额和预案信息。"),
            (y574, "partial_evidence", "只披露2024年度拟派方案，缺2023年度已实施方案。"),
        ],
        "yili_note_cost_scope": [
            (y49, "wrong_scope", "同公司分产品主营业务成本，不是附注合计口径。"),
            (y603, "wrong_scope", "同公司母公司口径营业成本，不是合并口径。"),
            (m264, "wrong_company", "另一家公司结构相同的附注营业成本表。"),
        ],
        "moutai_cost_components": [
            (m264, "wrong_scope", "营业成本总表不披露酒类五项成本构成。"),
            (y49, "wrong_company", "另一家公司分产品成本表。"),
        ],
        "yili_2025_plan_bounded": [
            (y10, "wrong_period", "2024年实际收入不能证明2025年计划必然实现。"),
        ],
        "audit_opinion_comparison": [
            (m133, "partial_evidence", "关键审计事项不能替代贵州茅台审计意见。"),
            (y202, "partial_evidence", "关键审计事项不能替代伊利股份审计意见。"),
        ],
        "moutai_concentration": [
            (y54, "wrong_company", "另一家公司同年度客户与供应商集中度表。"),
        ],
        "yili_concentration": [
            (m21, "wrong_company", "另一家公司同年度客户与供应商集中度表。"),
        ],
        "customer_concentration_comparison": [
            (m22, "wrong_scope", "贵州茅台研发占比不是客户集中度。"),
            (y55, "wrong_scope", "伊利股份研发占比不是客户集中度。"),
        ],
        "supplier_concentration_comparison": [
            (m22, "wrong_scope", "贵州茅台研发占比不是供应商集中度。"),
            (y55, "wrong_scope", "伊利股份研发占比不是供应商集中度。"),
        ],
        "moutai_disclosed_risks": [
            (y78, "wrong_company", "另一家公司披露的供应风险。"),
            (y79, "wrong_company", "另一家公司披露的财务与产品质量风险。"),
        ],
        "key_audit_matters_comparison": [
            (m131, "wrong_scope", "审计意见正文不是关键审计事项清单。"),
        ],
        "u_yili_cost_scope_ambiguous": [
            (y49, "wrong_scope", "分产品主营业务成本只是候选口径之一。"),
            (y484, "wrong_scope", "合并附注营业成本只是候选口径之一。"),
            (y603, "wrong_scope", "母公司营业成本是另一候选口径。"),
        ],
        "yili_consolidated_parent_revenue": [
            (y49, "wrong_scope", "分产品主营业务收入不能替代合并或母公司报表口径。"),
            (m343, "wrong_company", "另一家公司的母公司营业收入表。"),
        ],
        "u_yili_profit_causality": [
            (y51, "unsupported_causality", "春节备货只解释库存变化，不能解释净利润。"),
            (y46, "unsupported_causality", "春节备货只解释经营现金流变化。"),
            (y10, "partial_evidence", "年度利润数值不提供春节备货的利润因果证据。"),
        ],
        "u_yili_q4_loss_cause": [
            (y13, "partial_evidence", "季度表只证明第四季度为负，不证明原因。"),
            (y51, "unsupported_causality", "库存备货原因不能拼接为季度亏损原因。"),
            (y46, "unsupported_causality", "现金流变动原因不能拼接为季度亏损原因。"),
        ],
        "u_compare_audit_quality": [
            (m131, "partial_evidence", "标准审计意见不足以衡量相对审计质量。"),
            (y200, "partial_evidence", "标准审计意见不足以衡量相对审计质量。"),
        ],
        "u_yili_top_customer_names": [
            (y54, "partial_evidence", "只披露前五名客户汇总金额和占比，未披露名称。"),
        ],
        "u_moutai_stock_cause": [
            (m47, "partial_evidence", "公司风险披露不能证明二级市场股价上涨原因。"),
            (m8, "partial_evidence", "年度财务摘要不能单独证明股价变动原因。"),
        ],
        "u_moutai_2025_actual_revenue": [
            (m8, "wrong_period", "2024年实际收入不能当作2025年实际收入。"),
        ],
        "u_yili_moutai_wine": [
            (m20, "wrong_company", "问题中的指标存在于另一家公司年报。"),
        ],
        "u_moutai_liquid_milk": [
            (y51, "wrong_company", "问题中的指标存在于另一家公司年报。"),
        ],
        "u_moutai_profit_ambiguous": [
            (m8, "wrong_scope", "同一页存在多个利润口径，不能替用户选择。"),
        ],
        "u_investment_advice": [
            (m8, "partial_evidence", "财务摘要不能单独支持投资建议。"),
            (y10, "partial_evidence", "财务摘要不能单独支持投资建议。"),
        ],
    }
    item_by_id = {item.query_id: item for item in items}
    unknown_items = sorted(set(specs) - set(item_by_id))
    if unknown_items:
        raise ValueError(f"Hard-negative specs reference unknown items: {unknown_items}")

    enriched: list[GenerationEvaluationItem] = []
    for item in items:
        negatives = []
        for chunk_id, negative_type, reason in specs.get(item.query_id, []):
            if chunk_id not in chunks:
                raise ValueError(f"Missing hard-negative chunk: {chunk_id}")
            if chunk_id in item.gold_chunk_ids:
                raise ValueError(f"Hard negative duplicates gold for {item.query_id}: {chunk_id}")
            negatives.append(
                HardNegative(
                    chunk_id=chunk_id,
                    negative_type=negative_type,
                    reason=reason,
                )
            )
        enriched.append(item.model_copy(update={"hard_negatives": negatives}))
    return enriched


def build_extended_items(chunks: dict[str, tuple[dict, str]]) -> list[GenerationEvaluationItem]:
    """Build the narrative, audit, concentration and dividend cases."""
    m21 = "5299f4940e2c:c21:5f126f59fff9e0f8"
    m22 = "5299f4940e2c:c22:79c3acfcbf3676ef"
    m47 = "5299f4940e2c:c47:ec925bcc1be9f851"
    y24 = "a82a81e52f52:c24:7d9be8ba1ed5416f"
    y46 = "a82a81e52f52:c46:88b28456d076f92c"
    y51 = "a82a81e52f52:c51:1e8ec3f24e198935"
    y54 = "a82a81e52f52:c54:4fd21da6c3f59495"
    y55 = "a82a81e52f52:c55:0f3bef67c37107bf"
    items: list[GenerationEvaluationItem] = []

    facts = [
        fact("cashflow", "经营活动现金流量净额", "伊利股份", "经营活动产生的现金流量净额", "21739740393.38", "number", "FY2024", "主营业务分析", [y46], unit="元", currency="CNY"),
        fact("growth", "同比增幅", "伊利股份", "经营现金流同比增幅", "18.86", "percentage", "FY2024", "主营业务分析", [y46], unit="%"),
        fact("reason", "变动原因", "伊利股份", "经营现金流变动原因", "2025年春节备货导致预收经销商货款增加", "text", "FY2024", "管理层讨论与分析", [y46]),
    ]
    items.append(answerable(chunks, query_id="yili_cashflow_change", family_id="yili_cashflow_analysis", split="calibration", query="伊利股份2024年经营活动现金流量净额、同比增幅和变动原因是什么？", company="yili", category="narrative", difficulty="medium", reference_answer="经营活动现金流量净额为21,739,740,393.38元，同比增长18.86%；增加主要因2025年春节备货带来预收经销商货款增加[1]。", facts=facts, quotes=[(y46, "经营活动产生的现金流量净额\n21,739,740,393.38", ["cashflow", "growth"]), (y46, "主要因2025 年春节备货", ["reason"])], required_format="list", tags=["numeric", "causal", "time_boundary"]))

    facts = [
        fact("total", "研发投入合计", "伊利股份", "研发投入合计", "869976531.80", "number", "FY2024", "研发投入", [y55], unit="元", currency="CNY"),
        fact("ratio", "研发投入占营收比例", "伊利股份", "研发投入占营业收入比例", "0.75", "percentage", "FY2024", "研发投入", [y55], unit="%"),
        fact("staff", "研发人员数量", "伊利股份", "研发人员数量", "606", "number", "FY2024", "研发人员", [y55], unit="人"),
        fact("staff_ratio", "研发人员占比", "伊利股份", "研发人员占公司总人数比例", "0.96", "percentage", "FY2024", "研发人员", [y55], unit="%"),
    ]
    items.append(answerable(chunks, query_id="yili_rd_staff", family_id="yili_rd", split="dev", query="伊利股份2024年研发投入、占营收比例及研发人员数量和占比分别是多少？", company="yili", category="multi_fact_table", difficulty="medium", reference_answer="研发投入合计869,976,531.80元，占营业收入0.75%；研发人员606人，占公司总人数0.96%[1]。", facts=facts, quotes=[(y55, "研发投入合计\n869,976,531.80", ["total", "ratio"]), (y55, "公司研发人员的数量\n606", ["staff", "staff_ratio"])], required_format="list", tags=["numeric", "multi_fact"]))

    facts = [
        fact("moutai", "贵州茅台研发投入", "贵州茅台", "研发投入合计", "695376735.81", "number", "FY2024", "研发投入", [m22], unit="元", currency="CNY"),
        fact("yili", "伊利股份研发投入", "伊利股份", "研发投入合计", "869976531.80", "number", "FY2024", "研发投入", [y55], unit="元", currency="CNY"),
        fact("difference", "研发投入差额", "伊利股份与贵州茅台", "研发投入差额", "174599795.99", "number", "FY2024", "跨公司比较", [m22, y55], unit="元", currency="CNY", derivation="869976531.80-695376735.81"),
    ]
    items.append(answerable(chunks, query_id="rd_cross_company", family_id="rd_comparison", split="dev", query="比较贵州茅台和伊利股份2024年披露的研发投入合计，哪家更高，差额多少？", company="both", category="comparison", difficulty="hard", reference_answer="贵州茅台披露695,376,735.81元[1]，伊利股份披露869,976,531.80元[2]；伊利股份高174,599,795.99元。该比较仅限各自年报披露口径。", facts=facts, quotes=[(m22, "研发投入合计\n695,376,735.81", ["moutai", "difference"]), (y55, "研发投入合计\n869,976,531.80", ["yili", "difference"])], required_format="comparison", tags=["multi_document", "calculation", "scope_disclaimer"]))

    facts = [
        fact("production", "液体乳生产量", "伊利股份", "液体乳生产量", "9034835", "number", "FY2024", "产销量情况", [y51], unit="吨"),
        fact("sales", "液体乳销售量", "伊利股份", "液体乳销售量", "8926606", "number", "FY2024", "产销量情况", [y51], unit="吨"),
        fact("inventory", "液体乳库存量", "伊利股份", "液体乳库存量", "333555", "number", "FY2024", "产销量情况", [y51], unit="吨"),
        fact("reason", "库存增加原因", "伊利股份", "液体乳库存增加原因", "2025年春节备货导致期末库存量较大", "text", "FY2024", "产销量情况", [y51]),
    ]
    items.append(answerable(chunks, query_id="yili_liquid_milk_inventory", family_id="yili_production", split="dev", query="伊利股份2024年液体乳产量、销量、库存量及库存增加原因是什么？", company="yili", category="narrative", difficulty="medium", reference_answer="液体乳产量9,034,835吨、销量8,926,606吨、库存量333,555吨；库存增加主要因2025年春节备货使期末库存量较大[1]。", facts=facts, quotes=[(y51, "9,034,835\n8,926,606\n333,555", ["production", "sales", "inventory"]), (y51, "2025 年春节备货导致期末库存量较大", ["reason"])], required_format="list", tags=["table_linearization", "causal", "time_boundary"]))

    facts = [
        fact("customer", "前五名客户销售占比", "贵州茅台", "客户集中度", "11.52", "percentage", "FY2024", "主要客户", [m21], unit="%"),
        fact("supplier", "前五名供应商采购占比", "贵州茅台", "供应商集中度", "35.43", "percentage", "FY2024", "主要供应商", [m21], unit="%"),
    ]
    items.append(answerable(chunks, query_id="moutai_concentration", family_id="concentration_moutai", split="frozen_test", query="贵州茅台2024年前五名客户销售占比和前五名供应商采购占比分别是多少？", company="moutai", category="multi_fact_table", difficulty="easy", reference_answer="前五名客户销售额占年度销售总额11.52%，前五名供应商采购额占年度采购总额35.43%[1]。", facts=facts, quotes=[(m21, "占年度销售总额11.52%", ["customer"]), (m21, "占年度采购总额35.43%", ["supplier"])], required_format="list", tags=["percentage", "frozen_test"]))

    facts = [
        fact("customer", "前五名客户销售占比", "伊利股份", "客户集中度", "6.17", "percentage", "FY2024", "主要客户", [y54], unit="%"),
        fact("supplier", "前五名供应商采购占比", "伊利股份", "供应商集中度", "40.03", "percentage", "FY2024", "主要供应商", [y54], unit="%"),
    ]
    items.append(answerable(chunks, query_id="yili_concentration", family_id="concentration_yili", split="frozen_test", query="伊利股份2024年前五名客户销售占比和前五名供应商采购占比分别是多少？", company="yili", category="multi_fact_table", difficulty="easy", reference_answer="前五名客户销售占比6.17%，前五名供应商采购占比40.03%[1]。", facts=facts, quotes=[(y54, "占年度销售总额6.17%", ["customer"]), (y54, "占年度采购总额40.03%", ["supplier"])], required_format="list", tags=["percentage", "frozen_test"]))

    facts = [
        fact("moutai", "贵州茅台客户集中度", "贵州茅台", "前五名客户销售占比", "11.52", "percentage", "FY2024", "主要客户", [m21], unit="%"),
        fact("yili", "伊利股份客户集中度", "伊利股份", "前五名客户销售占比", "6.17", "percentage", "FY2024", "主要客户", [y54], unit="%"),
        fact("difference", "客户集中度差", "贵州茅台与伊利股份", "客户集中度差", "5.35", "percentage_point", "FY2024", "跨公司比较", [m21, y54], unit="个百分点", derivation="11.52-6.17"),
    ]
    items.append(answerable(chunks, query_id="customer_concentration_comparison", family_id="concentration_comparison", split="frozen_test", query="贵州茅台和伊利股份2024年前五名客户销售占比谁更高，高多少个百分点？", company="both", category="comparison", difficulty="hard", reference_answer="贵州茅台为11.52%[1]，伊利股份为6.17%[2]；贵州茅台高5.35个百分点。", facts=facts, quotes=[(m21, "占年度销售总额11.52%", ["moutai", "difference"]), (y54, "占年度销售总额6.17%", ["yili", "difference"])], required_format="comparison", tags=["multi_document", "calculation", "frozen_test"]))

    facts = [
        fact("moutai", "贵州茅台供应商集中度", "贵州茅台", "前五名供应商采购占比", "35.43", "percentage", "FY2024", "主要供应商", [m21], unit="%"),
        fact("yili", "伊利股份供应商集中度", "伊利股份", "前五名供应商采购占比", "40.03", "percentage", "FY2024", "主要供应商", [y54], unit="%"),
        fact("difference", "供应商集中度差", "伊利股份与贵州茅台", "供应商集中度差", "4.60", "percentage_point", "FY2024", "跨公司比较", [m21, y54], unit="个百分点", derivation="40.03-35.43"),
    ]
    items.append(answerable(chunks, query_id="supplier_concentration_comparison", family_id="concentration_comparison", split="frozen_test", query="贵州茅台和伊利股份2024年前五名供应商采购占比谁更高，高多少个百分点？", company="both", category="comparison", difficulty="hard", reference_answer="贵州茅台为35.43%[1]，伊利股份为40.03%[2]；伊利股份高4.60个百分点。", facts=facts, quotes=[(m21, "占年度采购总额35.43%", ["moutai", "difference"]), (y54, "占年度采购总额40.03%", ["yili", "difference"])], required_format="comparison", tags=["multi_document", "calculation", "frozen_test"]))

    facts = [fact("risks", "披露的风险类型", "贵州茅台", "可能面对的风险", "宏观经济风险；安全风险；舆情风险；环境保护风险", "text", "FY2024", "可能面对的风险", [m47])]
    items.append(answerable(chunks, query_id="moutai_disclosed_risks", family_id="moutai_risks", split="frozen_test", query="贵州茅台2024年报披露了哪些可能面对的风险？", company="moutai", category="narrative", difficulty="easy", reference_answer="年报披露了宏观经济风险、安全风险、舆情风险和环境保护风险四类[1]。", facts=facts, quotes=[(m47, "一是宏观经济风险；二是安全风险；三是舆情风险；四是环境保护风险", ["risks"])], required_format="list", tags=["narrative", "bounded_answer", "frozen_test"]))

    facts = [
        fact("mode", "销售模式", "伊利股份", "销售模式", "经销为主，直营为辅", "text", "FY2024", "营销与销售模式", [y24]),
        fact("channels", "覆盖渠道", "伊利股份", "销售渠道", "商超、便利店、传统渠道、零食店、电商、O2O、餐饮、特殊渠道", "text", "FY2024", "营销与销售模式", [y24]),
    ]
    items.append(answerable(chunks, query_id="yili_sales_channels", family_id="yili_business", split="dev", query="伊利股份采用什么销售模式，覆盖哪些渠道？", company="yili", category="narrative", difficulty="easy", reference_answer="公司采用经销为主、直营为辅的销售模式，覆盖商超、便利店、传统渠道、零食店、电商、O2O、餐饮和特殊渠道等[1]。", facts=facts, quotes=[(y24, "公司采用经销为主，直营为辅的销售模式", ["mode"]), (y24, "商超、便利店、传统渠道、零食店、电商、O2O、餐饮、特殊渠道", ["channels"])], required_format="list", tags=["narrative", "channel"]))
    return items


def build_depth_items(chunks: dict[str, tuple[dict, str]]) -> list[GenerationEvaluationItem]:
    """Add scope disambiguation, reconciliation, audit and adversarial cases."""
    m8 = "5299f4940e2c:c8:81a5543345db6ac2"
    m12 = "5299f4940e2c:c12:e0540e67db8c53cf"
    m20 = "5299f4940e2c:c20:dd96a623b5d39ea0"
    m131 = "5299f4940e2c:c131:f2471539423f408a"
    m133 = "5299f4940e2c:c133:a7cd2a96c54bb5de"
    m136 = "5299f4940e2c:c136:671ffd5693972ca2"
    m264 = "5299f4940e2c:c264:8c3ffd81c4fbcad1"
    y10 = "a82a81e52f52:c10:0e6aa9a18586e694"
    y13 = "a82a81e52f52:c13:1557c17c79b6bb19"
    y15 = "a82a81e52f52:c15:54215d8787ee02a9"
    y1 = "a82a81e52f52:c1:793234fe6adf3a5b"
    y106 = "a82a81e52f52:c106:637feb5ba3b1b6ff"
    y77 = "a82a81e52f52:c77:ed0d447e9403ac62"
    y78 = "a82a81e52f52:c78:cd65cd69e58f27e2"
    y79 = "a82a81e52f52:c79:a71230b53902b53b"
    y200 = "a82a81e52f52:c200:27ccd1f1492fcf0d"
    y202 = "a82a81e52f52:c202:8330cda8e76d7f13"
    y203 = "a82a81e52f52:c203:b6f4494584265fdc"
    y484 = "a82a81e52f52:c484:9cf7b65b3b49fef2"
    y603 = "a82a81e52f52:c603:73e1981521881f32"
    items: list[GenerationEvaluationItem] = []

    facts = [
        fact("reported", "年度归母净利润", "贵州茅台", "归母净利润", "86228146421.62", "number", "FY2024", "主要会计数据", [m8], unit="元", currency="CNY"),
        fact("deducted", "年度扣非归母净利润", "贵州茅台", "归母扣非净利润", "86240905977.42", "number", "FY2024", "主要会计数据", [m8], unit="元", currency="CNY"),
        fact("nonrecurring", "非经常性损益合计", "贵州茅台", "非经常性损益", "-12759555.80", "number", "FY2024", "非经常性损益", [m12], unit="元", currency="CNY"),
        fact("difference", "扣非净利润高出金额", "贵州茅台", "利润口径差额", "12759555.80", "number", "FY2024", "利润口径核对", [m8, m12], unit="元", currency="CNY", derivation="86240905977.42-86228146421.62"),
    ]
    items.append(answerable(chunks, query_id="moutai_annual_deducted_profit", family_id="moutai_annual_headline", split="calibration", query="贵州茅台2024年归母净利润与扣非归母净利润分别是多少？结合非经常性损益合计核对两者差额。", company="moutai", category="calculation", difficulty="hard", reference_answer="归母净利润为86,228,146,421.62元，扣非归母净利润为86,240,905,977.42元[1]；扣非口径高12,759,555.80元，与非经常性损益合计-12,759,555.80元的方向和金额一致[2]。", facts=facts, quotes=[(m8, "归属于上市公司股东的净\n利润", ["reported", "deducted", "difference"]), (m12, "合计\n-12,759,555.80", ["nonrecurring", "difference"])], required_format="comparison", tags=["scope_disambiguation", "calculation", "multi_evidence", "corrected_legacy_gold"]))

    facts = [
        fact("reported", "年度归母净利润", "伊利股份", "归母净利润", "8452859993.18", "number", "FY2024", "主要会计数据", [y10], unit="元", currency="CNY"),
        fact("deducted", "年度扣非归母净利润", "伊利股份", "归母扣非净利润", "6011274945.92", "number", "FY2024", "主要会计数据", [y10], unit="元", currency="CNY"),
        fact("nonrecurring", "非经常性损益合计", "伊利股份", "非经常性损益", "2441585047.26", "number", "FY2024", "非经常性损益", [y15], unit="元", currency="CNY"),
        fact("difference", "归母净利润高出金额", "伊利股份", "利润口径差额", "2441585047.26", "number", "FY2024", "利润口径核对", [y10, y15], unit="元", currency="CNY", derivation="8452859993.18-6011274945.92"),
    ]
    items.append(answerable(chunks, query_id="yili_annual_deducted_profit", family_id="yili_annual_headline", split="calibration", query="伊利股份2024年归母净利润与扣非归母净利润分别是多少？结合非经常性损益合计核对两者差额。", company="yili", category="calculation", difficulty="hard", reference_answer="归母净利润为8,452,859,993.18元，扣非归母净利润为6,011,274,945.92元[1]；归母口径高2,441,585,047.26元，与非经常性损益合计2,441,585,047.26元一致[2]。", facts=facts, quotes=[(y10, "归属于上市公司股东的净利润", ["reported", "deducted", "difference"]), (y15, "合计\n2,441,585,047.26", ["nonrecurring", "difference"])], required_format="comparison", tags=["scope_disambiguation", "calculation", "multi_evidence", "corrected_legacy_gold"]))

    quarterly = ["5922814507.71", "1608319620.91", "3337346216.72", "-2415620352.16"]
    facts = [fact(f"q{i}", f"第{i}季度归母净利润", "伊利股份", "归母净利润", value, "number", f"2024Q{i}", "季度主要财务数据", [y13], unit="元", currency="CNY") for i, value in enumerate(quarterly, 1)]
    facts.extend([
        fact("quarter_sum", "四个季度归母净利润合计", "伊利股份", "季度合计", "8452859993.18", "number", "FY2024", "季度数据加总", [y13], unit="元", currency="CNY", derivation="Q1+Q2+Q3+Q4"),
        fact("annual", "全年披露归母净利润", "伊利股份", "归母净利润", "8452859993.18", "number", "FY2024", "主要会计数据", [y10], unit="元", currency="CNY"),
        fact("reconciles", "季度合计是否与全年一致", "伊利股份", "勾稽关系", "true", "boolean", "FY2024", "季度与年度核对", [y13, y10], derivation="quarter_sum == annual", acceptable_values=["一致", "相等"]),
    ])
    items.append(answerable(chunks, query_id="yili_quarterly_profit_reconcile", family_id="yili_quarterly_table", split="calibration", query="伊利股份2024年四个季度归母净利润分别是多少？加总后是否等于全年披露值？", company="yili", category="calculation", difficulty="hard", reference_answer="第一至第四季度分别为5,922,814,507.71元、1,608,319,620.91元、3,337,346,216.72元和-2,415,620,352.16元[1]；合计8,452,859,993.18元，与全年披露值8,452,859,993.18元一致[2]。", facts=facts, quotes=[(y13, "归属于上市公司股东的净利润", [f"q{i}" for i in range(1, 5)] + ["quarter_sum", "reconciles"]), (y10, "8,452,859,993.18", ["annual", "reconciles"])], required_format="comparison", tags=["calculation", "negative_sensitive", "multi_evidence"]))

    facts = [
        fact("implemented_per_share", "2023年度现金分红每股金额", "伊利股份", "已实施每股现金红利", "1.20", "number", "2024 implementation", "2023年度利润分配实施", [y106], unit="元/股", currency="CNY"),
        fact("implemented_total", "2023年度现金分红总额", "伊利股份", "已实施现金红利总额", "7639318446.00", "number", "2024 implementation", "2023年度利润分配实施", [y106], unit="元", currency="CNY"),
        fact("proposal_per_share", "2024年度分红预案每股金额", "伊利股份", "拟派发每股现金红利", "1.22", "number", "2024 proposal", "2024年度利润分配预案", [y1], unit="元/股", currency="CNY"),
        fact("proposal_total", "2024年度分红预案总额", "伊利股份", "拟派发现金红利总额", "7726310439.68", "number", "2024 proposal", "2024年度利润分配预案", [y1], unit="元", currency="CNY"),
        fact("status", "实施状态差异", "伊利股份", "利润分配状态", "2023年度方案已实施，2024年度方案为拟派发预案", "text", "FY2024", "利润分配时点", [y106, y1]),
    ]
    items.append(answerable(chunks, query_id="yili_dividend_timing", family_id="yili_dividend", split="dev", query="区分伊利股份在2024年内实施的2023年度现金分红与年报提出的2024年度分红预案：每股和总额分别是多少，哪一项已实施？", company="yili", category="comparison", difficulty="hard", reference_answer="2024年内已实施的是2023年度方案：每股1.20元、总额7,639,318,446.00元[1]；2024年度方案仍是拟派发预案，每股1.22元、拟派总额7,726,310,439.68元[2]。", facts=facts, quotes=[(y106, chunks[y106][0]["text"], ["implemented_per_share", "implemented_total", "status"]), (y1, chunks[y1][0]["text"], ["proposal_per_share", "proposal_total", "status"])], required_format="comparison", tags=["time_boundary", "status_disambiguation", "multi_evidence", "calculation"]))

    facts = [
        fact("main", "主营业务本期成本", "伊利股份", "主营业务成本", "75283113436.98", "number", "FY2024", "财务报表附注61", [y484], unit="元", currency="CNY"),
        fact("other", "其他业务本期成本", "伊利股份", "其他业务成本", "1015719596.95", "number", "FY2024", "财务报表附注61", [y484], unit="元", currency="CNY"),
        fact("total", "营业成本合计", "伊利股份", "营业成本合计", "76298833033.93", "number", "FY2024", "财务报表附注61", [y484], unit="元", currency="CNY"),
    ]
    items.append(answerable(chunks, query_id="yili_note_cost_scope", family_id="yili_note_cost", split="dev", query="伊利股份财务报表附注中2024年主营业务、其他业务和营业成本合计分别是多少？", company="yili", category="multi_fact_table", difficulty="hard", reference_answer="主营业务成本75,283,113,436.98元，其他业务成本1,015,719,596.95元，营业成本合计76,298,833,033.93元[1]。", facts=facts, quotes=[(y484, "主营业务\n114,120,632,145.60\n75,283,113,436.98", ["main"]), (y484, "其他业务\n1,272,678,831.09\n1,015,719,596.95", ["other", "total"])], required_format="table", tags=["scope_disambiguation", "financial_note", "table_linearization"]))

    facts = [
        fact("main", "主营业务成本", "贵州茅台", "主营业务成本", "13629995812.89", "number", "FY2024", "财务报表附注40", [m264], unit="元", currency="CNY"),
        fact("total", "营业成本合计", "贵州茅台", "营业成本合计", "13789482367.98", "number", "FY2024", "财务报表附注40", [m264], unit="元", currency="CNY"),
        fact("difference", "口径差额", "贵州茅台", "其他业务成本", "159486555.09", "number", "FY2024", "口径核对", [m264], unit="元", currency="CNY", derivation="13789482367.98-13629995812.89"),
    ]
    items.append(answerable(chunks, query_id="moutai_cost_reconciliation", family_id="moutai_note_cost", split="dev", query="为什么贵州茅台2024年主营业务成本与营业成本合计不同，差额是多少？", company="moutai", category="calculation", difficulty="hard", reference_answer="主营业务成本13,629,995,812.89元，营业成本合计13,789,482,367.98元；差额159,486,555.09元，来自其他业务成本[1]。", facts=facts, quotes=[(m264, "其他业务\n287,314,224.32\n159,486,555.09", ["main", "total", "difference"])], required_format="comparison", tags=["scope_disambiguation", "calculation"]))

    facts = [
        fact("main", "主营业务成本", "伊利股份", "主营业务成本", "75283113436.98", "number", "FY2024", "财务报表附注61", [y484], unit="元", currency="CNY"),
        fact("total", "营业成本合计", "伊利股份", "营业成本合计", "76298833033.93", "number", "FY2024", "财务报表附注61", [y484], unit="元", currency="CNY"),
        fact("difference", "口径差额", "伊利股份", "其他业务成本", "1015719596.95", "number", "FY2024", "口径核对", [y484], unit="元", currency="CNY", derivation="76298833033.93-75283113436.98"),
    ]
    items.append(answerable(chunks, query_id="yili_cost_reconciliation", family_id="yili_note_cost", split="dev", query="为什么伊利股份2024年主营业务成本与营业成本合计不同，差额是多少？", company="yili", category="calculation", difficulty="hard", reference_answer="主营业务成本75,283,113,436.98元，营业成本合计76,298,833,033.93元；差额1,015,719,596.95元，来自其他业务成本[1]。", facts=facts, quotes=[(y484, "其他业务\n1,272,678,831.09\n1,015,719,596.95", ["main", "total", "difference"])], required_format="comparison", tags=["scope_disambiguation", "calculation"]))

    components = [("material", "直接材料", "6895320421.92"), ("labor", "直接人工", "5224448485.08"), ("manufacturing", "制造费用", "776373890.79"), ("energy", "燃料动力", "422328634.52"), ("transport", "运输费", "311524380.58")]
    facts = [fact(fid, name, "贵州茅台", name, value, "number", "FY2024", "酒类成本构成", [m20], unit="元", currency="CNY") for fid, name, value in components]
    items.append(answerable(chunks, query_id="moutai_cost_components", family_id="moutai_production", split="dev", query="贵州茅台2024年酒类成本由哪些项目构成，各项目金额是多少？", company="moutai", category="multi_fact_table", difficulty="hard", reference_answer="直接材料6,895,320,421.92元、直接人工5,224,448,485.08元、制造费用776,373,890.79元、燃料动力422,328,634.52元、运输费311,524,380.58元[1]。", facts=facts, quotes=[(m20, "直接材料\n6,895,320,421.92", [x[0] for x in components])], required_format="table", tags=["table_linearization", "multi_fact"]))

    facts = [
        fact("target", "2025年营业总收入计划", "伊利股份", "营业总收入计划", "1190", "number", "FY2025 plan", "经营计划", [y77], unit="亿元", currency="CNY"),
        fact("guarantee", "是否构成业绩承诺", "伊利股份", "业绩承诺", "不构成业绩承诺，存在不确定性", "text", "FY2025 plan", "经营计划", [y77]),
    ]
    items.append(answerable(chunks, query_id="yili_2025_plan_bounded", family_id="yili_plan", split="dev", query="伊利股份是否保证2025年一定实现1,190亿元营业总收入？", company="yili", category="narrative", difficulty="hard", reference_answer="不能保证。1,190亿元是2025年经营计划目标，受未来经营环境影响存在不确定性，且明确不构成业绩承诺[1]。", facts=facts, quotes=[(y77, "计划实现营业总收入1,190 亿元", ["target"]), (y77, "该经营目标受未来经营环境影响，存在一定的不确定性，并不构成对投资者的业绩承诺", ["guarantee"])], required_format="short", tags=["bounded_answer", "future_plan", "epistemic"]))

    facts = [
        fact("supply", "供应风险", "伊利股份", "可能面对的风险", "供应风险", "text", "FY2024", "可能面对的风险", [y78]),
        fact("financial", "财务风险", "伊利股份", "可能面对的风险", "财务风险", "text", "FY2024", "可能面对的风险", [y79]),
        fact("quality", "产品质量风险", "伊利股份", "可能面对的风险", "产品质量风险", "text", "FY2024", "可能面对的风险", [y79]),
    ]
    items.append(answerable(chunks, query_id="yili_disclosed_risks", family_id="yili_risks", split="dev", query="伊利股份2024年报披露了哪些主要风险？", company="yili", category="narrative", difficulty="medium", reference_answer="主要包括供应风险[1]、财务风险和产品质量风险[2]。", facts=facts, quotes=[(y78, "1、供应风险", ["supply"]), (y79, "2、财务风险", ["financial"]), (y79, "3、产品质量风险", ["quality"])], required_format="list", tags=["bounded_answer", "narrative"]))

    revenue_scope_facts = [
        fact("consolidated", "合并口径营业收入", "伊利股份", "营业收入", "115393310976.69", "number", "FY2024", "合并财务报表附注", [y484], unit="元", currency="CNY"),
        fact("parent", "母公司口径营业收入", "伊利股份", "营业收入", "102395394662.41", "number", "FY2024", "母公司财务报表附注", [y603], unit="元", currency="CNY"),
        fact("difference", "合并与母公司营业收入差额", "伊利股份", "营业收入差额", "12997916314.28", "number", "FY2024", "报表口径核对", [y484, y603], unit="元", currency="CNY", derivation="115393310976.69-102395394662.41"),
    ]
    items.extend([
        unanswerable(query_id="u_yili_cost_scope_ambiguous", split="dev", query="伊利股份2024年的成本是多少？", company="yili", reason="需要明确营业成本、分产品成本、期间费用以及合并或母公司等口径。", behavior="clarify", tags=["ambiguous_scope", "financial_scope"]),
        answerable(chunks, query_id="yili_consolidated_parent_revenue", family_id="yili_revenue_scope", split="dev", query="伊利股份2024年合并口径与母公司口径营业收入分别是多少，差额多少？", company="yili", category="comparison", difficulty="hard", reference_answer="合并口径营业收入为115,393,310,976.69元[1]，母公司口径为102,395,394,662.41元[2]；合并口径高12,997,916,314.28元。", facts=revenue_scope_facts, quotes=[(y484, "合计\n115,393,310,976.69", ["consolidated", "difference"]), (y603, "合计\n102,395,394,662.41", ["parent", "difference"])], required_format="comparison", tags=["scope_disambiguation", "consolidated_vs_parent", "calculation", "multi_evidence"]),
        unanswerable(query_id="u_yili_profit_causality", split="dev", query="伊利股份2024年净利润下降就是因为春节备货吗？", company="yili", reason="春节备货证据支持现金流或库存变化，不能据此推断全年净利润下降原因。", tags=["unsupported_causality", "evidence_splicing"]),
    ])

    facts = [
        fact("moutai_opinion", "贵州茅台审计意见", "贵州茅台", "审计意见", "财务报表在所有重大方面按企业会计准则编制并公允反映", "text", "FY2024", "审计报告", [m131]),
        fact("yili_opinion", "伊利股份审计意见", "伊利股份", "审计意见", "财务报表在所有重大方面按企业会计准则编制并公允反映", "text", "FY2024", "审计报告", [y200]),
    ]
    items.append(answerable(chunks, query_id="audit_opinion_comparison", family_id="audit_opinion", split="frozen_test", query="贵州茅台和伊利股份2024年财务报表的审计意见分别是什么？", company="both", category="comparison", difficulty="medium", reference_answer="两家公司审计报告均认为，财务报表在所有重大方面按照企业会计准则编制，并公允反映相应财务状况、经营成果和现金流量[1][2]。", facts=facts, quotes=[(m131, "在所有重大方面按照企业会计准则的规定编制，公允反映了贵州", ["moutai_opinion"]), (y200, "在所有重大方面按照企业会计准则的规定编制，公允反映了伊利", ["yili_opinion"])], required_format="comparison", tags=["multi_document", "bounded_answer", "audit"]))

    facts = [
        fact("moutai_revenue", "贵州茅台收入确认事项", "贵州茅台", "关键审计事项", "收入确认", "text", "FY2024", "审计报告", [m133]),
        fact("moutai_related", "贵州茅台关联方事项", "贵州茅台", "关键审计事项", "关联方及其交易", "text", "FY2024", "审计报告", [m136]),
        fact("yili_revenue", "伊利股份收入确认事项", "伊利股份", "关键审计事项", "收入确认", "text", "FY2024", "审计报告", [y202]),
        fact("yili_goodwill", "伊利股份减值事项", "伊利股份", "关键审计事项", "商誉和商标权减值", "text", "FY2024", "审计报告", [y203]),
    ]
    items.append(answerable(chunks, query_id="key_audit_matters_comparison", family_id="key_audit_matters", split="frozen_test", query="贵州茅台和伊利股份2024年关键审计事项分别有哪些？", company="both", category="comparison", difficulty="hard", reference_answer="贵州茅台的关键审计事项为收入确认[1]、关联方及其交易[2]；伊利股份为收入确认[3]、商誉和商标权减值[4]。", facts=facts, quotes=[(m133, "(一) 收入确认", ["moutai_revenue"]), (m136, "(二) 关联方及其交易", ["moutai_related"]), (y202, "1.收入确认事项", ["yili_revenue"]), (y203, "由于商誉和商标权金额较大", ["yili_goodwill"])], required_format="comparison", tags=["multi_document", "multi_evidence", "audit"]))

    items.append(unanswerable(query_id="u_compare_audit_quality", split="frozen_test", query="仅根据贵州茅台和伊利股份2024年审计意见，哪家公司的审计质量更高？", company="both", reason="两份报告的标准审计意见不能直接推出审计质量高低，需要更多审计过程与质量证据。", tags=["normative", "unsupported_comparison", "audit"]))
    return items


def main() -> None:
    chunks = load_chunks()
    items = build_items(chunks)
    payload_seed = "\n".join(item.model_dump_json() for item in items)
    dataset_id = "generation-eval-v1-" + hashlib.sha256(payload_seed.encode()).hexdigest()[:12]
    dataset = GenerationEvaluationDataset(
        dataset_id=dataset_id,
        corpus_index_id="10fb50419145d56720c9",
        independent_gold=False,
        reviewer="assistant source-verified; pending independent human/PDF visual review",
        status="assistant_curated_provisional",
        tracks=["oracle_context", "retrieved_context", "robustness"],
        item_count=len(items),
        items=items,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")
    print(f"Wrote {dataset.item_count} items to {OUTPUT}")
    print(f"Dataset ID: {dataset.dataset_id}")


if __name__ == "__main__":
    main()
