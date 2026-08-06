import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from findoc_rag.diagnostics import (
    DiagnosticDataset,
    DocumentProfile,
    EvidenceJudgment,
    RankingDiagnostic,
)
from findoc_rag.documents.models import DocumentChunk
from findoc_rag.indexing import PersistentIndex
from findoc_rag.io import read_jsonl
from findoc_rag.registry import DocumentRegistry


class HoldoutSpec(BaseModel):
    metric: str
    scope: str
    query_template: str
    cues: tuple[str, ...]


class ProposedEvidence(BaseModel):
    chunk_id: str
    page_start: int
    page_end: int
    section_path: list[str]
    excerpt: str
    reason: str


class HoldoutReviewItem(BaseModel):
    review_id: str
    query: str
    company: str
    year: int
    metric: str
    scope: str
    proposed_evidence: ProposedEvidence
    alternatives: list[ProposedEvidence]
    status: Literal["pending", "approved", "rejected", "edited"] = "pending"
    reviewer_query: str | None = None
    reviewer_gold_chunk_ids: list[str] = []
    reviewer_notes: str = ""


class HoldoutReviewPack(BaseModel):
    schema_version: int = 1
    pack_id: str
    source_index_id: str
    excluded_query_count: int
    item_count: int
    items: list[HoldoutReviewItem]


class AssistantReviewDecision(BaseModel):
    review_id: str
    status: Literal["approved", "edited", "rejected", "needs_review"]
    gold_chunk_ids: list[str] = []
    confidence: Literal["high", "medium", "low"]
    notes: str = ""


class AssistantReviewSet(BaseModel):
    schema_version: int = 1
    source_pack: str
    reviewer: str
    review_method: str
    independent_gold: bool = False
    items: list[AssistantReviewDecision]

    def benchmark_items(self) -> list[AssistantReviewDecision]:
        """Return only decisions safe for provisional retrieval evaluation."""
        return [item for item in self.items if item.status in {"approved", "edited"} and item.gold_chunk_ids]

    def validate_chunk_ids(self, available_chunk_ids: set[str]) -> dict[str, list[str]]:
        """Report missing evidence IDs before a provisional benchmark run."""
        missing: dict[str, list[str]] = {}
        for item in self.benchmark_items():
            absent = [chunk_id for chunk_id in item.gold_chunk_ids if chunk_id not in available_chunk_ids]
            if absent:
                missing[item.review_id] = absent
        return missing


def load_holdout_eval(path: Path) -> list[dict]:
    """Load the normalized, reviewed holdout manifest."""
    import json
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("independent_gold") is not False:
        raise ValueError("Holdout manifest must remain marked as non-independent gold")
    items = payload.get("items", [])
    if len(items) != 16:
        raise ValueError(f"Expected 16 reviewed items, got {len(items)}")
    return items


def holdout_eval_to_diagnostics(path: Path, index: PersistentIndex) -> DiagnosticDataset:
    """Convert reviewed holdout evidence into the existing retrieval evaluator schema."""
    manifest = load_holdout_eval(path)
    chunk_ids = sorted({chunk_id for item in manifest for chunk_id in item["gold_chunk_ids"]})
    chunks = index._load_chunks(chunk_ids)
    missing = sorted(set(chunk_ids) - set(chunks))
    if missing:
        raise ValueError(f"Holdout gold chunks are absent from index: {', '.join(missing)}")
    queries = []
    query_text = {
        "moutai_deducted_net_profit": "贵州茅台2024年扣除非经常性损益后的净利润是多少",
        "moutai_total_assets": "贵州茅台2024年末总资产是多少",
        "moutai_attributable_net_assets": "贵州茅台2024年末归属于上市公司股东的净资产是多少",
        "moutai_roe": "贵州茅台2024年加权平均净资产收益率是多少",
        "moutai_segment_cost": "贵州茅台2024年主营业务分行业或分产品营业成本是多少",
        "moutai_segment_gross_margin": "贵州茅台2024年主营业务分行业或分产品毛利率是多少",
        "moutai_note_operating_cost": "贵州茅台2024年财务报表附注披露的营业成本是多少",
        "yili_deducted_net_profit": "伊利股份2024年扣除非经常性损益后的净利润是多少",
        "yili_total_assets": "伊利股份2024年末总资产是多少",
        "yili_attributable_net_assets": "伊利股份2024年末归属于上市公司股东的净资产是多少",
        "yili_roe": "伊利股份2024年加权平均净资产收益率是多少",
        "yili_quarterly_net_profit": "伊利股份2024年各季度归母净利润分别是多少",
        "yili_quarterly_operating_cashflow": "伊利股份2024年各季度经营活动现金流量净额分别是多少",
        "yili_segment_cost": "伊利股份2024年主营业务分行业或分产品营业成本是多少",
        "yili_segment_gross_margin": "伊利股份2024年主营业务分行业或分产品毛利率是多少",
        "yili_note_operating_cost": "伊利股份2024年财务报表附注披露的营业成本是多少",
    }
    for item in manifest:
        judgments = []
        for chunk_id in item["gold_chunk_ids"]:
            chunk = chunks[chunk_id]
            judgments.append(EvidenceJudgment(
                chunk_id=chunk_id, label="relevant", reason="assistant-reviewed holdout evidence",
                page_start=chunk.page_start, page_end=chunk.page_end,
                section_path=chunk.section_path, excerpt=chunk.text[:800], source="structural_rule",
            ))
        queries.append(RankingDiagnostic(
            query_id=item["review_id"], query=query_text.get(item["query_id"], item["query_id"]),
            company=item["company"], year=item["year"], metric=item["query_id"],
            scope="unspecified", status="accepted", judgments=judgments,
        ))
    content = "\n".join(query.model_dump_json() for query in queries)
    return DiagnosticDataset(
        dataset_id=hashlib.sha256(content.encode()).hexdigest()[:20],
        index_id=index.manifest.index_id, generator="assistant-reviewed-holdout-v2",
        query_count=len(queries), accepted_count=len(queries), needs_review_count=0,
        queries=queries,
    )


HOLDOUT_SPECS = (
    HoldoutSpec(metric="扣除非经常性损益", scope="年度主要财务指标", query_template="{company}{year}年扣除非经常性损益后的净利润是多少", cues=("主要会计数据",)),
    HoldoutSpec(metric="总资产", scope="年度主要财务指标", query_template="{company}{year}年末总资产是多少", cues=("主要会计数据",)),
    HoldoutSpec(metric="净资产", scope="年度主要财务指标", query_template="{company}{year}年末归属于上市公司股东的净资产是多少", cues=("主要会计数据",)),
    HoldoutSpec(metric="加权平均净资产收益率", scope="年度主要财务指标", query_template="{company}{year}年加权平均净资产收益率是多少", cues=("主要财务指标",)),
    HoldoutSpec(metric="归属于上市公司股东的净利润", scope="季度财务数据", query_template="{company}{year}年各季度归母净利润分别是多少", cues=("分季度",)),
    HoldoutSpec(metric="经营活动产生的现金流量净额", scope="季度财务数据", query_template="{company}{year}年各季度经营活动现金流量净额分别是多少", cues=("分季度",)),
    HoldoutSpec(metric="营业成本", scope="主营业务分部", query_template="{company}{year}年主营业务分行业或分产品营业成本是多少", cues=("主营业务", "分行业", "分产品")),
    HoldoutSpec(metric="毛利率", scope="主营业务分部", query_template="{company}{year}年主营业务分行业或分产品毛利率是多少", cues=("主营业务", "分行业", "分产品")),
    HoldoutSpec(metric="营业利润", scope="合并利润表", query_template="{company}{year}年合并利润表中的营业利润是多少", cues=("合并利润表",)),
    HoldoutSpec(metric="利润总额", scope="合并利润表", query_template="{company}{year}年合并利润表中的利润总额是多少", cues=("合并利润表",)),
    HoldoutSpec(metric="净利润", scope="合并利润表", query_template="{company}{year}年合并利润表中的净利润是多少", cues=("合并利润表",)),
    HoldoutSpec(metric="营业成本", scope="财务报表附注", query_template="{company}{year}年财务报表附注披露的营业成本是多少", cues=("营业收入和营业成本",)),
)


def _evidence(chunk: DocumentChunk, reason: str) -> ProposedEvidence:
    return ProposedEvidence(
        chunk_id=chunk.chunk_id,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        section_path=chunk.section_path,
        excerpt=chunk.text[:800],
        reason=reason,
    )


def generate_holdout_review_pack(
    registry: DocumentRegistry,
    profiles: list[DocumentProfile],
    source_index_id: str,
    excluded_dataset: DiagnosticDataset,
) -> HoldoutReviewPack:
    excluded = {item.query for item in excluded_dataset.queries}
    active = {item.document_key: item for item in registry.active_versions()}
    items: list[HoldoutReviewItem] = []
    for profile in profiles:
        version = active.get(profile.document_key)
        if version is None or not version.chunks_path:
            raise ValueError(f"Profile is not active: {profile.document_key}")
        chunks = read_jsonl(Path(version.chunks_path), DocumentChunk)
        for spec in HOLDOUT_SPECS:
            query = spec.query_template.format(company=profile.company, year=profile.year)
            if query in excluded:
                continue
            matches = [chunk for chunk in chunks if spec.metric in chunk.text]
            anchored = [
                chunk
                for chunk in matches
                if any(cue in " ".join(chunk.section_path) for cue in spec.cues)
            ]
            if not anchored:
                continue
            def specificity(chunk: DocumentChunk, spec: HoldoutSpec = spec) -> tuple[int, int, int]:
                context = " ".join(chunk.section_path)
                score = sum(cue in context for cue in spec.cues)
                if spec.metric == "扣除非经常性损益" and "净利润" in chunk.text:
                    score += 5
                if spec.metric == "净资产" and "归属于上市公司股东的净资产" in chunk.text:
                    score += 5
                if spec.scope == "主营业务分部" and (
                    "分行业" in context or "分产品" in context
                ):
                    score += 5
                return (-score, chunk.chunk_index, -chunk.character_count)

            anchored.sort(key=specificity)
            proposed = anchored[0]
            alternatives = [chunk for chunk in matches if chunk.chunk_id != proposed.chunk_id][:3]
            seed = f"{profile.document_key}:{spec.scope}:{spec.metric}:holdout-v1"
            items.append(
                HoldoutReviewItem(
                    review_id=hashlib.sha256(seed.encode()).hexdigest()[:16],
                    query=query,
                    company=profile.company,
                    year=profile.year,
                    metric=spec.metric,
                    scope=spec.scope,
                    proposed_evidence=_evidence(
                        proposed,
                        f"metric occurs under structural cues: {', '.join(spec.cues)}",
                    ),
                    alternatives=[
                        _evidence(chunk, "same metric in another candidate passage")
                        for chunk in alternatives
                    ],
                )
            )
    content = "\n".join(item.model_dump_json() for item in items)
    return HoldoutReviewPack(
        pack_id=hashlib.sha256(content.encode()).hexdigest()[:20],
        source_index_id=source_index_id,
        excluded_query_count=len(excluded),
        item_count=len(items),
        items=items,
    )


def render_review_markdown(pack: HoldoutReviewPack) -> str:
    lines = [
        "# Holdout candidate review",
        "",
        f"Pack ID: `{pack.pack_id}`",
        "",
        "For each item, verify the question is unambiguous and the proposed passage directly answers it.",
        "Reply with `approve`, `reject`, or an edited query/gold chunk ID. Do not infer values from memory.",
        "",
    ]
    for number, item in enumerate(pack.items, start=1):
        evidence = item.proposed_evidence
        lines.extend(
            [
                f"## {number}. {item.query}",
                "",
                f"- Review ID: `{item.review_id}`",
                f"- Scope: `{item.scope}`",
                f"- Proposed pages: {evidence.page_start}-{evidence.page_end}",
                f"- Section: {' > '.join(evidence.section_path)}",
                f"- Proposed chunk: `{evidence.chunk_id}`",
                "- Decision: `pending`",
                "",
                "> " + evidence.excerpt.replace("\n", " ")[:500],
                "",
            ]
        )
    return "\n".join(lines) + "\n"
