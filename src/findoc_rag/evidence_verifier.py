"""Bounded evaluator-optimizer workflow for grounded extraction answers."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from findoc_rag.agent_tasks import (
    AgentTaskRequest,
    AgentTaskResult,
    AgentTaskTrace,
    ClaimRiskFinding,
    ClaimRiskGateTrace,
    EvidenceSupportProof,
    EvidenceVerificationFinding,
    EvidenceVerificationTrace,
    EvidenceVerificationTurn,
    SufficiencyDecision,
)
from findoc_rag.answer_generation import GeneratedAnswer
from findoc_rag.deepseek_agent import (
    DeepSeekExtractAgent,
    ToolCallingModel,
    _meaningful_numeric_tokens,
    _parse_extraction_submission,
    _repair_extract_authoritative_citations,
    _repair_submitted_fact_scope_labels,
    _submit_extraction_tool,
)

VERIFIER_PROMPT_REVISION = "evidence-verifier-p4e-bounded-proof-context-v11"
VERIFIER_ABLATION_PROMPT_REVISION = (
    "evidence-verifier-p4e-support-proof-off-v9"
)
CLAIM_RISK_GATE_REVISION = "claim-risk-gate-p4d-contract-completeness-v5"
MAX_VERIFIER_EVIDENCE_ITEMS = 12
MAX_VERIFIER_EXCERPT_CHARS = 1800
CLAIM_SUPPORT_BIGRAM_THRESHOLD = 0.72
CLAIM_SUPPORT_MIN_BIGRAMS = 6
REQUIREMENT_CONTRACT_SIMILARITY_THRESHOLD = 0.99
SUPPORT_PROOF_CHALLENGE_COVERAGE_THRESHOLD = 0.90
DEFAULT_MIN_VERIFIER_REQUIREMENTS = 4
YEAR_PATTERN = re.compile(r"(?<!\d)(?:19|20|21)\d{2}(?!\d)")
LANGUAGE_CHARACTER_PATTERN = re.compile(r"[\u4e00-\u9fffA-Za-z]")
CONTRACT_CHARACTER_PATTERN = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9%()+.\-（）]"
)
UNIT_PATTERN = re.compile(
    r"个百分点|万亿元|百万元|千万元|亿元|万元|千元|基点|BP|bp|%|元"
)
AMOUNT_PATTERN = re.compile(
    r"(?<![\d])-?\d[\d,]*(?:\.\d+)?\s*"
    r"(?:万亿元|百万元|千万元|亿元|万元|千元|元)"
)
SIGNED_NUMBER_PATTERN = re.compile(
    r"(?<!\d)-?\d[\d,]*(?:\.\d+)?%?(?!\d)"
)
UNIT_CANONICAL = {
    "个百分点": "percentage_point",
    "%": "percent",
    "基点": "basis_point",
    "BP": "basis_point",
    "bp": "basis_point",
    "万亿元": "trillion_yuan",
    "亿元": "hundred_million_yuan",
    "千万元": "ten_million_yuan",
    "百万元": "million_yuan",
    "万元": "ten_thousand_yuan",
    "千元": "thousand_yuan",
    "元": "yuan",
}
AMOUNT_FACTORS = {
    "万亿元": Decimal(1000000000000),
    "亿元": Decimal(100000000),
    "千万元": Decimal(10000000),
    "百万元": Decimal(1000000),
    "万元": Decimal(10000),
    "千元": Decimal(1000),
    "元": Decimal(1),
}


class SubmitEvidenceVerification(BaseModel):
    """Compact verifier response: supported IDs plus only actionable findings."""

    model_config = ConfigDict(extra="forbid")

    supported_requirement_ids: list[str] = Field(default_factory=list, max_length=24)
    support_proofs: list[EvidenceSupportProof] = Field(default_factory=list, max_length=24)
    findings: list[EvidenceVerificationFinding] = Field(default_factory=list, max_length=24)


def _parse_verification_submission(arguments: str) -> SubmitEvidenceVerification:
    payload = json.loads(arguments)
    for finding in payload.get("findings", []):
        if not str(finding.get("feedback", "")).strip():
            finding["feedback"] = "verifier supplied no feedback"
    return SubmitEvidenceVerification.model_validate(payload)


def _verification_tool(
    requirement_ids: list[str],
    evidence_ids: list[str],
    *,
    require_support_proof: bool,
    support_proof_requirement_ids: list[str] | None = None,
) -> list[dict]:
    properties: dict = {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "requirement_id": {
                        "type": "string",
                        "enum": requirement_ids,
                    },
                    "verdict": {
                        "type": "string",
                        "enum": [
                            "incomplete",
                            "contradicted",
                            "insufficient_evidence",
                        ],
                    },
                    "feedback": {"type": "string"},
                    "evidence_chunk_ids": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": evidence_ids,
                        },
                    },
                    "missing_supported_details": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "requirement_id",
                    "verdict",
                    "feedback",
                    "evidence_chunk_ids",
                    "missing_supported_details",
                ],
                "additionalProperties": False,
            },
        },
    }
    required = ["findings"]
    if require_support_proof:
        proof_requirement_ids = (
            requirement_ids
            if support_proof_requirement_ids is None
            else support_proof_requirement_ids
        )
        properties["support_proofs"] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "requirement_id": {
                        "type": "string",
                        "enum": proof_requirement_ids,
                    },
                    "claim": {"type": "string"},
                    "evidence_quotes": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": {
                            "type": "object",
                            "properties": {
                                "evidence_chunk_id": {
                                    "type": "string",
                                    "enum": evidence_ids,
                                },
                                "quote": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 320,
                                },
                            },
                            "required": ["evidence_chunk_id", "quote"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["requirement_id", "claim", "evidence_quotes"],
                "additionalProperties": False,
            },
        }
        required.append("support_proofs")
    properties["supported_requirement_ids"] = {
        "type": "array",
        "items": {"type": "string", "enum": requirement_ids},
    }
    required.append("supported_requirement_ids")
    return [
        {
            "type": "function",
            "function": {
                "name": "submit_evidence_verification",
                "description": (
                    "逐项审计当前回答是否被给定证据完整支持。支持项提交可核验证明；"
                    "不完整、矛盾或证据不足项提交结构化finding。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }
    ]


def _selected_evidence_ids(trace: AgentTaskTrace) -> list[str]:
    selected: list[str] = []
    for requirement in trace.plan.fact_requirements:
        selected.extend(trace.result.requirement_evidence.get(requirement.requirement_id, []))
    for requirement in trace.plan.fact_requirements:
        selected.extend(requirement.candidate_evidence_chunk_ids)
    known = {item.chunk_id for item in trace.evidence_memory.items}
    return [
        chunk_id
        for chunk_id in dict.fromkeys(selected)
        if chunk_id in known
    ][:MAX_VERIFIER_EVIDENCE_ITEMS]


def _verification_payload(trace: AgentTaskTrace) -> dict:
    evidence_ids = _selected_evidence_ids(trace)
    evidence_by_id = {item.chunk_id: item for item in trace.evidence_memory.items}
    selected = set(evidence_ids)
    return {
        "query": trace.query,
        "precheck_findings": (
            [
                finding.model_dump(mode="json")
                for finding in trace.claim_risk_gate.findings
            ]
            if trace.claim_risk_gate is not None
            else []
        ),
        "document_scope": (
            trace.plan.document_scope.model_dump(mode="json")
            if trace.plan.document_scope is not None
            else None
        ),
        "requirements": [
            {
                "requirement_id": requirement.requirement_id,
                "description": requirement.description,
                "subject": requirement.subject,
                "subject_scope": requirement.subject_scope,
                "fact_period": requirement.fact_period,
                "evidence_type": requirement.evidence_type,
                "current_claims": trace.result.requirement_claims.get(
                    requirement.requirement_id, []
                ),
                "cited_evidence_chunk_ids": [
                    chunk_id
                    for chunk_id in trace.result.requirement_evidence.get(
                        requirement.requirement_id, []
                    )
                    if chunk_id in selected
                ],
                "candidate_evidence_chunk_ids": [
                    chunk_id
                    for chunk_id in requirement.candidate_evidence_chunk_ids
                    if chunk_id in selected
                ],
            }
            for requirement in trace.plan.fact_requirements
        ],
        "evidence": [
            {
                "chunk_id": chunk_id,
                "company_name": evidence_by_id[chunk_id].company_name,
                "report_year": evidence_by_id[chunk_id].report_year,
                "pages": [
                    evidence_by_id[chunk_id].page_start,
                    evidence_by_id[chunk_id].page_end,
                ],
                "section_path": evidence_by_id[chunk_id].section_path,
                "text": evidence_by_id[chunk_id].excerpt[:MAX_VERIFIER_EXCERPT_CHARS],
            }
            for chunk_id in evidence_ids
        ],
    }


def _verification_messages(
    payload: dict,
    *,
    require_support_proof: bool,
) -> list[dict]:
    proof_instruction = (
        "support_proof_requirement_ids 列出的 requirement 若判为 supported，禁止只报"
        "ID：必须提交 support_proof；其他支持项提交 supported_requirement_ids。"
        "claim 必须逐字复制 current_claims 中对应 claim；每个 evidence_quote 必须逐字"
        "摘自该 requirement 当前 cited evidence，而不是 candidate evidence。引用最小但"
        "足以同时绑定主体/指标、关系和数值的连续片段，不得把同页不同指标拼成证明。"
        "系统会独立校验 claim、引用归属、原文片段和语言支持覆盖；证明缺失、非原文或"
        "无法支持 claim 时将升级人工复核。"
        if require_support_proof
        else "本次为 support proof 关闭消融，supported requirement 只提交 ID。"
    )
    return [
        {
            "role": "system",
            "content": (
                "你是独立证据审计 Agent。你没有主 Agent 的对话上下文，也不能使用外部知识。"
                "逐个 requirement 检查：主体、期间、集团/分部口径、数字、单位以及要求列举的"
                "细项是否都被当前 claim 回答，并被给定 evidence 直接支持。不要因为语句听起来"
                "合理就通过。supported 必须由该 requirement 的 cited_evidence_chunk_ids"
                "直接支持，candidate evidence 不能替代当前引用，只能帮助提出修复。证据支持"
                "但回答漏掉细项时标 incomplete，并在"
                "missing_supported_details 中写出证据可直接支持、需要补入的短事实；claim 与"
                "证据冲突标 contradicted；当前证据无法判断标 insufficient_evidence。"
                "如果 claim 新增了证据没有的因果、否定、预测、范围限定或其他结论，标"
                "insufficient_evidence，而不是 incomplete。incomplete 只表示证据已经"
                "支持、但当前回答遗漏了问题要求的事实，因此必须提供非空的"
                "missing_supported_details。"
                "requirement 是当前回答必须遵守的原子事实契约。即使 claim 中每个数字和词"
                "都能在同一 evidence 页面找到，也要核对这些数字是否属于 requirement 指定"
                "的指标、业务主体和关系，不能把同页另一行标签移到当前数值上。增长/下降、"
                "增加/减少、母公司/少数股东等关系必须与 requirement 和对应行一致。财务"
                "数字外层圆括号表示负数；claim 擅自增加或删除会计括号属于数值符号冲突。"
                "precheck_findings 只是本地异常提示，不是最终判决；必须用 evidence 独立"
                "核验后再分类。"
                "主体是硬约束：claim 如果显式写了与 requirement.subject、query 或"
                "document_scope 不同的公司/业务主体，必须标 contradicted。完整性也要按字段"
                "逐项对照，requirement 中要求的数字、比例、变动和原因少任一项都标"
                "incomplete，不能因为句子整体相关而通过。"
                f"{proof_instruction}"
                "只提交结构化工具结果，不输出思维过程。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        },
    ]


def _repair_messages(payload: dict, findings: list[EvidenceVerificationFinding]) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "你是抽取结果修复 Agent。只能使用给定 evidence，不得凭记忆补数。根据独立"
                "审计 finding 修复一次：覆盖全部 requirement；每个事实绑定直接支持它的"
                "evidence chunk；保留主体、期间、口径和单位。若证据不足则提交 abstain。"
                "不要讨论审计过程，只调用 submit_extraction。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    **payload,
                    "verifier_findings": [
                        finding.model_dump(mode="json") for finding in findings
                    ],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]


def _support_proof_challenge_messages(payload: dict) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "你是证据证明挑战器，只检查输入中列出的低语言对齐 support proof。"
                "逐项比较 claim 与 verbatim evidence_quotes 的主体、指标、关系、方向、"
                "数值和单位。claim 与 quote 出现上升/下降、增加/减少、收入/成本、"
                "母公司/少数股东等相反绑定时必须标 contradicted；quote 缺少判断所需"
                "信息时，可用 evidence_context 识别该 quote 的表头、年份、单位和相邻列；"
                "主体、关系和数值仍必须由 quote 绑定，不得用 context 中其他行或指标替代。"
                "只有 quote 与其表头上下文直接蕴含 claim 时才提交 supported_requirement_ids。"
                "只提交结构化工具结果，不输出思维过程。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]


def _prompt_revision(*, require_support_proof: bool) -> str:
    return (
        VERIFIER_PROMPT_REVISION
        if require_support_proof
        else VERIFIER_ABLATION_PROMPT_REVISION
    )


def _prompt_sha256(*, require_support_proof: bool) -> str:
    material = {
        "revision": _prompt_revision(
            require_support_proof=require_support_proof
        ),
        "verifier_system": _verification_messages(
            {},
            require_support_proof=require_support_proof,
        )[0]["content"],
        "repair_system": _repair_messages({}, [])[0]["content"],
        "support_proof_challenge_system": _support_proof_challenge_messages(
            {}
        )[0]["content"],
    }
    return hashlib.sha256(
        json.dumps(material, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _years(text: str) -> set[int]:
    return {int(value) for value in YEAR_PATTERN.findall(text)}


def _units(text: str) -> set[str]:
    return {
        UNIT_CANONICAL[match.group()]
        for match in UNIT_PATTERN.finditer(text)
    }


def _language_bigrams(text: str) -> list[str]:
    normalized = "".join(LANGUAGE_CHARACTER_PATTERN.findall(text.lower()))
    return [
        normalized[index : index + 2]
        for index in range(len(normalized) - 1)
    ]


def _language_coverage(claim: str, reference: str) -> tuple[float, int]:
    claim_bigrams = _language_bigrams(claim)
    if not claim_bigrams:
        return 1.0, 0
    reference_bigrams = set(_language_bigrams(reference))
    supported = sum(
        bigram in reference_bigrams for bigram in claim_bigrams
    )
    return supported / len(claim_bigrams), len(claim_bigrams)


def _contract_text(text: str) -> str:
    return "".join(CONTRACT_CHARACTER_PATTERN.findall(text.lower()))


def _requirement_claim_similarity(
    requirement_description: str,
    requirement_subject: str | None,
    requirement_period: str | None,
    claim: str,
) -> float:
    normalized_claim = _contract_text(claim)
    candidates = [
        requirement_description,
        (
            f"{requirement_subject or ''}{requirement_period or ''}"
            f"{requirement_description}"
        ),
    ]
    return max(
        SequenceMatcher(
            None,
            _contract_text(candidate),
            normalized_claim,
            autojunk=False,
        ).ratio()
        for candidate in candidates
    )


def _signed_numeric_forms(text: str) -> dict[str, set[str]]:
    forms: dict[str, set[str]] = {}
    for match in SIGNED_NUMBER_PATTERN.finditer(text):
        raw = match.group()
        normalized = raw.replace(",", "")
        unsigned = normalized.removeprefix("-").removesuffix("%")
        if unsigned.isdigit() and len(unsigned) == 4:
            year = int(unsigned)
            if 1900 <= year <= 2100:
                continue
        accounting_negative = (
            match.start() > 0
            and match.end() < len(text)
            and text[match.start() - 1] in {"(", "（"}
            and text[match.end()] in {")",
                "）",
            }
        )
        sign = (
            "negative"
            if normalized.startswith("-") or accounting_negative
            else "positive"
        )
        token = normalized.removeprefix("-")
        forms.setdefault(token, set()).add(sign)
    return forms


def _amount_pairs(text: str) -> list[tuple[str, str, Decimal]]:
    pairs: list[tuple[str, str, Decimal]] = []
    for match in AMOUNT_PATTERN.finditer(text):
        raw = re.search(r"-?\d[\d,]*(?:\.\d+)?", match.group())
        unit = next(
            (
                candidate
                for candidate in AMOUNT_FACTORS
                if candidate in match.group()
            ),
            None,
        )
        if raw is None or unit is None:
            continue
        normalized = raw.group().replace(",", "")
        try:
            base_value = Decimal(normalized) * AMOUNT_FACTORS[unit]
        except InvalidOperation:
            continue
        pairs.append((normalized, UNIT_CANONICAL[unit], base_value))
    return pairs


def _amounts_equivalent(left: Decimal, right: Decimal) -> bool:
    scale = max(abs(left), abs(right), Decimal(1))
    return abs(left - right) / scale <= Decimal("0.0005")


def _supported_numbers(claim: str, reference: str) -> set[str]:
    reference_numbers = _meaningful_numeric_tokens(reference)
    supported = set(reference_numbers)
    supported.update(value.removeprefix("-") for value in reference_numbers)
    if "percent" in _units(reference):
        for value in list(supported):
            if not value.endswith("%"):
                supported.add(f"{value}%")
    reference_amounts = _amount_pairs(reference)
    claim_amounts = _amount_pairs(claim)
    supported_claim_amounts = [
        claim_amount
        for token, _, claim_amount in claim_amounts
        if token in supported or token.removeprefix("-") in supported
    ]
    for token, _, claim_amount in claim_amounts:
        if any(
            _amounts_equivalent(claim_amount, reference_amount)
            for reference_amount in [
                *[value for _, _, value in reference_amounts],
                *supported_claim_amounts,
            ]
        ):
            supported.add(token)
            supported.add(token.removeprefix("-"))
    return supported


def _supported_units(claim: str, reference: str) -> set[str]:
    supported = _units(reference)
    reference_amounts = _amount_pairs(reference)
    reference_numbers = _meaningful_numeric_tokens(reference)
    claim_amounts = _amount_pairs(claim)
    supported_claim_amounts = [
        claim_amount
        for token, _, claim_amount in claim_amounts
        if token in reference_numbers
        or token.removeprefix("-") in reference_numbers
    ]
    for _, unit, claim_amount in claim_amounts:
        if any(
            _amounts_equivalent(claim_amount, reference_amount)
            for reference_amount in [
                *[value for _, _, value in reference_amounts],
                *supported_claim_amounts,
            ]
        ):
            supported.add(unit)
    return supported


def evaluate_claim_risk_gate(
    trace: AgentTaskTrace,
    *,
    known_companies: list[str] | None = None,
    enable_open_language_risk: bool = True,
    enable_requirement_contract_risk: bool = True,
    enable_accounting_sign_risk: bool = True,
) -> ClaimRiskGateTrace:
    """Find explicit claim/evidence conflicts without semantic model calls."""

    if (
        trace.task_type != "extract"
        or trace.result.outcome != "answer"
        or not trace.plan.fact_requirements
    ):
        return ClaimRiskGateTrace(
            revision=CLAIM_RISK_GATE_REVISION,
            status="not_applicable",
            checked_requirement_count=0,
        )
    evidence_by_id = {
        item.chunk_id: item for item in trace.evidence_memory.items
    }
    scope = trace.plan.document_scope
    allowed_companies = set(scope.company_names if scope is not None else [])
    allowed_report_years = set(scope.report_years if scope is not None else [])
    company_vocabulary = set(known_companies or []) | allowed_companies
    simple_fast_path = (
        len(trace.plan.fact_requirements) < DEFAULT_MIN_VERIFIER_REQUIREMENTS
        and not any(
            requirement.evidence_type in {"audit_risk", "audit_response"}
            for requirement in trace.plan.fact_requirements
        )
    )
    findings: list[ClaimRiskFinding] = []
    seen: set[tuple[str, str, str]] = set()

    def add_finding(finding: ClaimRiskFinding) -> None:
        key = (finding.requirement_id, finding.check, finding.detail)
        if key not in seen:
            seen.add(key)
            findings.append(finding)

    for requirement in trace.plan.fact_requirements:
        requirement_id = requirement.requirement_id
        claims = trace.result.requirement_claims.get(requirement_id, [])
        evidence_ids = trace.result.requirement_evidence.get(requirement_id, [])
        cited_evidence = [
            evidence_by_id[chunk_id]
            for chunk_id in evidence_ids
            if chunk_id in evidence_by_id
        ]
        reference_text = " ".join(
            [
                requirement.subject or "",
                requirement.fact_period or "",
                requirement.description,
                " ".join(sorted(allowed_companies)),
                " ".join(str(year) for year in sorted(allowed_report_years)),
            ]
            + [item.excerpt for item in cited_evidence]
        )
        joined_claims = " ".join(claims)
        requirement_supported_numbers = _supported_numbers(
            requirement.description,
            joined_claims,
        )
        missing_requirement_numbers = sorted(
            {
                value
                for value in _meaningful_numeric_tokens(
                    requirement.description
                )
                if value not in requirement_supported_numbers
                and value.removeprefix("-")
                not in requirement_supported_numbers
            }
        )
        if missing_requirement_numbers:
            add_finding(
                ClaimRiskFinding(
                    requirement_id=requirement_id,
                    check="missing_requirement_numeric",
                    claim=(joined_claims or requirement.description),
                    detail=(
                        "requirement numbers absent from current claims: "
                        + ", ".join(missing_requirement_numbers)
                    ),
                    evidence_chunk_ids=evidence_ids[:5],
                )
            )
        allowed_fact_years = _years(
            f"{requirement.fact_period or ''} {requirement.description} "
            + " ".join(item.excerpt for item in cited_evidence)
        )
        for claim in claims:
            claim_signed_forms = _signed_numeric_forms(claim)
            contract_signed_forms = _signed_numeric_forms(
                requirement.description
            )
            sign_conflicts = sorted(
                token
                for token, signs in claim_signed_forms.items()
                if enable_accounting_sign_risk
                and "negative" in signs
                and token in contract_signed_forms
                and "negative" not in contract_signed_forms[token]
            )
            if sign_conflicts:
                add_finding(
                    ClaimRiskFinding(
                        requirement_id=requirement_id,
                        check="accounting_sign_conflict",
                        claim=claim,
                        detail=(
                            "claim introduces a negative/accounting-parenthesis "
                            "sign absent from the requirement contract for: "
                            + ", ".join(sign_conflicts)
                        ),
                        evidence_chunk_ids=evidence_ids[:5],
                    )
                )
            contract_similarity = _requirement_claim_similarity(
                requirement.description,
                requirement.subject,
                requirement.fact_period,
                claim,
            )
            if (
                enable_requirement_contract_risk
                and simple_fast_path
                and contract_similarity
                < REQUIREMENT_CONTRACT_SIMILARITY_THRESHOLD
            ):
                add_finding(
                    ClaimRiskFinding(
                        requirement_id=requirement_id,
                        check="requirement_claim_divergence",
                        claim=claim,
                        detail=(
                            "claim/requirement contract similarity "
                            f"{contract_similarity:.3f} is below review threshold "
                            f"{REQUIREMENT_CONTRACT_SIMILARITY_THRESHOLD:.3f}"
                        ),
                        evidence_chunk_ids=evidence_ids[:5],
                    )
                )
            wrong_companies = sorted(
                company
                for company in company_vocabulary
                if company in claim and company not in allowed_companies
            )
            if wrong_companies:
                add_finding(
                    ClaimRiskFinding(
                        requirement_id=requirement_id,
                        check="subject_conflict",
                        claim=claim,
                        detail=(
                            "claim names out-of-scope company: "
                            + ", ".join(wrong_companies)
                        ),
                        evidence_chunk_ids=evidence_ids[:5],
                    )
                )
            claim_years = _years(claim)
            unexpected_years = sorted(
                claim_years - allowed_fact_years
                if allowed_fact_years
                else set()
            )
            if unexpected_years:
                add_finding(
                    ClaimRiskFinding(
                        requirement_id=requirement_id,
                        check="period_conflict",
                        claim=claim,
                        detail=(
                            "claim contains period absent from its requirement: "
                            + ", ".join(str(year) for year in unexpected_years)
                        ),
                        evidence_chunk_ids=evidence_ids[:5],
                    )
                )
            supported_numbers = _supported_numbers(claim, reference_text)
            unsupported_numbers = sorted(
                {
                    value
                    for value in _meaningful_numeric_tokens(claim)
                    if value not in supported_numbers
                    and value.removeprefix("-")
                    not in supported_numbers
                }
            )
            if unsupported_numbers:
                add_finding(
                    ClaimRiskFinding(
                        requirement_id=requirement_id,
                        check="unsupported_numeric",
                        claim=claim,
                        detail=(
                            "claim numbers absent from requirement and cited evidence: "
                            + ", ".join(unsupported_numbers)
                        ),
                        evidence_chunk_ids=evidence_ids[:5],
                    )
                )
            unsupported_units = sorted(
                _units(claim) - _supported_units(claim, reference_text)
            )
            if unsupported_units:
                add_finding(
                    ClaimRiskFinding(
                        requirement_id=requirement_id,
                        check="unsupported_unit",
                        claim=claim,
                        detail=(
                            "claim units absent from requirement and cited evidence: "
                            + ", ".join(unsupported_units)
                        ),
                        evidence_chunk_ids=evidence_ids[:5],
                    )
                )
            language_coverage, language_bigram_count = _language_coverage(
                claim, reference_text
            )
            if (
                enable_open_language_risk
                and
                language_bigram_count >= CLAIM_SUPPORT_MIN_BIGRAMS
                and language_coverage < CLAIM_SUPPORT_BIGRAM_THRESHOLD
            ):
                add_finding(
                    ClaimRiskFinding(
                        requirement_id=requirement_id,
                        check="low_evidence_language_coverage",
                        claim=claim,
                        detail=(
                            "claim/evidence language-bigram coverage "
                            f"{language_coverage:.3f} is below review threshold "
                            f"{CLAIM_SUPPORT_BIGRAM_THRESHOLD:.3f}"
                        ),
                        evidence_chunk_ids=evidence_ids[:5],
                    )
                )
        for item in cited_evidence:
            company_conflict = (
                allowed_companies
                and item.company_name is not None
                and item.company_name not in allowed_companies
            )
            year_conflict = (
                allowed_report_years
                and item.report_year is not None
                and item.report_year not in allowed_report_years
            )
            if company_conflict or year_conflict:
                add_finding(
                    ClaimRiskFinding(
                        requirement_id=requirement_id,
                        check="citation_scope_conflict",
                        claim=(claims[0] if claims else requirement.description),
                        detail=(
                            f"citation {item.chunk_id} is outside document scope"
                        ),
                        evidence_chunk_ids=[item.chunk_id],
                    )
                )
    hard_conflict = any(
        finding.check
        in {
            "subject_conflict",
            "period_conflict",
            "citation_scope_conflict",
            "accounting_sign_conflict",
        }
        for finding in findings
    )
    return ClaimRiskGateTrace(
        revision=CLAIM_RISK_GATE_REVISION,
        status=(
            "reject"
            if hard_conflict
            else "review"
            if findings
            else "pass"
        ),
        checked_requirement_count=len(trace.plan.fact_requirements),
        findings=findings,
    )


def _normalized_quote_text(value: str) -> str:
    return re.sub(r"\s+", "", value)


def _quote_is_grounded(quote: str, evidence_text: str) -> bool:
    normalized_quote = _normalized_quote_text(quote)
    normalized_evidence = _normalized_quote_text(evidence_text)
    if normalized_quote and normalized_quote in normalized_evidence:
        return True
    segments = [
        _normalized_quote_text(segment)
        for segment in quote.splitlines()
        if _normalized_quote_text(segment)
    ]
    if len(segments) < 2:
        return False
    cursor = 0
    for segment in segments:
        index = normalized_evidence.find(segment, cursor)
        if index < 0:
            return False
        cursor = index + len(segment)
    return True


def _support_proof_requirement_ids(trace: AgentTaskTrace) -> list[str]:
    """Select the weakest atomic contract for a bounded proof challenge."""
    if any(
        requirement.evidence_type in {"audit_risk", "audit_response"}
        for requirement in trace.plan.fact_requirements
    ):
        return []
    ranked: list[tuple[float, int, str]] = []
    for index, requirement in enumerate(trace.plan.fact_requirements):
        claims = trace.result.requirement_claims.get(requirement.requirement_id, [])
        if not claims:
            continue
        similarity = min(
            _requirement_claim_similarity(
                requirement.description,
                requirement.subject,
                requirement.fact_period,
                claim,
            )
            for claim in claims
        )
        ranked.append((similarity, index, requirement.requirement_id))
    return [min(ranked)[2]] if ranked else []


def _support_proof_review_reasons(
    trace: AgentTaskTrace,
    payload: dict,
    submission: SubmitEvidenceVerification,
    supported_requirement_ids: list[str],
    proof_requirement_ids: list[str],
) -> tuple[list[str], list[str]]:
    reasons: list[str] = []
    challenge_requirement_ids: list[str] = []
    requirements = {
        requirement.requirement_id: requirement
        for requirement in trace.plan.fact_requirements
    }
    payload_evidence = {
        item["chunk_id"]: item["text"] for item in payload["evidence"]
    }
    proofs_by_requirement: dict[str, list[EvidenceSupportProof]] = {}
    for proof in submission.support_proofs:
        proofs_by_requirement.setdefault(proof.requirement_id, []).append(proof)
    expected_proof_ids = set(supported_requirement_ids) & set(
        proof_requirement_ids
    )
    if set(proofs_by_requirement) != expected_proof_ids:
        reasons.append(
            "support proofs must cover every selected supported requirement exactly"
        )
    for requirement_id in expected_proof_ids:
        requirement = requirements[requirement_id]
        current_claims = trace.result.requirement_claims.get(requirement_id, [])
        proofs = proofs_by_requirement.get(requirement_id, [])
        proof_claims = [proof.claim for proof in proofs]
        if sorted(proof_claims) != sorted(current_claims):
            reasons.append(
                f"{requirement_id} support proofs must copy every current claim exactly"
            )
        cited_evidence = set(
            trace.result.requirement_evidence.get(requirement_id, [])
        )
        for proof in proofs:
            quote_texts: list[str] = []
            for evidence_quote in proof.evidence_quotes:
                chunk_id = evidence_quote.evidence_chunk_id
                quote = evidence_quote.quote
                if chunk_id not in cited_evidence:
                    reasons.append(
                        f"{requirement_id} support proof uses non-cited evidence {chunk_id}"
                    )
                    continue
                evidence_text = payload_evidence.get(chunk_id)
                if evidence_text is None or not _quote_is_grounded(
                    quote,
                    evidence_text,
                ):
                    reasons.append(
                        f"{requirement_id} support quote is not verbatim cited evidence"
                    )
                    continue
                quote_texts.append(quote)
            if not quote_texts:
                reasons.append(
                    f"{requirement_id} support proof has no valid evidence quote"
                )
                continue
            quote_reference = " ".join(quote_texts)
            unsupported_numbers = sorted(
                value
                for value in _meaningful_numeric_tokens(proof.claim)
                if value not in _supported_numbers(proof.claim, quote_reference)
                and value.removeprefix("-")
                not in _supported_numbers(proof.claim, quote_reference)
                and value.removeprefix("-").removesuffix("%")
                not in {
                    reference_value.removeprefix("-").removesuffix("%")
                    for reference_value in _meaningful_numeric_tokens(
                        quote_reference
                    )
                }
            )
            if unsupported_numbers:
                reasons.append(
                    f"{requirement_id} proof quote misses claim numbers: "
                    + ", ".join(unsupported_numbers)
                )
            proof_reference = " ".join(
                [
                    trace.query,
                    requirement.description,
                    requirement.subject or "",
                    requirement.fact_period or "",
                    quote_reference,
                ]
            )
            coverage, bigram_count = _language_coverage(
                proof.claim,
                proof_reference,
            )
            if (
                bigram_count >= CLAIM_SUPPORT_MIN_BIGRAMS
                and coverage < SUPPORT_PROOF_CHALLENGE_COVERAGE_THRESHOLD
            ):
                challenge_requirement_ids.append(requirement_id)
    return (
        list(dict.fromkeys(reasons)),
        list(dict.fromkeys(challenge_requirement_ids)),
    )


def _decision(findings: list[EvidenceVerificationFinding]) -> Literal[
    "accept", "revise", "abstain"
]:
    if any(
        finding.verdict in {"contradicted", "insufficient_evidence"}
        for finding in findings
    ):
        return "abstain"
    if findings:
        return "revise"
    return "accept"


class EvidenceVerifierAgent:
    """Review one completed extract trace in an isolated evaluator context."""

    def __init__(
        self,
        verifier_model: ToolCallingModel,
        *,
        optimizer_model: ToolCallingModel,
        min_requirements: int = 4,
        known_companies: list[str] | None = None,
        route_policy: Literal["auto", "always"] = "auto",
        enable_open_language_risk: bool = True,
        enable_requirement_contract_risk: bool = True,
        enable_accounting_sign_risk: bool = True,
        require_support_proof: bool = True,
    ) -> None:
        self.verifier_model = verifier_model
        self.optimizer_model = optimizer_model
        self.min_requirements = min_requirements
        self.known_companies = known_companies or []
        self.route_policy = route_policy
        self.enable_open_language_risk = enable_open_language_risk
        self.enable_requirement_contract_risk = enable_requirement_contract_risk
        self.enable_accounting_sign_risk = enable_accounting_sign_risk
        self.require_support_proof = require_support_proof

    def _route(self, trace: AgentTaskTrace) -> tuple[bool, str]:
        if trace.task_type != "extract" or trace.result.outcome != "answer":
            return False, "only completed extraction answers are eligible"
        requirements = trace.plan.fact_requirements
        if self.route_policy == "always":
            return True, "route policy requires verification for every answer"
        if any(
            requirement.evidence_type in {"audit_risk", "audit_response"}
            for requirement in requirements
        ):
            return True, "audit facts require independent completeness review"
        if len(requirements) >= self.min_requirements:
            return True, f"atomic requirement count >= {self.min_requirements}"
        return False, "simple extraction stays on the single-agent fast path"

    def _verify(
        self,
        trace: AgentTaskTrace,
        *,
        stage: Literal[
            "initial_verification",
            "support_proof_retry",
            "post_repair_verification",
        ],
        retry_feedback: list[str] | None = None,
    ) -> tuple[
        EvidenceVerificationTurn,
        list[EvidenceVerificationFinding] | None,
        list[str],
        list[str],
    ]:
        payload = _verification_payload(trace)
        requirement_ids = [
            requirement.requirement_id for requirement in trace.plan.fact_requirements
        ]
        proof_requirement_ids = (
            _support_proof_requirement_ids(trace)
            if self.require_support_proof
            else []
        )
        proof_required = self.require_support_proof and bool(
            proof_requirement_ids
        )
        payload["support_proof_requirement_ids"] = proof_requirement_ids
        if retry_feedback:
            payload["previous_validation_errors"] = retry_feedback
        evidence_ids = [item["chunk_id"] for item in payload["evidence"]]
        try:
            response = self.verifier_model.complete(
                _verification_messages(
                    payload,
                    require_support_proof=proof_required,
                ),
                _verification_tool(
                    requirement_ids,
                    evidence_ids,
                    require_support_proof=proof_required,
                    support_proof_requirement_ids=proof_requirement_ids,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - verifier fails closed
            return (
                EvidenceVerificationTurn(
                    stage=stage,
                    role="verifier",
                    provider=self.verifier_model.provider,
                    model=self.verifier_model.model,
                    endpoint=self.verifier_model.endpoint,
                    decision="error",
                    elapsed_ms=0,
                    validation_errors=[f"{type(exc).__name__}: {exc}"],
                ),
                None,
                [],
                [],
            )
        errors: list[str] = []
        submission: SubmitEvidenceVerification | None = None
        if len(response.tool_calls) != 1:
            errors.append("verifier must make exactly one tool call")
        elif response.tool_calls[0].name != "submit_evidence_verification":
            errors.append(f"unexpected tool: {response.tool_calls[0].name}")
        else:
            try:
                submission = _parse_verification_submission(
                    response.tool_calls[0].arguments
                )
            except (json.JSONDecodeError, ValidationError) as exc:
                errors.append(str(exc))
        manual_review_reasons: list[str] = []
        challenge_requirement_ids: list[str] = []
        if submission is not None:
            supported_findings = [
                finding
                for finding in submission.findings
                if finding.verdict == "supported"
            ]
            actionable_findings = [
                finding
                for finding in submission.findings
                if finding.verdict != "supported"
            ]
            supported = list(
                dict.fromkeys(
                    submission.supported_requirement_ids
                    + [
                        proof.requirement_id
                        for proof in submission.support_proofs
                    ]
                    + [finding.requirement_id for finding in supported_findings]
                )
            )
            actionable_ids = {
                finding.requirement_id for finding in actionable_findings
            }
            supported = [
                requirement_id
                for requirement_id in supported
                if requirement_id not in actionable_ids
            ]
            submission = submission.model_copy(
                update={
                    "supported_requirement_ids": supported,
                    "support_proofs": [
                        proof
                        for proof in submission.support_proofs
                        if proof.requirement_id not in actionable_ids
                    ],
                    "findings": actionable_findings,
                }
            )
            finding_ids = [finding.requirement_id for finding in actionable_findings]
            submitted_ids = supported + finding_ids
            if len(submitted_ids) != len(set(submitted_ids)):
                errors.append("requirement IDs must be unique across the submission")
            if set(submitted_ids) != set(requirement_ids):
                errors.append("verifier must classify every fact requirement exactly once")
            known_evidence = set(evidence_ids)
            for finding in submission.findings:
                if any(
                    chunk_id not in known_evidence
                    for chunk_id in finding.evidence_chunk_ids
                ):
                    errors.append(
                        f"{finding.requirement_id} cites unknown verifier evidence"
                    )
                if (
                    finding.verdict == "incomplete"
                    and not finding.missing_supported_details
                ):
                    errors.append(
                        f"{finding.requirement_id} incomplete finding needs supported details"
                    )
            if proof_required:
                if errors:
                    manual_review_reasons.extend(
                        f"support proof contract invalid: {error}"
                        for error in errors
                    )
                else:
                    (
                        manual_review_reasons,
                        challenge_requirement_ids,
                    ) = _support_proof_review_reasons(
                        trace,
                        payload,
                        submission,
                        supported,
                        proof_requirement_ids,
                    )
        findings = (
            submission.findings
            if submission is not None
            and (not errors or (proof_required and manual_review_reasons))
            else None
        )
        turn = EvidenceVerificationTurn(
            stage=stage,
            role="verifier",
            provider=self.verifier_model.provider,
            model=self.verifier_model.model,
            endpoint=self.verifier_model.endpoint,
            decision=(
                "error"
                if findings is None
                else "manual_review"
                if manual_review_reasons
                else _decision(findings)
            ),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            elapsed_ms=response.elapsed_ms,
            supported_requirement_ids=(
                submission.supported_requirement_ids
                if submission is not None
                and (not errors or manual_review_reasons)
                else []
            ),
            support_proofs=(
                submission.support_proofs
                if submission is not None
                and (not errors or manual_review_reasons)
                else []
            ),
            challenge_requirement_ids=challenge_requirement_ids,
            findings=(findings or []),
            manual_review_reasons=manual_review_reasons,
            validation_errors=errors,
        )
        return (
            turn,
            findings,
            manual_review_reasons,
            challenge_requirement_ids,
        )

    def _challenge_support_proofs(
        self,
        trace: AgentTaskTrace,
        source_turn: EvidenceVerificationTurn,
        requirement_ids: list[str],
    ) -> tuple[
        EvidenceVerificationTurn,
        Literal["accept", "abstain", "manual_review"],
        list[str],
    ]:
        selected = set(requirement_ids)
        requirements = [
            {
                "requirement_id": requirement.requirement_id,
                "description": requirement.description,
                "subject": requirement.subject,
                "fact_period": requirement.fact_period,
                "support_proofs": [
                    proof.model_dump(mode="json")
                    for proof in source_turn.support_proofs
                    if proof.requirement_id == requirement.requirement_id
                ],
            }
            for requirement in trace.plan.fact_requirements
            if requirement.requirement_id in selected
        ]
        evidence_ids = list(
            dict.fromkeys(
                evidence_quote.evidence_chunk_id
                for proof in source_turn.support_proofs
                if proof.requirement_id in selected
                for evidence_quote in proof.evidence_quotes
            )
        )
        verification_payload = _verification_payload(trace)
        evidence_contexts = [
            evidence
            for evidence in verification_payload["evidence"]
            if evidence["chunk_id"] in set(evidence_ids)
        ]
        try:
            response = self.verifier_model.complete(
                _support_proof_challenge_messages(
                    {
                        "requirements": requirements,
                        "evidence_contexts": evidence_contexts,
                    }
                ),
                _verification_tool(
                    requirement_ids,
                    evidence_ids,
                    require_support_proof=False,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - ambiguity escalates safely
            reason = f"support proof challenge failed: {type(exc).__name__}: {exc}"
            return (
                EvidenceVerificationTurn(
                    stage="support_proof_challenge",
                    role="verifier",
                    provider=self.verifier_model.provider,
                    model=self.verifier_model.model,
                    endpoint=self.verifier_model.endpoint,
                    decision="manual_review",
                    elapsed_ms=0,
                    challenge_requirement_ids=requirement_ids,
                    manual_review_reasons=[reason],
                ),
                "manual_review",
                [reason],
            )
        errors: list[str] = []
        submission: SubmitEvidenceVerification | None = None
        if len(response.tool_calls) != 1:
            errors.append("support proof challenge must make exactly one tool call")
        elif response.tool_calls[0].name != "submit_evidence_verification":
            errors.append(
                f"unexpected support proof challenge tool: {response.tool_calls[0].name}"
            )
        else:
            try:
                submission = _parse_verification_submission(
                    response.tool_calls[0].arguments
                )
            except (json.JSONDecodeError, ValidationError) as exc:
                errors.append(str(exc))
        findings: list[EvidenceVerificationFinding] = []
        supported: list[str] = []
        if submission is not None:
            supported_findings = [
                finding
                for finding in submission.findings
                if finding.verdict == "supported"
            ]
            findings = [
                finding
                for finding in submission.findings
                if finding.verdict != "supported"
            ]
            supported = list(
                dict.fromkeys(
                    submission.supported_requirement_ids
                    + [
                        finding.requirement_id
                        for finding in supported_findings
                    ]
                )
            )
            finding_ids = {finding.requirement_id for finding in findings}
            supported = [
                requirement_id
                for requirement_id in supported
                if requirement_id not in finding_ids
            ]
            submitted_ids = supported + [
                finding.requirement_id for finding in findings
            ]
            if len(submitted_ids) != len(set(submitted_ids)):
                errors.append(
                    "support proof challenge requirement IDs must be unique"
                )
            if set(submitted_ids) != selected:
                errors.append(
                    "support proof challenge must classify every flagged requirement"
                )
        reasons = [f"support proof challenge invalid: {error}" for error in errors]
        decision: Literal["accept", "abstain", "manual_review"]
        if errors:
            decision = "manual_review"
        elif any(finding.verdict == "incomplete" for finding in findings):
            decision = "manual_review"
            reasons = [
                "support proof challenge returned incomplete instead of a support decision"
            ]
        elif _decision(findings) == "abstain":
            decision = "abstain"
        else:
            decision = "accept"
        return (
            EvidenceVerificationTurn(
                stage="support_proof_challenge",
                role="verifier",
                provider=self.verifier_model.provider,
                model=self.verifier_model.model,
                endpoint=self.verifier_model.endpoint,
                decision=decision,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                elapsed_ms=response.elapsed_ms,
                supported_requirement_ids=supported,
                challenge_requirement_ids=requirement_ids,
                findings=findings,
                manual_review_reasons=reasons,
                validation_errors=errors,
            ),
            decision,
            reasons,
        )

    def _repair(
        self,
        trace: AgentTaskTrace,
        findings: list[EvidenceVerificationFinding],
    ) -> tuple[EvidenceVerificationTurn, AgentTaskResult | None]:
        payload = _verification_payload(trace)
        requirement_ids = [
            requirement.requirement_id for requirement in trace.plan.fact_requirements
        ]
        try:
            response = self.optimizer_model.complete(
                _repair_messages(payload, findings),
                _submit_extraction_tool(requirement_ids),
            )
        except Exception as exc:  # noqa: BLE001 - repair fails closed
            return (
                EvidenceVerificationTurn(
                    stage="repair",
                    role="optimizer",
                    provider=self.optimizer_model.provider,
                    model=self.optimizer_model.model,
                    endpoint=self.optimizer_model.endpoint,
                    decision="error",
                    elapsed_ms=0,
                    validation_errors=[f"{type(exc).__name__}: {exc}"],
                ),
                None,
            )
        errors: list[str] = []
        result: AgentTaskResult | None = None
        if len(response.tool_calls) != 1:
            errors.append("repair agent must make exactly one tool call")
        elif response.tool_calls[0].name != "submit_extraction":
            errors.append(f"unexpected tool: {response.tool_calls[0].name}")
        else:
            try:
                submission = _parse_extraction_submission(
                    response.tool_calls[0].arguments
                )
            except ValidationError as exc:
                errors.append(str(exc))
            else:
                submission = _repair_submitted_fact_scope_labels(
                    submission, trace.plan
                )
                submission = _repair_extract_authoritative_citations(
                    trace.query,
                    submission,
                    trace.plan,
                    trace.evidence_memory,
                )
                result, result_errors = DeepSeekExtractAgent._result(
                    submission,
                    trace.plan,
                    trace.evidence_memory,
                )
                errors.extend(result_errors)
        if errors:
            result = None
        return (
            EvidenceVerificationTurn(
                stage="repair",
                role="optimizer",
                provider=self.optimizer_model.provider,
                model=self.optimizer_model.model,
                endpoint=self.optimizer_model.endpoint,
                decision=(
                    "error"
                    if result is None
                    else "accept"
                    if result.outcome == "answer"
                    else "abstain"
                ),
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                elapsed_ms=response.elapsed_ms,
                validation_errors=errors,
            ),
            result,
        )

    @staticmethod
    def _abstain(
        trace: AgentTaskTrace,
        verification: EvidenceVerificationTrace,
        *,
        error: bool,
    ) -> AgentTaskTrace:
        return trace.model_copy(
            update={
                "status": "failed" if error else "completed",
                "stop_reason": (
                    "evidence_verifier_error"
                    if error
                    else "evidence_verifier_rejected"
                ),
                "completed_at": datetime.now(UTC),
                "sufficiency": SufficiencyDecision(
                    status="incomplete",
                    evidence_count_by_target={
                        "task:extract": len(trace.evidence_memory.items)
                    },
                    gaps=["independent evidence verification"],
                    requirement_gaps=[
                        finding.requirement_id
                        for turn in verification.turns
                        for finding in turn.findings
                    ],
                ),
                "result": AgentTaskResult(
                    outcome="abstain",
                    answer=GeneratedAnswer(
                        answer=(
                            "独立证据审计调用失败，系统已安全停止。"
                            if error
                            else "独立证据审计未通过，系统拒绝输出未经充分支持的结果。"
                        ),
                        citations=[],
                        provider="evidence-verifier-gate",
                        grounded=False,
                    ),
                    target_evidence={},
                ),
                "evidence_verification": verification,
            }
        )

    @staticmethod
    def _risk_gate_abstain(
        trace: AgentTaskTrace,
        risk_gate: ClaimRiskGateTrace,
        verification: EvidenceVerificationTrace,
    ) -> AgentTaskTrace:
        return trace.model_copy(
            update={
                "status": "completed",
                "stop_reason": "claim_risk_gate_rejected",
                "completed_at": datetime.now(UTC),
                "sufficiency": SufficiencyDecision(
                    status="incomplete",
                    evidence_count_by_target={
                        "task:extract": len(trace.evidence_memory.items)
                    },
                    gaps=["deterministic claim risk gate"],
                    requirement_gaps=list(
                        dict.fromkeys(
                            finding.requirement_id
                            for finding in risk_gate.findings
                        )
                    ),
                ),
                "result": AgentTaskResult(
                    outcome="abstain",
                    answer=GeneratedAnswer(
                        answer=(
                            "本地主体、期间、数值、单位或引用范围门禁发现明确冲突，"
                            "系统拒绝输出该结果。"
                        ),
                        citations=[],
                        provider="claim-risk-gate",
                        grounded=False,
                    ),
                    target_evidence={},
                ),
                "claim_risk_gate": risk_gate,
                "evidence_verification": verification,
            }
        )

    @staticmethod
    def _manual_review(
        trace: AgentTaskTrace,
        verification: EvidenceVerificationTrace,
        reasons: list[str],
    ) -> AgentTaskTrace:
        verification.human_review_required = True
        verification.human_review_reasons = list(dict.fromkeys(reasons))
        verification.final_decision = "manual_review"
        verification.candidate_result = trace.result.model_copy(deep=True)
        return trace.model_copy(
            update={
                "status": "completed",
                "stop_reason": "evidence_verifier_manual_review",
                "completed_at": datetime.now(UTC),
                "sufficiency": SufficiencyDecision(
                    status="incomplete",
                    evidence_count_by_target={
                        "task:extract": len(trace.evidence_memory.items)
                    },
                    gaps=["verifiable evidence support proof"],
                    requirement_gaps=list(
                        dict.fromkeys(
                            proof.requirement_id
                            for turn in verification.turns
                            for proof in turn.support_proofs
                        )
                    ),
                ),
                "result": AgentTaskResult(
                    outcome="abstain",
                    answer=GeneratedAnswer(
                        answer=(
                            "证据复核未能提供可独立验证的逐项支持证明，"
                            "系统已暂停自动回答并升级人工复核。"
                        ),
                        citations=[],
                        provider="evidence-support-proof-gate",
                        grounded=False,
                    ),
                    target_evidence={},
                ),
                "evidence_verification": verification,
            }
        )

    def review(self, trace: AgentTaskTrace) -> AgentTaskTrace:
        risk_gate = evaluate_claim_risk_gate(
            trace,
            known_companies=self.known_companies,
            enable_open_language_risk=self.enable_open_language_risk,
            enable_requirement_contract_risk=(
                self.enable_requirement_contract_risk
            ),
            enable_accounting_sign_risk=self.enable_accounting_sign_risk,
        )
        trace = trace.model_copy(update={"claim_risk_gate": risk_gate})
        routed, route_reason = self._route(trace)
        if risk_gate.status == "review":
            routed = True
            route_reason = (
                "deterministic open claim/evidence risk warning requires model verification"
            )
        proof_required = self.require_support_proof and bool(
            _support_proof_requirement_ids(trace)
        )
        verification = EvidenceVerificationTrace(
            prompt_revision=_prompt_revision(
                require_support_proof=proof_required
            ),
            prompt_sha256=_prompt_sha256(
                require_support_proof=proof_required
            ),
            routed=routed,
            route_reason=route_reason,
            final_decision="not_routed",
        )
        if risk_gate.status == "reject":
            verification.route_reason = (
                "deterministic claim risk gate rejected before model verification"
            )
            return self._risk_gate_abstain(trace, risk_gate, verification)
        if not routed:
            return trace.model_copy(update={"evidence_verification": verification})
        (
            first_turn,
            findings,
            manual_review_reasons,
            challenge_requirement_ids,
        ) = self._verify(trace, stage="initial_verification")
        verification.turns.append(first_turn)
        if findings is None:
            verification.final_decision = "error"
            return self._abstain(trace, verification, error=True)
        if manual_review_reasons and all(
            reason.startswith("support proof contract invalid:")
            for reason in manual_review_reasons
        ):
            verification.verification_retry_attempted = True
            (
                first_turn,
                findings,
                manual_review_reasons,
                challenge_requirement_ids,
            ) = self._verify(
                trace,
                stage="support_proof_retry",
                retry_feedback=verification.turns[-1].validation_errors,
            )
            verification.turns.append(first_turn)
            if findings is None:
                verification.final_decision = "error"
                return self._abstain(trace, verification, error=True)
        if manual_review_reasons:
            return self._manual_review(
                trace,
                verification,
                manual_review_reasons,
            )
        if challenge_requirement_ids:
            challenge_turn, challenge_decision, challenge_reasons = (
                self._challenge_support_proofs(
                    trace,
                    first_turn,
                    challenge_requirement_ids,
                )
            )
            verification.turns.append(challenge_turn)
            if challenge_decision == "manual_review":
                return self._manual_review(
                    trace,
                    verification,
                    challenge_reasons,
                )
            if challenge_decision == "abstain":
                verification.final_decision = "abstain"
                return self._abstain(trace, verification, error=False)
        initial_decision = _decision(findings)
        if initial_decision == "accept":
            verification.final_decision = "accept_original"
            return trace.model_copy(update={"evidence_verification": verification})
        if initial_decision == "abstain":
            verification.final_decision = "abstain"
            return self._abstain(trace, verification, error=False)

        verification.repair_attempted = True
        repair_turn, repaired_result = self._repair(trace, findings)
        verification.turns.append(repair_turn)
        if repaired_result is None:
            verification.final_decision = "error"
            return self._abstain(trace, verification, error=True)
        if repaired_result.outcome != "answer":
            verification.final_decision = "abstain"
            return self._abstain(trace, verification, error=False)
        repaired_trace = trace.model_copy(update={"result": repaired_result})
        repaired_risk_gate = evaluate_claim_risk_gate(
            repaired_trace,
            known_companies=self.known_companies,
            enable_open_language_risk=self.enable_open_language_risk,
            enable_requirement_contract_risk=(
                self.enable_requirement_contract_risk
            ),
            enable_accounting_sign_risk=self.enable_accounting_sign_risk,
        )
        repaired_trace = repaired_trace.model_copy(
            update={"claim_risk_gate": repaired_risk_gate}
        )
        if repaired_risk_gate.status == "reject":
            verification.final_decision = "abstain"
            return self._risk_gate_abstain(
                repaired_trace,
                repaired_risk_gate,
                verification,
            )
        (
            final_turn,
            final_findings,
            final_manual_review_reasons,
            final_challenge_requirement_ids,
        ) = self._verify(
            repaired_trace, stage="post_repair_verification"
        )
        verification.turns.append(final_turn)
        if final_findings is None:
            verification.final_decision = "error"
            return self._abstain(trace, verification, error=True)
        if (
            final_manual_review_reasons
            and not verification.verification_retry_attempted
            and all(
                reason.startswith("support proof contract invalid:")
                for reason in final_manual_review_reasons
            )
        ):
            verification.verification_retry_attempted = True
            (
                final_turn,
                final_findings,
                final_manual_review_reasons,
                final_challenge_requirement_ids,
            ) = self._verify(
                repaired_trace,
                stage="support_proof_retry",
                retry_feedback=verification.turns[-1].validation_errors,
            )
            verification.turns.append(final_turn)
            if final_findings is None:
                verification.final_decision = "error"
                return self._abstain(trace, verification, error=True)
        if final_manual_review_reasons:
            return self._manual_review(
                repaired_trace,
                verification,
                final_manual_review_reasons,
            )
        if final_challenge_requirement_ids:
            challenge_turn, challenge_decision, challenge_reasons = (
                self._challenge_support_proofs(
                    repaired_trace,
                    final_turn,
                    final_challenge_requirement_ids,
                )
            )
            verification.turns.append(challenge_turn)
            if challenge_decision == "manual_review":
                return self._manual_review(
                    repaired_trace,
                    verification,
                    challenge_reasons,
                )
            if challenge_decision == "abstain":
                verification.final_decision = "abstain"
                return self._abstain(
                    repaired_trace,
                    verification,
                    error=False,
                )
        if _decision(final_findings) != "accept":
            verification.final_decision = "abstain"
            return self._abstain(trace, verification, error=False)
        verification.final_decision = "accept_repaired"
        return repaired_trace.model_copy(
            update={
                "completed_at": datetime.now(UTC),
                "evidence_verification": verification,
            }
        )


class EvidenceVerifiedExtractAgent:
    """Production wrapper: base extraction followed by conditional verification."""

    def __init__(
        self,
        base_agent: DeepSeekExtractAgent,
        verifier: EvidenceVerifierAgent,
    ) -> None:
        self.base_agent = base_agent
        self.verifier = verifier

    def run(self, request: AgentTaskRequest) -> AgentTaskTrace:
        return self.verifier.review(self.base_agent.run(request))
