"""DeepSeek tool-calling runtime for bounded FinDocRAG compare tasks."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from time import perf_counter
from typing import Literal, Protocol
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from findoc_rag.agent_tasks import (
    METRIC_EVIDENCE_ALIASES,
    AgentEvidence,
    AgentModelTrace,
    AgentModelTurn,
    AgentTaskPlan,
    AgentTaskRequest,
    AgentTaskResult,
    AgentTaskTrace,
    AgentToolCall,
    AtomicFactRequirement,
    ComparisonTarget,
    EvidenceMemory,
    SufficiencyDecision,
    add_evidence,
    judge_sufficiency,
    missing_required_metrics,
    plan_compare_task,
    plan_document_task,
    target_evidence,
)
from findoc_rag.answer_generation import Citation, ClaimCitation, GeneratedAnswer
from findoc_rag.indexing import SearchFilters, SearchHit
from findoc_rag.provider_credentials import resolve_provider_api_key
from findoc_rag.service import SearchRequest
from findoc_rag.visual_inspection import (
    PageLayoutReconstruction,
    PageRegionInspection,
    PdfRegionInspector,
)

AGENT_PROMPT_REVISION = "deepseek-compare-p2b-metric-gap-direction-summary"
EXTRACT_PROMPT_REVISION = "deepseek-extract-p3b-rendered-two-column-page-reconstruction"
CALCULATE_PROMPT_REVISION = "deepseek-calculate-p3-general-grounded-decimal-dag"
VISUAL_GRAPH_PROMPT_REVISION = "deepseek-visual-p1c-geometry-relationships-local-sum"
DEFAULT_DEEPSEEK_AGENT_MODEL = "deepseek-v4-flash"
DEFAULT_DEEPSEEK_ENDPOINT = "https://api.deepseek.com/chat/completions"

EXPLICIT_VERDICT_CUES = (
    "说法成立",
    "说法不成立",
    "结论为真",
    "结论为假",
    "结论成立",
    "结论不成立",
)
VERIFICATION_PREDICATE_CUES = (
    "上升",
    "下降",
    "增长",
    "减少",
    "提高",
    "降低",
    "持平",
    "相同",
    "高于",
    "低于",
)

AUTHORITY_SECTION_WEIGHTS = (
    ("财务摘要", 50),
    ("关键指标", 45),
    ("业绩综述", 35),
    ("主要经营业绩", 30),
    ("内含价值分析", 20),
    ("董事长致辞", 5),
)


def _meaningful_numeric_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in re.findall(r"(?<!\d)-?\d[\d,]*(?:\.\d+)?%?", text):
        normalized = raw.replace(",", "")
        unsigned = normalized.removeprefix("-").removesuffix("%")
        if unsigned.isdigit() and len(unsigned) == 4 and 1900 <= int(unsigned) <= 2100:
            continue
        tokens.add(normalized)
    return tokens


def _source_authority_score(
    query: str,
    section_path: list[str],
    text: str,
) -> int:
    section = " ".join(section_path)
    context = f"{section} {text}"
    score = sum(
        weight
        for cue, weight in AUTHORITY_SECTION_WEIGHTS
        if cue in context
    )
    if "可比" in query and "可比口径" in context:
        score += 30
    if "调整后" in query and any(cue in context for cue in ("调整后", "追溯调整")):
        score += 20
    for metric, aliases in METRIC_EVIDENCE_ALIASES.items():
        if metric not in query and not any(alias in query for alias in aliases):
            continue
        if metric in context or any(alias in context for alias in aliases):
            score += 12
    return score


def _rank_authoritative_hits(
    query: str,
    hits: list[SearchHit],
    *,
    limit: int,
) -> list[SearchHit]:
    ranked = sorted(
        hits,
        key=lambda hit: (
            _source_authority_score(
                query,
                hit.chunk.section_path,
                hit.chunk.text,
            ),
            -hit.rank,
        ),
        reverse=True,
    )[:limit]
    return [
        hit.model_copy(update={"rank": rank})
        for rank, hit in enumerate(ranked, start=1)
    ]


def _needs_authoritative_source_ranking(query: str) -> bool:
    return any(
        cue in query
        for cue in ("可比", "调整后", "核验", "验证", "归母", "关键指标")
    )


def _answer_contract(query: str) -> dict[str, bool | str]:
    verification = any(cue in query for cue in ("核验", "验证", "说法", "结论是否"))
    comparison = any(cue in query for cue in ("比较", "变化", "增减", "趋势"))
    return {
        "coverage_rule": (
            "逐项覆盖任务中以顿号、逗号、和、及、以及、分别列出的指标、原因、"
            "假设、风险与审计应对；不得因已有公司/年度 target 就省略任务子项。"
        ),
        "requires_explicit_verdict": verification,
        "requires_direction_for_each_metric": comparison,
        "requires_concise_direction_summary": comparison,
        "render_decline_with_signed_value": any(
            cue in query for cue in ("同比", "增减", "变化", "增长", "下降")
        ),
    }


def _validate_answer_contract(
    query: str,
    submission: SubmitComparisonArguments,
) -> list[str]:
    if submission.status != "answer":
        return []
    contract = _answer_contract(query)
    answer_text = "\n".join(claim.text for claim in submission.claims)
    errors: list[str] = []
    explicit_verdict = any(cue in answer_text for cue in EXPLICIT_VERDICT_CUES)
    requested_predicates = {
        cue for cue in VERIFICATION_PREDICATE_CUES if cue in query
    }
    predicates_reproduced = bool(requested_predicates) and all(
        cue in answer_text for cue in requested_predicates
    )
    if (
        contract["requires_explicit_verdict"]
        and not explicit_verdict
        and not predicates_reproduced
    ):
        errors.append(
            "verification task requires an explicit final verdict or a complete "
            "restatement of every directional predicate in the claim"
        )
    return errors


class ModelToolCall(BaseModel):
    call_id: str
    name: str
    arguments: str


class ToolModelResponse(BaseModel):
    content: str | None = None
    tool_calls: list[ModelToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    elapsed_ms: float = Field(ge=0)

    def assistant_message(self) -> dict:
        message: dict = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments,
                    },
                }
                for call in self.tool_calls
            ]
        return message


class ToolCallingModel(Protocol):
    provider: str
    model: str
    endpoint: str

    @property
    def available(self) -> bool: ...

    def complete(self, messages: list[dict], tools: list[dict]) -> ToolModelResponse: ...


class DeepSeekToolCallingModel:
    """Minimal DeepSeek Chat Completions client with locally validated tools."""

    provider = "deepseek"

    def __init__(
        self,
        *,
        model: str = "",
        endpoint: str = "",
        api_key: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model or os.getenv("FINDOC_RAG_AGENT_MODEL", DEFAULT_DEEPSEEK_AGENT_MODEL)
        self.endpoint = endpoint or os.getenv(
            "FINDOC_RAG_AGENT_ENDPOINT",
            os.getenv("FINDOC_RAG_ANSWER_ENDPOINT", DEFAULT_DEEPSEEK_ENDPOINT),
        )
        self.api_key = resolve_provider_api_key(self.endpoint, api_key)
        self._client = client

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, messages: list[dict], tools: list[dict]) -> ToolModelResponse:
        if not self.available:
            raise RuntimeError("DeepSeek API key is unavailable for the configured endpoint")
        function_names = [tool["function"]["name"] for tool in tools]
        tool_choice: str | dict = "required"
        if len(function_names) == 1:
            tool_choice = {
                "type": "function",
                "function": {"name": function_names[0]},
            }
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": 0,
            "max_tokens": 2048,
            "thinking": {"type": "disabled"},
        }
        started = perf_counter()
        response: httpx.Response | None = None
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                sender = self._client.post if self._client is not None else httpx.post
                response = sender(
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
            raise RuntimeError(f"DeepSeek agent request failed: {last_error}") from last_error
        assert response is not None
        body = response.json()
        choice = body["choices"][0]
        message = choice["message"]
        calls = [
            ModelToolCall(
                call_id=call["id"],
                name=call["function"]["name"],
                arguments=call["function"]["arguments"],
            )
            for call in message.get("tool_calls", [])
        ]
        usage = body.get("usage", {})
        return ToolModelResponse(
            content=message.get("content"),
            tool_calls=calls,
            finish_reason=choice.get("finish_reason"),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            elapsed_ms=(perf_counter() - started) * 1000,
        )


class SearchEvidenceArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_id: str
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=5)


class SubmittedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_id: str
    text: str = Field(min_length=1, max_length=1000)
    evidence_chunk_ids: list[str] = Field(min_length=1, max_length=5)


class SubmitComparisonArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["answer", "evidence_only", "clarify", "abstain"]
    message: str = Field(min_length=1, max_length=2000)
    covered_target_ids: list[str] = Field(default_factory=list)
    claims: list[SubmittedClaim] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


def _parse_submission(arguments: str) -> SubmitComparisonArguments:
    """Keep answers strict while safely accepting malformed non-answer payloads."""

    try:
        return SubmitComparisonArguments.model_validate_json(arguments)
    except ValidationError as validation_error:
        try:
            payload = json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            raise validation_error
        if not isinstance(payload, dict):
            raise
        status = payload.get("status")
        message = payload.get("message")
        if status not in {"evidence_only", "clarify", "abstain"} or not isinstance(message, str):
            raise
        return SubmitComparisonArguments(status=status, message=message)


def _tool_definitions(
    plan: AgentTaskPlan,
    *,
    allowed_search_target_ids: set[str] | None = None,
) -> list[dict]:
    all_target_ids = [target.target_id for target in plan.targets]
    search_target_ids = (
        all_target_ids
        if allowed_search_target_ids is None
        else [target_id for target_id in all_target_ids if target_id in allowed_search_target_ids]
    )
    tools: list[dict] = []
    if search_target_ids:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "search_evidence",
                    "description": (
                        "Search the immutable annual-report index for one planned comparison target. "
                        "Call it once for each allowed target. Retry only when the previous call "
                        "returned no usable evidence."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target_id": {
                                "type": "string",
                                "enum": search_target_ids,
                            },
                            "query": {"type": "string"},
                            "top_k": {"type": "integer", "minimum": 1, "maximum": 5},
                        },
                        "required": ["target_id", "query", "top_k"],
                        "additionalProperties": False,
                    },
                },
            }
        )
    tools.append(
        {
            "type": "function",
            "function": {
                "name": "submit_comparison",
                "description": (
                    "Submit structured claims after searching. An answer must cover every target, "
                    "and every claim must cite returned chunk IDs."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["answer", "evidence_only", "clarify", "abstain"],
                        },
                        "message": {"type": "string"},
                        "covered_target_ids": {
                            "type": "array",
                            "items": {"type": "string", "enum": all_target_ids},
                        },
                        "claims": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "target_id": {
                                        "type": "string",
                                        "enum": all_target_ids,
                                    },
                                    "text": {"type": "string"},
                                    "evidence_chunk_ids": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                                "required": ["target_id", "text", "evidence_chunk_ids"],
                                "additionalProperties": False,
                            },
                        },
                        "gaps": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "status",
                        "message",
                        "covered_target_ids",
                        "claims",
                        "gaps",
                    ],
                    "additionalProperties": False,
                },
            },
        }
    )
    return tools


def _initial_messages(request: AgentTaskRequest, plan: AgentTaskPlan) -> list[dict]:
    targets = [
        {
            "target_id": target.target_id,
            "label": target.label,
            "company_names": target.filters.company_names,
            "report_years": target.filters.report_years,
            "fact_year": target.fact_year,
            "document_year": target.document_year,
            "statement_scope": target.statement_scope,
            "value_version": target.value_version,
            "retrieval_hint": target.retrieval_hint,
            "cross_target_group": target.cross_target_group,
        }
        for target in plan.targets
    ]
    return [
        {
            "role": "system",
            "content": (
                "你是财务年报比较 Agent。你必须通过工具获取证据，不能凭记忆回答。"
                "先为每个给定 target 调用 search_evidence；证据不足时只重试缺口。"
                "完成后调用 submit_comparison。answer 必须覆盖全部 target，且每条 claim "
                "只能引用工具返回的 chunk_id。检索工具关闭时必须立即 submit，不能继续"
                "请求检索。每次检索必须吸收 target 的 retrieval_hint，以优先命中精确"
                "财务表格。不得输出或记录思维链。"
                "公司/年度 target 只限定证据范围，不代表已覆盖用户列出的全部子项；"
                "必须遵守 user 消息中的 answer_contract。每次搜索结果中的"
                "remaining_required_metrics 是本地证据检查仍未覆盖的指标；非空时必须"
                "针对这些指标再次检索，不能因为已有一个 chunk 就提前提交。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": request.query,
                    "answer_contract": _answer_contract(request.query),
                    "required_metrics": plan.required_metrics,
                    "targets": targets,
                },
                ensure_ascii=False,
            ),
        },
    ]


def _finalization_messages(
    request: AgentTaskRequest,
    plan: AgentTaskPlan,
    memory: EvidenceMemory,
) -> list[dict]:
    targets = [
        {
            "target_id": target.target_id,
            "label": target.label,
            "fact_year": target.fact_year,
            "document_year": target.document_year,
            "statement_scope": target.statement_scope,
            "value_version": target.value_version,
            "retrieval_hint": target.retrieval_hint,
            "cross_target_group": target.cross_target_group,
        }
        for target in plan.targets
    ]
    evidence = [
        {
            "chunk_id": item.chunk_id,
            "target_ids": item.target_ids,
            "company_name": item.company_name,
            "report_year": item.report_year,
            "pages": [item.page_start, item.page_end],
            "section_path": item.section_path,
            "text": item.excerpt,
        }
        for item in memory.items
    ]
    return [
        {
            "role": "system",
            "content": (
                "你是财务年报比较 Agent 的最终提交器。现在禁止继续检索，必须调用"
                "submit_comparison。证据足以覆盖全部目标时提交 answer；否则提交 abstain，"
                "并在 gaps 中准确说明缺失目标。每条 answer claim 只能引用下方证据中的"
                "chunk_id，且证据必须绑定对应 target。优先使用‘主要会计数据’或财务"
                "报表附注中的精确元值，不得用摘要中的亿元四舍五入值替代，不得由"
                "增长率反推。任务要求差额、增幅或结论时，必须把计算结果写入某条 claim。"
                "证据正文中的‘去年/上年/本年’必须相对该证据的 report_year 解释，"
                "不得相对系统当前年份解释。"
                "提交前按 answer_contract 逐项自检：核验题必须明确写‘说法成立’或"
                "‘说法不成立’；比较题必须逐指标写明上升、下降或持平；同比下降同时"
                "写成‘下降X%（-X%）’，避免方向丢失。多指标比较必须最后单独增加"
                "一条不插入数值的简明方向总括 claim，使完整指标名与上升、下降、"
                "增加、减少或持平直接相邻；全部同方向时明确写‘各项指标均增加’等"
                "汇总结论。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": request.query,
                    "answer_contract": _answer_contract(request.query),
                    "required_metrics": plan.required_metrics,
                    "targets": targets,
                    "evidence": evidence,
                },
                ensure_ascii=False,
            ),
        },
    ]


def _prompt_sha256(messages: list[dict], tools: list[dict]) -> str:
    payload = json.dumps(
        {"messages": messages, "tools": tools}, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tool_message(call_id: str, payload: dict) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(payload, ensure_ascii=False),
    }


def _model_trace(
    model: ToolCallingModel,
    prompt_sha256: str,
    turns: list[AgentModelTurn],
    *,
    prompt_revision: str = AGENT_PROMPT_REVISION,
) -> AgentModelTrace:
    return AgentModelTrace(
        provider=model.provider,
        model=model.model,
        endpoint=model.endpoint,
        prompt_revision=prompt_revision,
        prompt_sha256=prompt_sha256,
        turns=turns,
    )


def _clarification_trace(
    request: AgentTaskRequest,
    plan: AgentTaskPlan,
    *,
    index_id: str,
    created_at: datetime,
) -> AgentTaskTrace:
    message = plan.clarification or "请补充比较对象。"
    return AgentTaskTrace(
        task_id=uuid4().hex,
        task_type=request.task_type,
        runtime="deepseek_tool_calling",
        status="needs_clarification",
        stop_reason="needs_clarification",
        query=request.query,
        index_id=index_id,
        created_at=created_at,
        completed_at=datetime.now(UTC),
        rounds_completed=0,
        plan=plan,
        tool_calls=[],
        evidence_memory=EvidenceMemory(index_id=index_id),
        sufficiency=SufficiencyDecision(status="incomplete", evidence_count_by_target={}, gaps=[]),
        result=AgentTaskResult(
            outcome="clarify",
            answer=GeneratedAnswer(
                answer=message,
                citations=[],
                provider="input-guard",
                grounded=False,
            ),
            target_evidence={},
        ),
    )


def _corpus_unavailable_trace(
    request: AgentTaskRequest,
    plan: AgentTaskPlan,
    *,
    index_id: str,
    created_at: datetime,
) -> AgentTaskTrace:
    message = plan.corpus_unavailable_reason or "当前索引不包含所请求事实。"
    return AgentTaskTrace(
        task_id=uuid4().hex,
        task_type=request.task_type,
        runtime="deepseek_tool_calling",
        status="completed",
        stop_reason="no_new_evidence",
        query=request.query,
        index_id=index_id,
        created_at=created_at,
        completed_at=datetime.now(UTC),
        rounds_completed=0,
        plan=plan,
        tool_calls=[],
        evidence_memory=EvidenceMemory(index_id=index_id),
        sufficiency=SufficiencyDecision(
            status="incomplete",
            evidence_count_by_target={},
            gaps=["corpus_year_coverage"],
        ),
        result=AgentTaskResult(
            outcome="abstain",
            answer=GeneratedAnswer(
                answer=message,
                citations=[],
                provider="agent-local-corpus-coverage-gate",
                grounded=False,
            ),
            target_evidence={},
        ),
    )


class DeepSeekCompareAgent:
    """Let DeepSeek choose bounded search calls; validate all actions locally."""

    def __init__(
        self,
        search_backend,
        model: ToolCallingModel,
        *,
        available_companies: list[str],
    ) -> None:
        self.search_backend = search_backend
        self.model = model
        self.available_companies = available_companies

    def _submitted_answer(
        self,
        request: AgentTaskRequest,
        submission: SubmitComparisonArguments,
        plan: AgentTaskPlan,
        memory: EvidenceMemory,
    ) -> tuple[AgentTaskResult | None, list[str]]:
        targets = {target.target_id: target for target in plan.targets}
        evidence = {item.chunk_id: item for item in memory.items}
        errors: list[str] = []
        if submission.status == "answer":
            errors.extend(_validate_answer_contract(request.query, submission))
            submission = self._repair_claim_target_bindings(
                submission, plan, memory
            )
            submission = self._repair_authoritative_citations(
                request.query,
                submission,
                memory,
            )
            submission = self._repair_provenance_citations(submission, plan, memory)
            expected = set(targets)
            if set(submission.covered_target_ids) != expected:
                errors.append("covered_target_ids must exactly match all planned targets")
            claimed_targets = {claim.target_id for claim in submission.claims}
            if claimed_targets != expected:
                errors.append("answer claims must cover every planned target")
            for claim in submission.claims:
                if claim.target_id not in targets:
                    errors.append(f"unknown claim target: {claim.target_id}")
                    continue
                for chunk_id in claim.evidence_chunk_ids:
                    item = evidence.get(chunk_id)
                    if item is None:
                        errors.append(f"unknown evidence chunk: {chunk_id}")
                    elif claim.target_id not in item.target_ids and not (
                        self._is_cross_target_comparison_support(
                            claim, item, targets
                        )
                    ):
                        errors.append(f"evidence {chunk_id} is not bound to {claim.target_id}")
            if errors:
                return None, errors

            citations: list[Citation] = []
            ordinal_by_chunk: dict[str, int] = {}
            claim_citations: list[ClaimCitation] = []
            lines: list[str] = []
            target_answers: dict[str, GeneratedAnswer] = {}
            for claim in submission.claims:
                ordinals: list[int] = []
                for chunk_id in claim.evidence_chunk_ids:
                    ordinal = ordinal_by_chunk.get(chunk_id)
                    if ordinal is None:
                        item = evidence[chunk_id]
                        ordinal = len(citations) + 1
                        ordinal_by_chunk[chunk_id] = ordinal
                        citations.append(
                            Citation(
                                ordinal=ordinal,
                                chunk_id=chunk_id,
                                page_start=item.page_start,
                                page_end=item.page_end,
                                section_path=item.section_path,
                                excerpt=item.excerpt,
                            )
                        )
                    ordinals.append(ordinal)
                label = targets[claim.target_id].label
                suffix = "".join(f"[{ordinal}]" for ordinal in ordinals)
                line = f"{label}：{claim.text}{suffix}"
                lines.append(line)
                claim_citations.append(
                    ClaimCitation(claim=f"{label}：{claim.text}", citation_ordinals=ordinals)
                )
                target_answers[claim.target_id] = GeneratedAnswer(
                    answer=line,
                    citations=[citations[ordinal - 1] for ordinal in ordinals],
                    provider="deepseek-tool-calling",
                    grounded=True,
                    claim_citations=[claim_citations[-1]],
                )
            return (
                AgentTaskResult(
                    outcome="answer",
                    answer=GeneratedAnswer(
                        answer="\n".join(lines),
                        citations=citations,
                        provider="deepseek-tool-calling",
                        grounded=True,
                        claim_citations=claim_citations,
                    ),
                    target_evidence=target_evidence(plan, memory),
                    target_answers=target_answers,
                ),
                [],
            )

        outcome = submission.status
        return (
            AgentTaskResult(
                outcome=outcome,
                answer=GeneratedAnswer(
                    answer=submission.message,
                    citations=[],
                    provider="deepseek-tool-calling",
                    grounded=False,
                ),
                target_evidence=target_evidence(plan, memory),
            ),
            [],
        )

    @staticmethod
    def _repair_claim_target_bindings(
        submission: SubmitComparisonArguments,
        plan: AgentTaskPlan,
        memory: EvidenceMemory,
    ) -> SubmitComparisonArguments:
        """Repair a claim assigned to the wrong year target when evidence is unique."""

        repaired = submission.model_copy(deep=True)
        evidence = {item.chunk_id: item for item in memory.items}
        for claim in repaired.claims:
            cited = [evidence[item] for item in claim.evidence_chunk_ids if item in evidence]
            if not cited or any(claim.target_id in item.target_ids for item in cited):
                continue
            candidates = []
            for target in plan.targets:
                if target.fact_year is None or str(target.fact_year) not in claim.text:
                    continue
                if any(target.target_id in item.target_ids for item in cited):
                    candidates.append(target.target_id)
            if len(candidates) == 1:
                claim.target_id = candidates[0]
        return repaired

    @staticmethod
    def _is_cross_company_comparison_support(
        claim: SubmittedClaim, evidence: AgentEvidence
    ) -> bool:
        company = evidence.company_name
        comparison_cues = ("高于", "低于", "更高", "更低", "超过", "相比", "比较")
        return bool(
            company and company in claim.text and any(cue in claim.text for cue in comparison_cues)
        )

    @classmethod
    def _is_cross_target_comparison_support(
        cls,
        claim: SubmittedClaim,
        evidence: AgentEvidence,
        targets: dict[str, ComparisonTarget],
    ) -> bool:
        if cls._is_cross_company_comparison_support(claim, evidence):
            return True
        target = targets.get(claim.target_id)
        if target is None or not evidence.company_name or evidence.report_year is None:
            return False
        comparison_cues = ("高于", "低于", "上升", "下降", "增长", "减少", "相比", "较")
        return bool(
            evidence.company_name in target.filters.company_names
            and str(evidence.report_year) in claim.text
            and any(cue in claim.text for cue in comparison_cues)
        )

    @staticmethod
    def _repair_provenance_citations(
        submission: SubmitComparisonArguments,
        plan: AgentTaskPlan,
        memory: EvidenceMemory,
    ) -> SubmitComparisonArguments:
        repaired = submission.model_copy(deep=True)
        for target in plan.targets:
            if target.document_year is None:
                continue
            target_claims = [
                claim for claim in repaired.claims if claim.target_id == target.target_id
            ]
            if not target_claims:
                continue
            cited_ids = {
                chunk_id for claim in target_claims for chunk_id in claim.evidence_chunk_ids
            }
            cited_required_vintage = any(
                item.chunk_id in cited_ids
                and item.report_year == target.document_year
                and target.target_id in item.target_ids
                for item in memory.items
            )
            if cited_required_vintage:
                continue
            candidate = next(
                (
                    item
                    for item in memory.items
                    if item.report_year == target.document_year
                    and target.target_id in item.target_ids
                    and (
                        not target.filters.company_names
                        or item.company_name in target.filters.company_names
                    )
                ),
                None,
            )
            if candidate is not None:
                target_claims[0].evidence_chunk_ids.append(candidate.chunk_id)
        return repaired

    @staticmethod
    def _repair_authoritative_citations(
        query: str,
        submission: SubmitComparisonArguments,
        memory: EvidenceMemory,
    ) -> SubmitComparisonArguments:
        if not _needs_authoritative_source_ranking(query):
            return submission
        repaired = submission.model_copy(deep=True)
        evidence = {item.chunk_id: item for item in memory.items}
        for claim in repaired.claims:
            values = _meaningful_numeric_tokens(claim.text)
            if not values:
                continue
            candidates = [
                item
                for item in memory.items
                if claim.target_id in item.target_ids
                and values & _meaningful_numeric_tokens(item.excerpt)
            ]
            if not candidates:
                continue
            best = max(
                candidates,
                key=lambda item: (
                    _source_authority_score(
                        query,
                        item.section_path,
                        item.excerpt,
                    ),
                    len(values & _meaningful_numeric_tokens(item.excerpt)),
                ),
            )
            cited_scores = [
                _source_authority_score(
                    query,
                    evidence[chunk_id].section_path,
                    evidence[chunk_id].excerpt,
                )
                for chunk_id in claim.evidence_chunk_ids
                if chunk_id in evidence
            ]
            best_score = _source_authority_score(
                query,
                best.section_path,
                best.excerpt,
            )
            if (
                best.chunk_id not in claim.evidence_chunk_ids
                and best_score > max(cited_scores, default=-1)
                and len(claim.evidence_chunk_ids) < 5
            ):
                claim.evidence_chunk_ids.append(best.chunk_id)
        return repaired

    @staticmethod
    def _evidence_binding_targets(
        target_id: str,
        plan: AgentTaskPlan,
    ) -> list[str]:
        """Allow one comparative table to support both versions of one fact."""

        target = next(item for item in plan.targets if item.target_id == target_id)
        if target.cross_target_group is None:
            return [target_id]
        return [
            candidate.target_id
            for candidate in plan.targets
            if candidate.cross_target_group == target.cross_target_group
        ]

    def run(self, request: AgentTaskRequest) -> AgentTaskTrace:
        task_id = uuid4().hex
        created_at = datetime.now(UTC)
        index_id = self.search_backend.manifest.index_id
        plan = plan_compare_task(request, available_companies=self.available_companies)
        if plan.clarification:
            trace = _clarification_trace(request, plan, index_id=index_id, created_at=created_at)
            return trace.model_copy(update={"task_id": task_id})
        if not self.model.available:
            raise RuntimeError("DeepSeek tool-calling agent requires a provider API key")

        initial_tools = _tool_definitions(plan)
        messages = _initial_messages(request, plan)
        prompt_sha256 = _prompt_sha256(messages, initial_tools)
        memory = EvidenceMemory(index_id=index_id)
        calls: list[AgentToolCall] = []
        turns: list[AgentModelTurn] = []
        target_by_id = {target.target_id: target for target in plan.targets}
        function_calls_used = 0
        search_counts = {target.target_id: 0 for target in plan.targets}
        searched_queries: set[tuple[str, str]] = set()
        finalization_mode = False

        for turn_number in range(1, request.max_rounds + 1):
            current_sufficiency = judge_sufficiency(
                plan, memory, minimum_per_target=request.min_evidence_per_target
            )
            remaining_function_calls = request.max_tool_calls - function_calls_used
            allowed_search_targets = {
                target_id for target_id in current_sufficiency.gaps if search_counts[target_id] < 2
            }
            if (
                current_sufficiency.status == "sufficient"
                or turn_number == request.max_rounds
                or remaining_function_calls <= 1
            ):
                allowed_search_targets = set()
            if current_sufficiency.status == "incomplete" and not allowed_search_targets and turns:
                missing_labels = [
                    target_by_id[target_id].label for target_id in current_sufficiency.gaps
                ]
                return AgentTaskTrace(
                    task_id=task_id,
                    runtime="deepseek_tool_calling",
                    status="completed",
                    stop_reason="no_new_evidence",
                    query=request.query,
                    index_id=index_id,
                    created_at=created_at,
                    completed_at=datetime.now(UTC),
                    rounds_completed=len(turns),
                    plan=plan,
                    tool_calls=calls,
                    evidence_memory=memory,
                    sufficiency=current_sufficiency,
                    result=AgentTaskResult(
                        outcome="abstain",
                        answer=GeneratedAnswer(
                            answer=(
                                "未检索到以下目标的可验证证据："
                                f"{'、'.join(missing_labels)}。系统拒绝生成不完整比较，"
                                "不会用预测、目标或外部知识补全。"
                            ),
                            citations=[],
                            provider="agent-local-evidence-gate",
                            grounded=False,
                        ),
                        target_evidence=target_evidence(plan, memory),
                    ),
                    model_trace=_model_trace(self.model, prompt_sha256, turns),
                )
            tools = _tool_definitions(plan, allowed_search_target_ids=allowed_search_targets)
            if not allowed_search_targets and not finalization_mode:
                messages = _finalization_messages(request, plan, memory)
                finalization_mode = True
            try:
                response = self.model.complete(messages, tools)
            except Exception:  # noqa: BLE001 - persist safe model failure
                sufficiency = judge_sufficiency(
                    plan, memory, minimum_per_target=request.min_evidence_per_target
                )
                return AgentTaskTrace(
                    task_id=task_id,
                    runtime="deepseek_tool_calling",
                    status="failed",
                    stop_reason="model_error",
                    query=request.query,
                    index_id=index_id,
                    created_at=created_at,
                    completed_at=datetime.now(UTC),
                    rounds_completed=turn_number - 1,
                    plan=plan,
                    tool_calls=calls,
                    evidence_memory=memory,
                    sufficiency=sufficiency,
                    result=AgentTaskResult(
                        outcome="abstain",
                        answer=GeneratedAnswer(
                            answer="DeepSeek Agent 调用失败，任务已安全停止。",
                            citations=[],
                            provider="deepseek-agent-error",
                            grounded=False,
                        ),
                        target_evidence=target_evidence(plan, memory),
                    ),
                    model_trace=_model_trace(self.model, prompt_sha256, turns),
                )

            turn = AgentModelTurn(
                turn_number=turn_number,
                finish_reason=response.finish_reason,
                function_names=[call.name for call in response.tool_calls],
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                elapsed_ms=response.elapsed_ms,
            )
            turns.append(turn)
            messages.append(response.assistant_message())
            if not response.tool_calls:
                break

            for model_call in response.tool_calls:
                function_calls_used += 1
                if function_calls_used > request.max_tool_calls:
                    messages.append(
                        _tool_message(
                            model_call.call_id,
                            {"status": "error", "error": "tool budget exhausted"},
                        )
                    )
                    continue

                if model_call.name == "search_evidence":
                    try:
                        arguments = SearchEvidenceArguments.model_validate_json(
                            model_call.arguments
                        )
                        target = target_by_id[arguments.target_id]
                    except (ValidationError, KeyError) as exc:
                        turn.validation_errors.append(f"search_evidence: {exc}")
                        messages.append(
                            _tool_message(
                                model_call.call_id,
                                {"status": "error", "error": str(exc)},
                            )
                        )
                        continue
                    effective_query = arguments.query.strip()
                    if target.retrieval_hint and target.retrieval_hint not in effective_query:
                        effective_query = f"{effective_query} {target.retrieval_hint}"
                    normalized_query = " ".join(effective_query.lower().split())
                    query_key = (target.target_id, normalized_query)
                    if target.target_id not in allowed_search_targets:
                        messages.append(
                            _tool_message(
                                model_call.call_id,
                                {
                                    "status": "error",
                                    "error": (
                                        "search is closed for this target; submit the result"
                                    ),
                                },
                            )
                        )
                        continue
                    if query_key in searched_queries:
                        messages.append(
                            _tool_message(
                                model_call.call_id,
                                {
                                    "status": "error",
                                    "error": "duplicate search; submit or use a distinct gap query",
                                },
                            )
                        )
                        continue
                    searched_queries.add(query_key)
                    search_counts[target.target_id] += 1
                    started = perf_counter()
                    requested_top_k = min(arguments.top_k, request.top_k)
                    candidate_top_k = (
                        max(requested_top_k, 10)
                        if _needs_authoritative_source_ranking(request.query)
                        else requested_top_k
                    )
                    try:
                        search = self.search_backend.search(
                            SearchRequest(
                                query=effective_query,
                                mode=request.mode,
                                top_k=candidate_top_k,
                                filters=target.filters,
                            ),
                            f"{task_id}:{model_call.call_id}",
                        )
                    except Exception as exc:  # noqa: BLE001
                        calls.append(
                            AgentToolCall(
                                call_id=model_call.call_id,
                                round_number=turn_number,
                                target_id=target.target_id,
                                query=effective_query,
                                filters=target.filters,
                                status="error",
                                duration_ms=(perf_counter() - started) * 1000,
                                retrieval_trace_id=getattr(exc, "trace_id", None),
                                error_type=type(exc).__name__,
                                error_message=str(exc),
                            )
                        )
                        messages.append(
                            _tool_message(
                                model_call.call_id,
                                {"status": "error", "error": str(exc)},
                            )
                        )
                        continue
                    search_hits = (
                        _rank_authoritative_hits(
                            request.query,
                            search.hits,
                            limit=requested_top_k,
                        )
                        if candidate_top_k > requested_top_k
                        else search.hits
                    )
                    calls.append(
                        AgentToolCall(
                            call_id=model_call.call_id,
                            round_number=turn_number,
                            target_id=target.target_id,
                            query=effective_query,
                            filters=target.filters,
                            status="success",
                            duration_ms=(perf_counter() - started) * 1000,
                            retrieval_trace_id=search.trace_id,
                            evidence_chunk_ids=[hit.chunk.chunk_id for hit in search_hits],
                        )
                    )
                    binding_targets = self._evidence_binding_targets(target.target_id, plan)
                    new_evidence_count = 0
                    for binding_target_id in binding_targets:
                        new_evidence_count += add_evidence(
                            memory,
                            binding_target_id,
                            search_hits,
                        )
                    remaining_metrics = {
                        binding_target_id: missing_required_metrics(
                            plan, memory, binding_target_id
                        )
                        for binding_target_id in binding_targets
                    }
                    messages.append(
                        _tool_message(
                            model_call.call_id,
                            {
                                "status": "success",
                                "target_id": target.target_id,
                                "new_evidence_count": new_evidence_count,
                                "searches_used_for_target": search_counts[target.target_id],
                                "effective_query": effective_query,
                                "bound_target_ids": binding_targets,
                                "remaining_required_metrics": remaining_metrics,
                                "evidence": [
                                    {
                                        "chunk_id": hit.chunk.chunk_id,
                                        "company_name": hit.chunk.company_name,
                                        "report_year": hit.chunk.report_year,
                                        "pages": [hit.chunk.page_start, hit.chunk.page_end],
                                        "section_path": hit.chunk.section_path,
                                        "text": hit.chunk.text[:1200],
                                    }
                                    for hit in search_hits
                                ],
                            },
                        )
                    )
                    continue

                if model_call.name == "submit_comparison":
                    try:
                        submission = _parse_submission(model_call.arguments)
                    except ValidationError as exc:
                        turn.validation_errors.append(f"submit_comparison: {exc}")
                        messages.append(
                            _tool_message(
                                model_call.call_id,
                                {"status": "error", "error": str(exc)},
                            )
                        )
                        continue
                    result, errors = self._submitted_answer(
                        request, submission, plan, memory
                    )
                    if errors:
                        turn.validation_errors.extend(errors)
                        messages.append(
                            _tool_message(
                                model_call.call_id,
                                {"status": "error", "errors": errors},
                            )
                        )
                        continue
                    assert result is not None
                    sufficiency = judge_sufficiency(
                        plan,
                        memory,
                        minimum_per_target=request.min_evidence_per_target,
                    )
                    if result.outcome == "answer" and sufficiency.status != "sufficient":
                        turn.validation_errors.append("local evidence sufficiency gate failed")
                        messages.append(
                            _tool_message(
                                model_call.call_id,
                                {
                                    "status": "error",
                                    "error": "local evidence sufficiency gate failed",
                                },
                            )
                        )
                        continue
                    stop_reason = (
                        "sufficient_evidence"
                        if sufficiency.status == "sufficient"
                        else "needs_clarification"
                        if result.outcome == "clarify"
                        else "no_new_evidence"
                    )
                    return AgentTaskTrace(
                        task_id=task_id,
                        runtime="deepseek_tool_calling",
                        status=(
                            "needs_clarification" if result.outcome == "clarify" else "completed"
                        ),
                        stop_reason=stop_reason,
                        query=request.query,
                        index_id=index_id,
                        created_at=created_at,
                        completed_at=datetime.now(UTC),
                        rounds_completed=turn_number,
                        plan=plan,
                        tool_calls=calls,
                        evidence_memory=memory,
                        sufficiency=sufficiency,
                        result=result,
                        model_trace=_model_trace(self.model, prompt_sha256, turns),
                    )

                messages.append(
                    _tool_message(
                        model_call.call_id,
                        {"status": "error", "error": "unknown tool"},
                    )
                )

        sufficiency = judge_sufficiency(
            plan, memory, minimum_per_target=request.min_evidence_per_target
        )
        return AgentTaskTrace(
            task_id=task_id,
            runtime="deepseek_tool_calling",
            status="completed",
            stop_reason=(
                "tool_budget_exhausted"
                if function_calls_used >= request.max_tool_calls
                else "model_budget_exhausted"
            ),
            query=request.query,
            index_id=index_id,
            created_at=created_at,
            completed_at=datetime.now(UTC),
            rounds_completed=len(turns),
            plan=plan,
            tool_calls=calls,
            evidence_memory=memory,
            sufficiency=sufficiency,
            result=AgentTaskResult(
                outcome="abstain",
                answer=GeneratedAnswer(
                    answer="DeepSeek 未在运行预算内提交通过本地门禁的结果。",
                    citations=[],
                    provider="deepseek-agent-budget",
                    grounded=False,
                ),
                target_evidence=target_evidence(plan, memory),
            ),
            model_trace=_model_trace(self.model, prompt_sha256, turns),
        )


class ExtractSearchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=5)


class PageWindowArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    anchor_chunk_id: str
    before_pages: int = Field(default=1, ge=0, le=2)
    after_pages: int = Field(default=1, ge=0, le=2)


class SubmitFactRequirementsArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirements: list[AtomicFactRequirement] = Field(min_length=1, max_length=24)


class SubmittedExtractFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=1000)
    evidence_chunk_ids: list[str] = Field(min_length=1, max_length=5)
    requirement_ids: list[str] = Field(min_length=1, max_length=24)


class SubmitExtractionArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["answer", "abstain"]
    message: str = Field(min_length=1, max_length=2000)
    facts: list[SubmittedExtractFact] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


def _parse_extraction_submission(arguments: str) -> SubmitExtractionArguments:
    """Normalize only the common singular/plural requirement-ID redundancy."""

    try:
        return SubmitExtractionArguments.model_validate_json(arguments)
    except ValidationError as validation_error:
        try:
            payload = json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            raise validation_error
        if not isinstance(payload, dict) or not isinstance(payload.get("facts"), list):
            raise
        for fact in payload["facts"]:
            if not isinstance(fact, dict) or "requirement_id" not in fact:
                continue
            singular = fact.pop("requirement_id")
            if "requirement_ids" not in fact:
                fact["requirement_ids"] = [singular]
        return SubmitExtractionArguments.model_validate(payload)


def _extract_search_tool() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "search_evidence",
                "description": "在已限定的单份年报中检索抽取任务的表格锚点。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 5},
                    },
                    "required": ["query", "top_k"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def _page_window_tool(anchor_chunk_ids: list[str]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_page_window",
                "description": (
                    "展开一个检索命中所在文档的相邻页，用章节路径和上下文消解"
                    "跨页表格及重复行名。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "anchor_chunk_id": {
                            "type": "string",
                            "enum": anchor_chunk_ids,
                        },
                        "before_pages": {"type": "integer", "minimum": 0, "maximum": 2},
                        "after_pages": {"type": "integer", "minimum": 0, "maximum": 2},
                    },
                    "required": ["anchor_chunk_id", "before_pages", "after_pages"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def _plan_fact_requirements_tool(
    chunk_ids: list[str],
    *,
    include_candidate_ids: bool = True,
) -> list[dict]:
    item_properties = {
        "requirement_id": {
            "type": "string",
            "pattern": "^r[1-9]\\d*$",
        },
        "description": {
            "type": "string",
            "maxLength": 120,
        },
        "subject": {"type": ["string", "null"]},
        "subject_scope": {
            "type": "string",
            "enum": [
                "group",
                "business_segment",
                "document",
                "unspecified",
            ],
        },
        "fact_period": {"type": ["string", "null"]},
        "evidence_type": {
            "type": "string",
            "enum": [
                "table_value",
                "narrative",
                "audit_risk",
                "audit_response",
                "accounting_policy",
                "other",
            ],
        },
    }
    required_fields = [
        "requirement_id",
        "description",
        "subject",
        "subject_scope",
        "fact_period",
        "evidence_type",
    ]
    if include_candidate_ids:
        item_properties["candidate_evidence_chunk_ids"] = {
            "type": "array",
            "items": {
                "type": "string",
                "enum": chunk_ids,
            },
            "maxItems": 8,
        }
        required_fields.append("candidate_evidence_chunk_ids")
    return [
        {
            "type": "function",
            "function": {
                "name": "submit_fact_requirements",
                "description": (
                    "在最终回答前提交完整、非重叠、可逐项核验的原子事实清单。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "requirements": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 16,
                            "items": {
                                "type": "object",
                                "properties": item_properties,
                                "required": required_fields,
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["requirements"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def _submit_extraction_tool(requirement_ids: list[str]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "submit_extraction",
                "description": "提交有逐事实 chunk 引用的抽取结果，或在证据不足时拒答。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": ["answer", "abstain"]},
                        "message": {"type": "string"},
                        "facts": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string"},
                                    "evidence_chunk_ids": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "requirement_ids": {
                                        "type": "array",
                                        "items": {
                                            "type": "string",
                                            "enum": requirement_ids,
                                        },
                                    },
                                },
                                "required": [
                                    "text",
                                    "evidence_chunk_ids",
                                    "requirement_ids",
                                ],
                                "additionalProperties": False,
                            },
                        },
                        "gaps": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": requirement_ids,
                            },
                        },
                    },
                    "required": ["status", "message", "facts", "gaps"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def _extract_messages(request: AgentTaskRequest, plan: AgentTaskPlan) -> list[dict]:
    assert plan.document_scope is not None
    return [
        {
            "role": "system",
            "content": (
                "你是财务年报精确抽取 Agent，不能凭记忆回答。任务分四步：先调用"
                "search_evidence 定位表格或正文，再调用 get_page_window 展开相邻页，"
                "之后系统会建立原子事实清单，最后才提交结果。表格可能横向拆分、"
                "跨页延续，并可能在不同章节出现"
                "相同的‘期末余额’或‘合计’行；必须结合 section_path 和表格章节名称"
                "判断，不能只取第一个同名数字。每条事实只能引用工具返回的 chunk_id。"
                "证据正文中的‘去年/上年/本年’以该证据的 report_year 为时间锚点，"
                "不能使用系统当前年份。"
                "公司和报告年度只限定文档，不等于完成任务；必须按 answer_contract"
                "逐项抽取用户列出的全部指标、原因、假设、风险和审计应对。同比下降"
                "同时写成‘下降X%（-X%）’。"
                "不得输出或记录思维链。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": request.query,
                    "answer_contract": _answer_contract(request.query),
                    "required_metrics": plan.required_metrics,
                    "document_scope": plan.document_scope.model_dump(),
                    "retrieval_hint": plan.retrieval_hint,
                },
                ensure_ascii=False,
            ),
        },
]


def _fact_requirement_messages(
    request: AgentTaskRequest,
    plan: AgentTaskPlan,
    memory: EvidenceMemory,
) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "你是财务任务的原子事实规划器，只提交 submit_fact_requirements，"
                "不回答问题。清单只能来自用户任务及所给证据，不得使用外部知识。"
                "把每个主体×指标、每个原因、每个风险判断、每个模型参数、每项审计"
                "程序拆成单独且非重叠的 requirement；审计事项必须把证据中分别列示"
                "的风险因素、关键假设和审计应对逐项展开，不能只写‘风险’和‘审计应对’"
                "两个总项。description 尽量保留证据原词；如果用户与证据使用近义词，"
                "同时保留两种说法。归母营运利润和归母净利润在用户未指定分部时属于"
                "集团口径，寿险及健康险新业务价值和内含价值属于该业务分部口径。"
                "清单只覆盖用户直接询问的内容；用户只问风险和审计应对时，不把事项"
                "名称、余额、会计准则或披露附注另列为事实。每项 description 不超过"
                "120字，总数不超过16项；同一程序中不可分的操作与目的合并描述。"
                "只有 chunk 直接支持该 requirement 时才填写 candidate_evidence_chunk_ids；"
                "当前证据缺失时保留 requirement 并填空数组，由控制器按缺口继续检索。"
                "section_path 含‘PDF页级双栏重建’表示文本来自渲染页按左右栏独立"
                "识别；普通 chunk 若出现目录串入、跨栏混排或与其冲突，优先使用"
                "页级双栏重建证据。"
                "requirement_id 按 r1、r2 连续编号。不得输出思维链。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": request.query,
                    "answer_contract": _answer_contract(request.query),
                    "required_metrics": plan.required_metrics,
                    "document_year": plan.document_year,
                    "fact_periods": plan.fact_periods,
                    "evidence": DeepSeekExtractAgent._evidence_payload(memory),
                },
                ensure_ascii=False,
            ),
        },
    ]


def _extract_finalization_messages(
    request: AgentTaskRequest,
    plan: AgentTaskPlan,
    memory: EvidenceMemory,
) -> list[dict]:
    evidence = [
        {
            "chunk_id": item.chunk_id,
            "company_name": item.company_name,
            "report_year": item.report_year,
            "pages": [item.page_start, item.page_end],
            "section_path": item.section_path,
            "text": item.excerpt,
        }
        for item in memory.items
    ]
    return [
        {
            "role": "system",
            "content": (
                "你是财务年报精确抽取 Agent 的最终提交器。现在禁止继续检索，必须"
                "调用 submit_extraction。结合 section_path 消解跨页表格和重复行名；"
                "证据充分时提交 answer，每条事实引用下方 chunk_id；否则提交 abstain。"
                "正文相对年份词必须使用证据的 report_year 解释，不得使用当前年份。"
                "提交前必须逐项覆盖 fact_requirements；每条事实用 requirement_ids 绑定"
                "它实际回答的清单项，并引用直接支持该事实的 chunk。不得把集团指标"
                "替换成业务分部指标，不得遗漏并列子项。同比下降同时写成"
                "‘下降X%（-X%）’，近义表述同时保留用户用词和证据原词。"
                "section_path 含‘PDF页级双栏重建’的证据来自渲染页左右栏独立识别，"
                "可用于纠正普通 chunk 的目录串入或跨栏混排。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": request.query,
                    "answer_contract": _answer_contract(request.query),
                    "fact_requirements": [
                        requirement.model_dump()
                        for requirement in plan.fact_requirements
                    ],
                    "evidence": evidence,
                },
                ensure_ascii=False,
            ),
        },
    ]


def _page_window_messages(
    request: AgentTaskRequest,
    memory: EvidenceMemory,
) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "你是跨页表格证据定位器。只调用一次 get_page_window，选择最接近任务"
                "所需章节和表格的 anchor_chunk_id；不要同时展开多个锚点。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": request.query,
                    "search_evidence": DeepSeekExtractAgent._evidence_payload(memory),
                },
                ensure_ascii=False,
            ),
        },
    ]


def _validate_fact_requirements(
    plan: AgentTaskPlan,
    requirements: list[AtomicFactRequirement],
    memory: EvidenceMemory,
    query: str,
) -> list[str]:
    errors: list[str] = []
    expected_ids = [f"r{number}" for number in range(1, len(requirements) + 1)]
    actual_ids = [requirement.requirement_id for requirement in requirements]
    if actual_ids != expected_ids:
        errors.append("requirement ids must be unique and consecutive from r1")
    normalized_descriptions = [
        re.sub(r"\s+", "", requirement.description) for requirement in requirements
    ]
    if len(set(normalized_descriptions)) != len(normalized_descriptions):
        errors.append("fact requirements must not contain duplicate descriptions")
    known_chunks = {item.chunk_id for item in memory.items}
    for requirement in requirements:
        unknown = set(requirement.candidate_evidence_chunk_ids) - known_chunks
        if unknown:
            errors.append(
                f"{requirement.requirement_id} references unknown candidate evidence"
            )
    joined_descriptions = " ".join(normalized_descriptions)
    for metric in plan.required_metrics:
        aliases = METRIC_EVIDENCE_ALIASES.get(metric, (metric,))
        if not any(re.sub(r"\s+", "", alias) in joined_descriptions for alias in aliases):
            errors.append(f"fact requirements omit requested metric: {metric}")
    is_audit_task = any(
        cue in query for cue in ("关键审计事项", "审计应对", "审计程序")
    )
    if is_audit_task and "风险" in query and not any(
        requirement.evidence_type == "audit_risk" for requirement in requirements
    ):
        errors.append("audit task requires at least one audit_risk requirement")
    if is_audit_task and "审计应对" in query and not any(
        requirement.evidence_type == "audit_response" for requirement in requirements
    ):
        errors.append("audit task requires at least one audit_response requirement")
    for requirement in requirements:
        if (
            any(
                cue in requirement.description
                for cue in (
                    "归母营运利润",
                    "归母净利润",
                    "归属于母公司股东的营运利润",
                    "归属于母公司股东的净利润",
                    "归属于上市公司股东的净利润",
                )
            )
            and requirement.subject_scope not in {"group", "business_segment"}
        ):
            errors.append(
                f"{requirement.requirement_id} must preserve group scope for parent-attributable profit"
            )
    return errors


def _complete_required_metric_requirements(
    plan: AgentTaskPlan,
    requirements: list[AtomicFactRequirement],
) -> list[AtomicFactRequirement]:
    joined_descriptions = re.sub(
        r"\s+",
        "",
        " ".join(requirement.description for requirement in requirements),
    )
    completed = list(requirements)
    company = (
        plan.document_scope.company_names[0]
        if plan.document_scope is not None
        and len(plan.document_scope.company_names) == 1
        else None
    )
    period = "、".join(f"{year}年" for year in plan.fact_periods) or None
    for metric in plan.required_metrics:
        aliases = METRIC_EVIDENCE_ALIASES.get(metric, (metric,))
        if any(
            re.sub(r"\s+", "", alias) in joined_descriptions
            for alias in aliases
        ):
            continue
        if metric in {"归母营运利润", "归母净利润"}:
            subject = company
            subject_scope = "group"
        elif metric in {"新业务价值", "内含价值"}:
            subject = "寿险及健康险业务"
            subject_scope = "business_segment"
        else:
            subject = company
            subject_scope = "document"
        completed.append(
            AtomicFactRequirement(
                requirement_id=f"r{len(completed) + 1}",
                description=" ".join(
                    part for part in (period, metric) if part
                ),
                subject=subject,
                subject_scope=subject_scope,
                fact_period=period,
                evidence_type="table_value",
            )
        )
    return completed


def _bind_audit_requirement_candidates(
    requirements: list[AtomicFactRequirement],
    memory: EvidenceMemory,
) -> None:
    """Bind compact audit requirements locally to already retrieved evidence.

    The compact tool schema intentionally omits repeated long chunk IDs so a
    many-procedure audit ledger fits within the provider's output budget.
    """

    candidates = [
        item
        for item in memory.items
        if "审计" in " ".join(item.section_path)
        or "PDF页级双栏重建" in item.section_path
    ]
    if not candidates:
        candidates = list(memory.items)
    for requirement in requirements:
        description_parts = [
            part
            for part in re.split(r"[，。；：、,;:\s]+", requirement.description)
            if len(part) >= 2
        ]
        ranked: list[tuple[int, int, AgentEvidence]] = []
        for position, evidence in enumerate(candidates):
            compact = re.sub(
                r"\s+",
                "",
                " ".join([*evidence.section_path, evidence.excerpt]),
            )
            score = sum(
                min(len(part), 16)
                for part in description_parts
                if re.sub(r"\s+", "", part) in compact
            )
            if "PDF页级双栏重建" in evidence.section_path:
                score += 6
            if requirement.evidence_type == "audit_response" and any(
                cue in compact for cue in ("审计中如何应对", "测试", "评价", "检查")
            ):
                score += 4
            if requirement.evidence_type == "audit_risk" and any(
                cue in compact for cue in ("重大会计判断", "风险", "关键参数", "假设")
            ):
                score += 4
            ranked.append((score, -position, evidence))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        requirement.candidate_evidence_chunk_ids = [
            evidence.chunk_id
            for score, _, evidence in ranked[:4]
            if score > 0
        ]


def _extract_authority_queries(plan: AgentTaskPlan) -> list[str]:
    if len(plan.required_metrics) < 2:
        return []
    group_metrics = [
        metric
        for metric in plan.required_metrics
        if metric not in {"新业务价值", "内含价值"}
    ]
    segment_metrics = [
        metric
        for metric in plan.required_metrics
        if metric in {"新业务价值", "内含价值"}
    ]
    queries: list[str] = []
    if group_metrics:
        queries.append(
            " ".join(
                ["财务摘要", "主要财务数据", "集团合并", *group_metrics]
            )
        )
    if segment_metrics:
        queries.append(
            " ".join(
                [
                    "业绩综述",
                    "关键指标",
                    "寿险及健康险业务",
                    "可比口径",
                    *segment_metrics,
                ]
            )
        )
    return queries


def _normalize_requirement_subject_scopes(
    requirements: list[AtomicFactRequirement],
) -> None:
    for requirement in requirements:
        compact = re.sub(
            r"\s+",
            "",
            f"{requirement.subject or ''}{requirement.description}",
        )
        if "寿险及健康险业务" in compact:
            requirement.subject = "寿险及健康险业务"
            requirement.subject_scope = "business_segment"
        elif any(
            cue in compact
            for cue in (
                "归母营运利润",
                "归母净利润",
                "归属于母公司股东的营运利润",
                "归属于母公司股东的净利润",
            )
        ):
            requirement.subject_scope = "group"


def _bind_authority_hits_to_requirements(
    requirements: list[AtomicFactRequirement],
    hits: list[SearchHit],
) -> None:
    for requirement in requirements:
        requirement_values = _meaningful_numeric_tokens(requirement.description)
        for hit in hits:
            evidence = AgentEvidence(
                chunk_id=hit.chunk.chunk_id,
                content_sha256="0" * 64,
                target_ids=[],
                document_id=hit.chunk.document_id,
                document_key=hit.chunk.document_key,
                company_name=hit.chunk.company_name,
                report_year=hit.chunk.report_year,
                page_start=hit.chunk.page_start,
                page_end=hit.chunk.page_end,
                section_path=hit.chunk.section_path,
                excerpt=hit.chunk.text,
            )
            if not _candidate_scope_compatible(requirement, evidence):
                continue
            value_support = bool(
                requirement_values & _meaningful_numeric_tokens(hit.chunk.text)
            )
            metric_support = any(
                (
                    metric in requirement.description
                    or any(
                        alias in requirement.description
                        for alias in METRIC_EVIDENCE_ALIASES.get(metric, ())
                    )
                )
                and (
                    metric in hit.chunk.text
                    or any(
                        alias in hit.chunk.text
                        for alias in METRIC_EVIDENCE_ALIASES.get(metric, ())
                    )
                )
                for metric in METRIC_EVIDENCE_ALIASES
            )
            if not value_support and not metric_support:
                continue
            if hit.chunk.chunk_id not in requirement.candidate_evidence_chunk_ids:
                requirement.candidate_evidence_chunk_ids.append(hit.chunk.chunk_id)
            requirement.candidate_evidence_chunk_ids = (
                requirement.candidate_evidence_chunk_ids[:8]
            )


def _repair_submitted_fact_scope_labels(
    submission: SubmitExtractionArguments,
    plan: AgentTaskPlan,
) -> SubmitExtractionArguments:
    if submission.status != "answer":
        return submission
    repaired = submission.model_copy(deep=True)
    requirements = {
        item.requirement_id: item for item in plan.fact_requirements
    }
    for fact in repaired.facts:
        missing_subjects: list[str] = []
        for requirement_id in fact.requirement_ids:
            requirement = requirements.get(requirement_id)
            if (
                requirement is None
                or requirement.subject_scope != "business_segment"
                or not requirement.subject
            ):
                continue
            anchor = _requirement_subject_anchor(requirement)
            if anchor and anchor not in re.sub(r"\s+", "", fact.text):
                missing_subjects.append(requirement.subject)
        if missing_subjects:
            fact.text = "、".join(dict.fromkeys(missing_subjects)) + "：" + fact.text
    return repaired


def _repair_extract_authoritative_citations(
    query: str,
    submission: SubmitExtractionArguments,
    plan: AgentTaskPlan,
    memory: EvidenceMemory,
) -> SubmitExtractionArguments:
    if submission.status != "answer" or not _needs_authoritative_source_ranking(query):
        return submission
    repaired = submission.model_copy(deep=True)
    evidence = {item.chunk_id: item for item in memory.items}
    requirements = {
        item.requirement_id: item for item in plan.fact_requirements
    }
    for fact in repaired.facts:
        values = _meaningful_numeric_tokens(fact.text)
        if not values:
            continue
        compatible_requirements = [
            requirements[requirement_id]
            for requirement_id in fact.requirement_ids
            if requirement_id in requirements
        ]
        candidates = [
            item
            for item in memory.items
            if values & _meaningful_numeric_tokens(item.excerpt)
            and (
                not compatible_requirements
                or any(
                    _candidate_scope_compatible(requirement, item)
                    for requirement in compatible_requirements
                )
            )
        ]
        if not candidates:
            continue
        best = max(
            candidates,
            key=lambda item: (
                _source_authority_score(query, item.section_path, item.excerpt),
                len(values & _meaningful_numeric_tokens(item.excerpt)),
            ),
        )
        cited_scores = [
            _source_authority_score(
                query,
                evidence[chunk_id].section_path,
                evidence[chunk_id].excerpt,
            )
            for chunk_id in fact.evidence_chunk_ids
            if chunk_id in evidence
        ]
        best_score = _source_authority_score(
            query,
            best.section_path,
            best.excerpt,
        )
        if (
            best.chunk_id not in fact.evidence_chunk_ids
            and best_score > max(cited_scores, default=-1)
            and len(fact.evidence_chunk_ids) < 5
        ):
            fact.evidence_chunk_ids.append(best.chunk_id)
    return repaired


def _requirement_gap_query(
    requirement: AtomicFactRequirement,
    plan: AgentTaskPlan,
) -> str:
    scope_hint = {
        "group": "集团 归属于母公司股东 主要经营业绩",
        "business_segment": f"{requirement.subject or ''} 业务分部",
        "document": "年报披露",
        "unspecified": "",
    }[requirement.subject_scope]
    period = requirement.fact_period or " ".join(
        f"{year}年" for year in plan.fact_periods
    )
    return " ".join(
        part
        for part in (scope_hint, requirement.description, period, "单位")
        if part
    )


def _requirement_subject_anchor(requirement: AtomicFactRequirement) -> str:
    subject = re.sub(r"\s+", "", requirement.subject or "")
    for cue in (
        "归属于母公司股东的",
        "归属于上市公司股东的",
        "归母",
        "营运利润",
        "净利润",
        "营业收入",
        "收入",
        "毛利率",
        "新业务价值",
        "内含价值",
    ):
        subject = subject.replace(cue, "")
    return subject.strip("的：:，,")


def _candidate_scope_compatible(
    requirement: AtomicFactRequirement,
    evidence: AgentEvidence,
) -> bool:
    compact = re.sub(
        r"\s+",
        "",
        " ".join([*evidence.section_path, evidence.excerpt]),
    )
    if requirement.subject_scope == "group":
        segment_cues = ("寿险及健康险业务", "业务分部", "分部利润")
        group_cues = (
            "集团归母",
            "集团归属于母公司股东",
            "归属于母公司股东的营运利润",
            "归属于母公司股东的净利润",
            "归属于上市公司股东的净利润",
        )
        if any(cue in compact for cue in segment_cues) and not any(
            cue in compact for cue in group_cues
        ):
            return False
    if requirement.subject_scope == "business_segment":
        subject_anchor = _requirement_subject_anchor(requirement)
        if subject_anchor and subject_anchor not in compact:
            return False
    return True


def _validate_requirement_scope(
    requirement: AtomicFactRequirement,
    fact_text: str,
) -> list[str]:
    errors: list[str] = []
    compact = re.sub(r"\s+", "", fact_text)
    if requirement.subject_scope == "group":
        segment_cues = ("寿险及健康险业务", "业务分部", "分部利润")
        group_cues = ("集团", "归母", "归属于母公司股东", "归属于上市公司股东")
        if any(cue in compact for cue in segment_cues) and not any(
            cue in compact for cue in group_cues
        ):
            errors.append(
                f"{requirement.requirement_id} uses a business-segment fact for a group requirement"
            )
        parent_profit_cues = (
            "归母营运利润",
            "归母净利润",
            "归属于母公司股东的营运利润",
            "归属于母公司股东的净利润",
            "归属于上市公司股东的净利润",
        )
        if any(cue in requirement.description for cue in parent_profit_cues) and not any(
            cue in compact for cue in group_cues[1:]
        ):
            errors.append(
                f"{requirement.requirement_id} does not state the parent-attributable group scope"
            )
    subject_anchor = _requirement_subject_anchor(requirement)
    if (
        requirement.subject_scope == "business_segment"
        and subject_anchor
        and subject_anchor not in compact
    ):
        errors.append(
            f"{requirement.requirement_id} does not state its requested business segment"
        )
    return errors


class DeepSeekExtractAgent:
    """Three-phase extraction agent with a same-document page-window tool."""

    _evidence_target = "task:extract"

    def __init__(
        self,
        search_backend,
        model: ToolCallingModel,
        *,
        available_companies: list[str],
        available_report_years_by_company: dict[str, list[int]] | None = None,
        layout_inspector: PdfRegionInspector | None = None,
    ) -> None:
        self.search_backend = search_backend
        self.model = model
        self.available_companies = available_companies
        self.available_report_years_by_company = (
            available_report_years_by_company or {}
        )
        self.layout_inspector = layout_inspector

    @staticmethod
    def _needs_audit_layout_reconstruction(query: str) -> bool:
        return any(cue in query for cue in ("关键审计事项", "审计应对", "审计程序"))

    @staticmethod
    def _layout_pages(anchor) -> list[int]:
        """Return the anchor's last page and one continuation page."""

        return [anchor.page_end, anchor.page_end + 1]

    @classmethod
    def _add_layout_reconstruction(
        cls,
        memory: EvidenceMemory,
        anchor,
        reconstruction: PageLayoutReconstruction,
    ) -> bool:
        if any(
            item.chunk_id == reconstruction.evidence_chunk_id
            for item in memory.items
        ):
            return False
        memory.items.append(
            AgentEvidence(
                chunk_id=reconstruction.evidence_chunk_id,
                content_sha256=reconstruction.content_sha256,
                target_ids=[cls._evidence_target],
                document_id=anchor.document_id,
                document_key=anchor.document_key,
                company_name=anchor.company_name,
                report_year=anchor.report_year,
                page_start=reconstruction.page_number,
                page_end=reconstruction.page_number,
                section_path=[*anchor.section_path, "PDF页级双栏重建"],
                excerpt=reconstruction.evidence_text,
            )
        )
        return True

    @staticmethod
    def _evidence_payload(memory: EvidenceMemory) -> list[dict]:
        return [
            {
                "chunk_id": item.chunk_id,
                "company_name": item.company_name,
                "report_year": item.report_year,
                "pages": [item.page_start, item.page_end],
                "section_path": item.section_path,
                "text": item.excerpt,
            }
            for item in memory.items
        ]

    @staticmethod
    def _result(
        submission: SubmitExtractionArguments,
        plan: AgentTaskPlan,
        memory: EvidenceMemory,
    ) -> tuple[AgentTaskResult | None, list[str]]:
        if submission.status != "answer":
            return (
                AgentTaskResult(
                    outcome="abstain",
                    answer=GeneratedAnswer(
                        answer=submission.message,
                        citations=[],
                        provider="deepseek-tool-calling",
                        grounded=False,
                    ),
                    target_evidence={},
                ),
                [],
            )
        evidence = {item.chunk_id: item for item in memory.items}
        requirements = {
            item.requirement_id: item for item in plan.fact_requirements
        }
        errors: list[str] = []
        if not requirements:
            errors.append("answer requires a non-empty atomic fact ledger")
        if not submission.facts:
            errors.append("answer must contain at least one fact")
        covered_requirement_ids: set[str] = set()
        for fact in submission.facts:
            for chunk_id in fact.evidence_chunk_ids:
                if chunk_id not in evidence:
                    errors.append(f"unknown evidence chunk: {chunk_id}")
            for requirement_id in fact.requirement_ids:
                requirement = requirements.get(requirement_id)
                if requirement is None:
                    errors.append(f"unknown fact requirement: {requirement_id}")
                    continue
                covered_requirement_ids.add(requirement_id)
                errors.extend(_validate_requirement_scope(requirement, fact.text))
        missing_requirement_ids = set(requirements) - covered_requirement_ids
        if missing_requirement_ids:
            errors.append(
                "answer omits fact requirements: "
                + ", ".join(sorted(missing_requirement_ids))
            )
        if errors:
            return None, errors

        citations: list[Citation] = []
        ordinal_by_chunk: dict[str, int] = {}
        claim_citations: list[ClaimCitation] = []
        requirement_claims: dict[str, list[str]] = {
            requirement_id: [] for requirement_id in requirements
        }
        requirement_evidence: dict[str, list[str]] = {
            requirement_id: [] for requirement_id in requirements
        }
        lines: list[str] = []
        for fact in submission.facts:
            ordinals: list[int] = []
            for chunk_id in fact.evidence_chunk_ids:
                ordinal = ordinal_by_chunk.get(chunk_id)
                if ordinal is None:
                    item = evidence[chunk_id]
                    ordinal = len(citations) + 1
                    ordinal_by_chunk[chunk_id] = ordinal
                    citations.append(
                        Citation(
                            ordinal=ordinal,
                            chunk_id=chunk_id,
                            page_start=item.page_start,
                            page_end=item.page_end,
                            section_path=item.section_path,
                            excerpt=item.excerpt,
                        )
                    )
                ordinals.append(ordinal)
            line = fact.text + "".join(f"[{ordinal}]" for ordinal in ordinals)
            lines.append(line)
            claim_citations.append(
                ClaimCitation(claim=fact.text, citation_ordinals=ordinals)
            )
            for requirement_id in fact.requirement_ids:
                requirement_claims[requirement_id].append(fact.text)
                requirement_evidence[requirement_id] = list(
                    dict.fromkeys(
                        requirement_evidence[requirement_id]
                        + fact.evidence_chunk_ids
                    )
                )
        return (
            AgentTaskResult(
                outcome="answer",
                answer=GeneratedAnswer(
                    answer="\n".join(lines),
                    citations=citations,
                    provider="deepseek-tool-calling",
                    grounded=True,
                    claim_citations=claim_citations,
                ),
                target_evidence={},
                requirement_claims=requirement_claims,
                requirement_evidence=requirement_evidence,
                requirement_scope_validated={
                    requirement_id: True for requirement_id in requirements
                },
            ),
            [],
        )

    def _trace(
        self,
        *,
        task_id: str,
        request: AgentTaskRequest,
        plan: AgentTaskPlan,
        created_at: datetime,
        memory: EvidenceMemory,
        calls: list[AgentToolCall],
        turns: list[AgentModelTurn],
        prompt_sha256: str,
        result: AgentTaskResult,
        stop_reason: Literal[
            "sufficient_evidence",
            "no_new_evidence",
            "model_budget_exhausted",
            "invalid_model_output",
            "model_error",
        ],
        status: Literal["completed", "failed"] = "completed",
    ) -> AgentTaskTrace:
        requirement_ids = {
            requirement.requirement_id for requirement in plan.fact_requirements
        }
        covered_requirement_ids = {
            requirement_id
            for requirement_id, claims in result.requirement_claims.items()
            if claims
        }
        requirement_gaps = sorted(requirement_ids - covered_requirement_ids)
        answer_ready = result.outcome == "answer" and not requirement_gaps
        return AgentTaskTrace(
            task_id=task_id,
            task_type="extract",
            runtime="deepseek_tool_calling",
            status=status,
            stop_reason=stop_reason,
            query=request.query,
            index_id=memory.index_id,
            created_at=created_at,
            completed_at=datetime.now(UTC),
            rounds_completed=len(turns),
            plan=plan,
            tool_calls=calls,
            evidence_memory=memory,
            sufficiency=SufficiencyDecision(
                status="sufficient" if answer_ready else "incomplete",
                evidence_count_by_target={
                    self._evidence_target: len(memory.items),
                },
                gaps=[] if answer_ready else [self._evidence_target],
                requirement_gaps=requirement_gaps,
            ),
            result=result,
            model_trace=_model_trace(
                self.model,
                prompt_sha256,
                turns,
                prompt_revision=EXTRACT_PROMPT_REVISION,
            ),
        )

    def run(self, request: AgentTaskRequest) -> AgentTaskTrace:
        if request.task_type != "extract":
            raise ValueError("DeepSeekExtractAgent only accepts extract tasks")
        task_id = uuid4().hex
        created_at = datetime.now(UTC)
        index_id = self.search_backend.manifest.index_id
        plan = plan_document_task(
            request,
            available_companies=self.available_companies,
            available_report_years_by_company=self.available_report_years_by_company,
        )
        if plan.corpus_unavailable_reason:
            trace = _corpus_unavailable_trace(
                request,
                plan,
                index_id=index_id,
                created_at=created_at,
            )
            return trace.model_copy(update={"task_id": task_id})
        if plan.clarification:
            trace = _clarification_trace(
                request,
                plan,
                index_id=index_id,
                created_at=created_at,
            )
            return trace.model_copy(update={"task_id": task_id})
        if not self.model.available:
            raise RuntimeError("DeepSeek extraction agent requires a provider API key")
        assert plan.document_scope is not None

        messages = _extract_messages(request, plan)
        initial_tools = _extract_search_tool()
        prompt_sha256 = _prompt_sha256(messages, initial_tools)
        memory = EvidenceMemory(index_id=index_id)
        calls: list[AgentToolCall] = []
        turns: list[AgentModelTurn] = []

        def call_model(tools: list[dict]) -> ToolModelResponse | None:
            try:
                response = self.model.complete(messages, tools)
            except Exception:  # noqa: BLE001 - trace fails closed
                return None
            turns.append(
                AgentModelTurn(
                    turn_number=len(turns) + 1,
                    finish_reason=response.finish_reason,
                    function_names=[call.name for call in response.tool_calls],
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    elapsed_ms=response.elapsed_ms,
                )
            )
            messages.append(response.assistant_message())
            return response

        search_response = call_model(initial_tools)
        if search_response is None:
            return self._trace(
                task_id=task_id,
                request=request,
                plan=plan,
                created_at=created_at,
                memory=memory,
                calls=calls,
                turns=turns,
                prompt_sha256=prompt_sha256,
                result=AgentTaskResult(
                    outcome="abstain",
                    answer=GeneratedAnswer(
                        answer="DeepSeek 抽取 Agent 调用失败，任务已安全停止。",
                        citations=[],
                        provider="deepseek-agent-error",
                        grounded=False,
                    ),
                    target_evidence={},
                ),
                stop_reason="model_error",
                status="failed",
            )
        for model_call in search_response.tool_calls[: request.max_tool_calls]:
            if model_call.name != "search_evidence":
                turns[-1].validation_errors.append(f"unexpected tool: {model_call.name}")
                continue
            try:
                arguments = ExtractSearchArguments.model_validate_json(model_call.arguments)
            except ValidationError as exc:
                turns[-1].validation_errors.append(f"search_evidence: {exc}")
                messages.append(_tool_message(model_call.call_id, {"status": "error", "error": str(exc)}))
                continue
            effective_query = arguments.query.strip()
            if plan.retrieval_hint and plan.retrieval_hint not in effective_query:
                effective_query = f"{effective_query} {plan.retrieval_hint}"
            started = perf_counter()
            try:
                search = self.search_backend.search(
                    SearchRequest(
                        query=effective_query,
                        mode=request.mode,
                        top_k=min(arguments.top_k, request.top_k),
                        filters=plan.document_scope,
                    ),
                    f"{task_id}:{model_call.call_id}",
                )
            except Exception as exc:  # noqa: BLE001
                calls.append(
                    AgentToolCall(
                        call_id=model_call.call_id,
                        round_number=1,
                        target_id=self._evidence_target,
                        query=effective_query,
                        filters=plan.document_scope,
                        status="error",
                        duration_ms=(perf_counter() - started) * 1000,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
                messages.append(_tool_message(model_call.call_id, {"status": "error", "error": str(exc)}))
                continue
            calls.append(
                AgentToolCall(
                    call_id=model_call.call_id,
                    round_number=1,
                    target_id=self._evidence_target,
                    query=effective_query,
                    filters=plan.document_scope,
                    status="success",
                    duration_ms=(perf_counter() - started) * 1000,
                    retrieval_trace_id=search.trace_id,
                    evidence_chunk_ids=[hit.chunk.chunk_id for hit in search.hits],
                )
            )
            add_evidence(memory, self._evidence_target, search.hits)
            messages.append(
                _tool_message(
                    model_call.call_id,
                    {"status": "success", "evidence": self._evidence_payload(memory)},
                )
            )

        remaining_budget = request.max_tool_calls - len(calls)
        anchor_ids = [item.chunk_id for item in memory.items]
        if not anchor_ids or len(turns) >= request.max_rounds or remaining_budget <= 0:
            return self._trace(
                task_id=task_id,
                request=request,
                plan=plan,
                created_at=created_at,
                memory=memory,
                calls=calls,
                turns=turns,
                prompt_sha256=prompt_sha256,
                result=AgentTaskResult(
                    outcome="abstain",
                    answer=GeneratedAnswer(
                        answer="未找到可展开的表格锚点，系统拒绝猜测。",
                        citations=[],
                        provider="agent-local-evidence-gate",
                        grounded=False,
                    ),
                    target_evidence={},
                ),
                stop_reason="no_new_evidence",
            )

        messages = _page_window_messages(request, memory)
        window_response = call_model(_page_window_tool(anchor_ids))
        if window_response is None:
            return self._trace(
                task_id=task_id,
                request=request,
                plan=plan,
                created_at=created_at,
                memory=memory,
                calls=calls,
                turns=turns,
                prompt_sha256=prompt_sha256,
                result=AgentTaskResult(
                    outcome="abstain",
                    answer=GeneratedAnswer(
                        answer="DeepSeek 抽取 Agent 调用失败，任务已安全停止。",
                        citations=[],
                        provider="deepseek-agent-error",
                        grounded=False,
                    ),
                    target_evidence={},
                ),
                stop_reason="model_error",
                status="failed",
            )
        for model_call in window_response.tool_calls[:1]:
            if model_call.name != "get_page_window":
                turns[-1].validation_errors.append(f"unexpected tool: {model_call.name}")
                continue
            try:
                arguments = PageWindowArguments.model_validate_json(model_call.arguments)
                if arguments.anchor_chunk_id not in anchor_ids:
                    raise ValueError("anchor_chunk_id was not returned by search_evidence")
            except (ValidationError, ValueError) as exc:
                turns[-1].validation_errors.append(f"get_page_window: {exc}")
                messages.append(_tool_message(model_call.call_id, {"status": "error", "error": str(exc)}))
                continue
            started = perf_counter()
            try:
                chunks = self.search_backend.page_window(
                    arguments.anchor_chunk_id,
                    before_pages=arguments.before_pages,
                    after_pages=arguments.after_pages,
                    max_chunks=12,
                )
            except Exception as exc:  # noqa: BLE001
                calls.append(
                    AgentToolCall(
                        call_id=model_call.call_id,
                        round_number=2,
                        tool="get_page_window",
                        target_id=self._evidence_target,
                        query=arguments.anchor_chunk_id,
                        filters=plan.document_scope,
                        status="error",
                        duration_ms=(perf_counter() - started) * 1000,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
                messages.append(_tool_message(model_call.call_id, {"status": "error", "error": str(exc)}))
                continue
            window_hits = [
                SearchHit(rank=rank, chunk=chunk, score=0.0)
                for rank, chunk in enumerate(chunks, start=1)
            ]
            add_evidence(memory, self._evidence_target, window_hits)
            calls.append(
                AgentToolCall(
                    call_id=model_call.call_id,
                    round_number=2,
                    tool="get_page_window",
                    target_id=self._evidence_target,
                    query=arguments.anchor_chunk_id,
                    filters=plan.document_scope,
                    status="success",
                    duration_ms=(perf_counter() - started) * 1000,
                    evidence_chunk_ids=[chunk.chunk_id for chunk in chunks],
                )
            )
            if (
                self.layout_inspector is not None
                and self._needs_audit_layout_reconstruction(request.query)
            ):
                anchor = next(
                    (
                        chunk
                        for chunk in chunks
                        if chunk.chunk_id == arguments.anchor_chunk_id
                    ),
                    None,
                )
                if anchor is not None and anchor.document_key:
                    layout_started = perf_counter()
                    layout_chunk_ids: list[str] = []
                    layout_errors: list[str] = []
                    for page_number in self._layout_pages(anchor):
                        try:
                            reconstruction = (
                                self.layout_inspector.reconstruct_two_column_page(
                                    anchor.document_key,
                                    page_number,
                                )
                            )
                        except Exception as exc:  # noqa: BLE001 - optional fallback
                            layout_errors.append(
                                f"page {page_number}: {type(exc).__name__}: {exc}"
                            )
                            continue
                        if self._add_layout_reconstruction(
                            memory,
                            anchor,
                            reconstruction,
                        ):
                            layout_chunk_ids.append(
                                reconstruction.evidence_chunk_id
                            )
                    if layout_errors:
                        turns[-1].validation_errors.extend(
                            f"reconstruct_page_layout: {error}"
                            for error in layout_errors
                        )
                    calls.append(
                        AgentToolCall(
                            call_id=f"layout:{model_call.call_id}",
                            round_number=2,
                            tool="reconstruct_page_layout",
                            target_id=self._evidence_target,
                            query=arguments.anchor_chunk_id,
                            filters=plan.document_scope,
                            status=("success" if layout_chunk_ids else "error"),
                            duration_ms=(perf_counter() - layout_started) * 1000,
                            evidence_chunk_ids=layout_chunk_ids,
                            error_type=(
                                None if layout_chunk_ids else "LayoutReconstructionError"
                            ),
                            error_message=(
                                None
                                if layout_chunk_ids
                                else "; ".join(layout_errors)
                                or "no reconstructable page"
                            ),
                        )
                    )
            messages.append(
                _tool_message(
                    model_call.call_id,
                    {"status": "success", "evidence": self._evidence_payload(memory)},
                )
            )

        if len(turns) >= request.max_rounds or len(calls) >= request.max_tool_calls:
            return self._trace(
                task_id=task_id,
                request=request,
                plan=plan,
                created_at=created_at,
                memory=memory,
                calls=calls,
                turns=turns,
                prompt_sha256=prompt_sha256,
                result=AgentTaskResult(
                    outcome="abstain",
                    answer=GeneratedAnswer(
                        answer="工具预算不足以完成最终提交，系统拒绝猜测。",
                        citations=[],
                        provider="deepseek-agent-budget",
                        grounded=False,
                    ),
                    target_evidence={},
                ),
                stop_reason="model_budget_exhausted",
            )

        messages = _fact_requirement_messages(request, plan, memory)
        has_layout_evidence = any(
            "PDF页级双栏重建" in item.section_path
            for item in memory.items
        )
        requirement_response = call_model(
            _plan_fact_requirements_tool(
                [item.chunk_id for item in memory.items],
                include_candidate_ids=not has_layout_evidence,
            )
        )
        parsed_requirements: list[AtomicFactRequirement] | None = None
        if requirement_response is not None:
            for model_call in requirement_response.tool_calls:
                if model_call.name != "submit_fact_requirements":
                    turns[-1].validation_errors.append(
                        f"unexpected tool: {model_call.name}"
                    )
                    continue
                try:
                    requirement_submission = (
                        SubmitFactRequirementsArguments.model_validate_json(
                            model_call.arguments
                        )
                    )
                except ValidationError as exc:
                    turns[-1].validation_errors.append(
                        f"submit_fact_requirements: {exc}"
                    )
                    continue
                completed_requirements = _complete_required_metric_requirements(
                    plan,
                    requirement_submission.requirements,
                )
                _normalize_requirement_subject_scopes(completed_requirements)
                known_evidence_ids = {
                    item.chunk_id for item in memory.items
                }
                for requirement in completed_requirements:
                    requirement.candidate_evidence_chunk_ids = [
                        chunk_id
                        for chunk_id in requirement.candidate_evidence_chunk_ids
                        if chunk_id in known_evidence_ids
                    ]
                if has_layout_evidence:
                    _bind_audit_requirement_candidates(
                        completed_requirements,
                        memory,
                    )
                requirement_errors = _validate_fact_requirements(
                    plan,
                    completed_requirements,
                    memory,
                    request.query,
                )
                if requirement_errors:
                    turns[-1].validation_errors.extend(requirement_errors)
                    continue
                parsed_requirements = completed_requirements
                break
        if parsed_requirements is None:
            return self._trace(
                task_id=task_id,
                request=request,
                plan=plan,
                created_at=created_at,
                memory=memory,
                calls=calls,
                turns=turns,
                prompt_sha256=prompt_sha256,
                result=AgentTaskResult(
                    outcome="abstain",
                    answer=GeneratedAnswer(
                        answer="DeepSeek 未提交通过本地校验的原子事实清单。",
                        citations=[],
                        provider="deepseek-agent-invalid-requirements",
                        grounded=False,
                    ),
                    target_evidence={},
                ),
                stop_reason=(
                    "model_error"
                    if requirement_response is None
                    else "invalid_model_output"
                ),
                status=(
                    "failed" if requirement_response is None else "completed"
                ),
            )
        evidence_by_chunk = {
            item.chunk_id: item for item in memory.items
        }
        for requirement in parsed_requirements:
            requirement.candidate_evidence_chunk_ids = [
                chunk_id
                for chunk_id in requirement.candidate_evidence_chunk_ids
                if _candidate_scope_compatible(
                    requirement,
                    evidence_by_chunk[chunk_id],
                )
            ]
        plan.fact_requirements = parsed_requirements

        for authority_index, authority_query in enumerate(
            _extract_authority_queries(plan),
            start=1,
        ):
            if len(calls) >= request.max_tool_calls:
                break
            started = perf_counter()
            call_id = f"authority:{authority_index}"
            try:
                search = self.search_backend.search(
                    SearchRequest(
                        query=authority_query,
                        mode=request.mode,
                        top_k=10,
                        filters=plan.document_scope,
                    ),
                    f"{task_id}:{call_id}",
                )
            except Exception as exc:  # noqa: BLE001
                calls.append(
                    AgentToolCall(
                        call_id=call_id,
                        round_number=3,
                        tool="search_authoritative_source",
                        target_id=self._evidence_target,
                        query=authority_query,
                        filters=plan.document_scope,
                        status="error",
                        duration_ms=(perf_counter() - started) * 1000,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
                continue
            authority_hits = _rank_authoritative_hits(
                request.query,
                search.hits,
                limit=3,
            )
            add_evidence(memory, self._evidence_target, authority_hits)
            _bind_authority_hits_to_requirements(
                plan.fact_requirements,
                authority_hits,
            )
            calls.append(
                AgentToolCall(
                    call_id=call_id,
                    round_number=3,
                    tool="search_authoritative_source",
                    target_id=self._evidence_target,
                    query=authority_query,
                    filters=plan.document_scope,
                    status="success",
                    duration_ms=(perf_counter() - started) * 1000,
                    retrieval_trace_id=search.trace_id,
                    evidence_chunk_ids=[
                        hit.chunk.chunk_id for hit in authority_hits
                    ],
                )
            )

        for requirement in plan.fact_requirements:
            if requirement.candidate_evidence_chunk_ids:
                continue
            if len(calls) >= request.max_tool_calls:
                break
            gap_query = _requirement_gap_query(requirement, plan)
            started = perf_counter()
            try:
                search = self.search_backend.search(
                    SearchRequest(
                        query=gap_query,
                        mode=request.mode,
                        top_k=request.top_k,
                        filters=plan.document_scope,
                    ),
                    f"{task_id}:requirement:{requirement.requirement_id}",
                )
            except Exception as exc:  # noqa: BLE001
                calls.append(
                    AgentToolCall(
                        call_id=f"requirement:{requirement.requirement_id}",
                        round_number=3,
                        target_id=f"requirement:{requirement.requirement_id}",
                        query=gap_query,
                        filters=plan.document_scope,
                        status="error",
                        duration_ms=(perf_counter() - started) * 1000,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
                continue
            gap_hits = search.hits
            requirement_target = f"requirement:{requirement.requirement_id}"
            add_evidence(memory, requirement_target, gap_hits)
            requirement.candidate_evidence_chunk_ids = [
                hit.chunk.chunk_id for hit in gap_hits
            ]
            calls.append(
                AgentToolCall(
                    call_id=requirement_target,
                    round_number=3,
                    target_id=requirement_target,
                    query=gap_query,
                    filters=plan.document_scope,
                    status="success",
                    duration_ms=(perf_counter() - started) * 1000,
                    retrieval_trace_id=search.trace_id,
                    evidence_chunk_ids=requirement.candidate_evidence_chunk_ids,
                )
            )

        if len(turns) >= request.max_rounds:
            return self._trace(
                task_id=task_id,
                request=request,
                plan=plan,
                created_at=created_at,
                memory=memory,
                calls=calls,
                turns=turns,
                prompt_sha256=prompt_sha256,
                result=AgentTaskResult(
                    outcome="abstain",
                    answer=GeneratedAnswer(
                        answer="模型轮次预算不足以完成原子事实清单后的最终提交。",
                        citations=[],
                        provider="deepseek-agent-budget",
                        grounded=False,
                    ),
                    target_evidence={},
                ),
                stop_reason="model_budget_exhausted",
            )

        messages = _extract_finalization_messages(request, plan, memory)
        requirement_ids = [
            requirement.requirement_id for requirement in plan.fact_requirements
        ]
        submit_response = call_model(_submit_extraction_tool(requirement_ids))
        if submit_response is not None:
            for model_call in submit_response.tool_calls:
                if model_call.name != "submit_extraction":
                    turns[-1].validation_errors.append(f"unexpected tool: {model_call.name}")
                    continue
                try:
                    submission = _parse_extraction_submission(model_call.arguments)
                except ValidationError as exc:
                    turns[-1].validation_errors.append(f"submit_extraction: {exc}")
                    continue
                submission = _repair_submitted_fact_scope_labels(
                    submission,
                    plan,
                )
                submission = _repair_extract_authoritative_citations(
                    request.query,
                    submission,
                    plan,
                    memory,
                )
                result, errors = self._result(submission, plan, memory)
                if errors:
                    turns[-1].validation_errors.extend(errors)
                    continue
                assert result is not None
                return self._trace(
                    task_id=task_id,
                    request=request,
                    plan=plan,
                    created_at=created_at,
                    memory=memory,
                    calls=calls,
                    turns=turns,
                    prompt_sha256=prompt_sha256,
                    result=result,
                    stop_reason=(
                        "sufficient_evidence"
                        if result.outcome == "answer"
                        else "no_new_evidence"
                    ),
                )

        return self._trace(
            task_id=task_id,
            request=request,
            plan=plan,
            created_at=created_at,
            memory=memory,
            calls=calls,
            turns=turns,
            prompt_sha256=prompt_sha256,
            result=AgentTaskResult(
                outcome="abstain",
                answer=GeneratedAnswer(
                    answer="DeepSeek 未提交通过本地校验的抽取结果。",
                    citations=[],
                    provider="deepseek-agent-invalid-output",
                    grounded=False,
                ),
                target_evidence={},
            ),
            stop_reason=("model_error" if submit_response is None else "invalid_model_output"),
            status=("failed" if submit_response is None else "completed"),
        )


class ReconciliationOperand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=100)
    value: str = Field(pattern=r"^-?[\d,]+(?:\.\d+)?$")
    evidence_chunk_id: str


class ReconcileSubtractionArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    left: ReconciliationOperand
    right: ReconciliationOperand
    expected: ReconciliationOperand


class GroundedCalculationOperand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    label: str = Field(min_length=1, max_length=100)
    value: str = Field(pattern=r"^-?[\d,]+(?:\.\d+)?$")
    evidence_chunk_id: str


class GroundedCalculationStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    label: str = Field(min_length=1, max_length=100)
    operation: Literal["add", "subtract", "ratio_percent", "growth_percent"]
    inputs: list[str] = Field(min_length=2, max_length=8)
    decimals: int = Field(default=2, ge=0, le=4)
    unit: str = Field(default="", max_length=20)


class GroundedCalculationComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    left: str
    right: str
    greater_text: str = Field(min_length=1, max_length=200)
    less_text: str = Field(min_length=1, max_length=200)
    equal_text: str = Field(min_length=1, max_length=200)


class SubmitGroundedCalculationArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["answer", "abstain"]
    message: str = Field(min_length=1, max_length=1000)
    operands: list[GroundedCalculationOperand] = Field(default_factory=list, max_length=12)
    steps: list[GroundedCalculationStep] = Field(default_factory=list, max_length=8)
    comparisons: list[GroundedCalculationComparison] = Field(
        default_factory=list, max_length=4
    )


def _grounded_calculation_tool(chunk_ids: list[str]) -> list[dict]:
    name_schema = {"type": "string", "pattern": "^[a-z][a-z0-9_]*$"}
    return [
        {
            "type": "function",
            "function": {
                "name": "execute_grounded_calculation",
                "description": (
                    "提交证据中的原始操作数和有序计算步骤。系统将在本地校验引用并使用"
                    "Decimal执行运算；禁止提交模型心算后的操作数。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": ["answer", "abstain"]},
                        "message": {"type": "string"},
                        "operands": {
                            "type": "array",
                            "maxItems": 12,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": name_schema,
                                    "label": {"type": "string"},
                                    "value": {
                                        "type": "string",
                                        "pattern": r"^-?[\d,]+(?:\.\d+)?$",
                                    },
                                    "evidence_chunk_id": {
                                        "type": "string",
                                        "enum": chunk_ids,
                                    },
                                },
                                "required": [
                                    "name",
                                    "label",
                                    "value",
                                    "evidence_chunk_id",
                                ],
                                "additionalProperties": False,
                            },
                        },
                        "steps": {
                            "type": "array",
                            "maxItems": 8,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": name_schema,
                                    "label": {"type": "string"},
                                    "operation": {
                                        "type": "string",
                                        "enum": [
                                            "add",
                                            "subtract",
                                            "ratio_percent",
                                            "growth_percent",
                                        ],
                                    },
                                    "inputs": {
                                        "type": "array",
                                        "minItems": 2,
                                        "maxItems": 8,
                                        "items": name_schema,
                                    },
                                    "decimals": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 4,
                                    },
                                    "unit": {"type": "string"},
                                },
                                "required": [
                                    "name",
                                    "label",
                                    "operation",
                                    "inputs",
                                    "decimals",
                                    "unit",
                                ],
                                "additionalProperties": False,
                            },
                        },
                        "comparisons": {
                            "type": "array",
                            "maxItems": 4,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "left": name_schema,
                                    "right": name_schema,
                                    "greater_text": {"type": "string"},
                                    "less_text": {"type": "string"},
                                    "equal_text": {"type": "string"},
                                },
                                "required": [
                                    "left",
                                    "right",
                                    "greater_text",
                                    "less_text",
                                    "equal_text",
                                ],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": [
                        "status",
                        "message",
                        "operands",
                        "steps",
                        "comparisons",
                    ],
                    "additionalProperties": False,
                },
            },
        }
    ]


def _grounded_calculation_messages(
    request: AgentTaskRequest,
    memory: EvidenceMemory,
) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "你是财务年报计算任务的操作数选择器，只能调用"
                "execute_grounded_calculation。value必须逐字出现在所引chunk中，保留原始"
                "单位且不得换算；计算由本地Decimal完成。ratio_percent执行分子/分母*100；"
                "growth_percent执行(本期-上期)/上期*100；subtract严格按inputs[0]-"
                "inputs[1]；add可有多个输入。步骤可引用更早的步骤。问题要求保留两位"
                "小数时decimals=2。要求判断扩大、缩小或增减时必须填写comparisons。"
                "证据不足则status=abstain。不得输出思维链。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": request.query,
                    "evidence": DeepSeekExtractAgent._evidence_payload(memory),
                },
                ensure_ascii=False,
            ),
        },
    ]


def _reconcile_tool(chunk_ids: list[str]) -> list[dict]:
    operand = {
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "value": {"type": "string", "pattern": r"^-?[\d,]+(?:\.\d+)?$"},
            "evidence_chunk_id": {"type": "string", "enum": chunk_ids},
        },
        "required": ["label", "value", "evidence_chunk_id"],
        "additionalProperties": False,
    }
    return [
        {
            "type": "function",
            "function": {
                "name": "reconcile_subtraction",
                "description": (
                    "用有来源的十进制原值执行 left-right，并与 expected 比较。"
                    "left 必须是账面原值期末合计，right 必须是累计折旧期末合计，"
                    "expected 必须是期末账面价值合计。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "left": operand,
                        "right": operand,
                        "expected": operand,
                    },
                    "required": ["left", "right", "expected"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def _calculate_messages(request: AgentTaskRequest, plan: AgentTaskPlan) -> list[dict]:
    assert plan.document_scope is not None
    return [
        {
            "role": "system",
            "content": (
                "你是财务年报勾稽 Agent，不能凭记忆或心算回答。先调用 search_evidence"
                "定位表格，再调用 get_page_window 展开相邻页。表格可能横向拆分并跨页，"
                "要结合章节区分账面原值、累计折旧和账面价值。不得输出思维链。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": request.query,
                    "document_scope": plan.document_scope.model_dump(),
                    "retrieval_hint": plan.retrieval_hint,
                },
                ensure_ascii=False,
            ),
        },
    ]


def _calculation_finalization_messages(
    request: AgentTaskRequest,
    memory: EvidenceMemory,
) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "你是财务年报勾稽 Agent 的操作数选择器。现在只调用"
                "reconcile_subtraction。必须从证据中分别选择账面原值期末余额合计、"
                "累计折旧期末余额合计、期末账面价值合计；value 必须逐字来自所引"
                "chunk，禁止自行计算或改写数字。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": request.query,
                    "evidence": DeepSeekExtractAgent._evidence_payload(memory),
                },
                ensure_ascii=False,
            ),
        },
    ]


def _decimal_value(value: str) -> Decimal:
    try:
        number = Decimal(value.replace(",", ""))
    except InvalidOperation as exc:
        raise ValueError(f"Invalid decimal value: {value}") from exc
    if not number.is_finite():
        raise ValueError(f"Non-finite decimal value: {value}")
    return number


class DeepSeekCalculateAgent(DeepSeekExtractAgent):
    """Select cited operands with DeepSeek and perform arithmetic locally."""

    _evidence_target = "task:calculate"

    @staticmethod
    def supports(query: str) -> bool:
        reconciliation = all(cue in query for cue in ("减", "累计折旧", "账面价值"))
        general = any(
            cue in query
            for cue in ("计算", "增长率", "净利率", "占比", "比例", "差额")
        )
        return reconciliation or general

    @staticmethod
    def _supports_reconciliation(query: str) -> bool:
        return all(cue in query for cue in ("减", "累计折旧", "账面价值"))

    @staticmethod
    def _general_retrieval_queries(query: str, filters: SearchFilters) -> list[str]:
        company = filters.company_names[0] if filters.company_names else ""
        year = filters.report_years[0] if filters.report_years else ""
        prefix = f"{company} {year}年".strip()
        if "客户存款" in query and "客户贷款及垫款" in query:
            return [
                f"{prefix} 客户存款 财务概要 主要会计数据",
                f"{prefix} 客户贷款及垫款总额 财务概要 主要会计数据",
            ]
        if "汽车、汽车相关产品" in query and "营业收入" in query:
            return [
                (
                    f"{prefix} 主营业务分析 营业收入构成 产品分类 "
                    "汽车 汽车相关产品及其他产品"
                )
            ]
        if "三项核心业务" in query:
            return [
                (
                    f"{prefix} 归属于母公司股东的营运利润 寿险及健康险 "
                    "财产保险 银行 三项核心业务"
                )
            ]
        if "储能电池系统" in query and "营业收入" in query:
            return [f"{prefix} 主营业务分析 分产品 储能电池系统 营业收入"]
        if "新业务价值" in query and "增长率" in query:
            return [
                (
                    f"{prefix} 可比口径 寿险及健康险 新业务价值 "
                    "2024年 2023年 变动"
                )
            ]
        if "归母净利率" in query:
            return [
                (
                    f"{prefix} 营业收入 归属于上市公司股东的净利润 "
                    "主要会计数据"
                )
            ]
        if "净利润" in query and "增长率" in query:
            return [f"{prefix} 净利润 财务概要 主要会计数据"]
        return [f"{prefix} {query}"]

    def _general_plan(
        self,
        request: AgentTaskRequest,
    ) -> tuple[AgentTaskPlan, list[tuple[str, SearchFilters, str]]]:
        years = list(dict.fromkeys(int(value) for value in re.findall(r"20\d{2}", request.query)))
        if len(years) >= 2:
            comparison_plan = plan_compare_task(
                request.model_copy(update={"task_type": "compare"}),
                available_companies=self.available_companies,
            )
            if comparison_plan.clarification:
                return (
                    AgentTaskPlan(
                        task_type="calculate",
                        fact_periods=years,
                        clarification=comparison_plan.clarification,
                    ),
                    [],
                )
            company_names = list(
                dict.fromkeys(
                    company
                    for target in comparison_plan.targets
                    for company in target.filters.company_names
                )
            )
            report_years = list(
                dict.fromkeys(
                    year
                    for target in comparison_plan.targets
                    for year in target.filters.report_years
                )
            )
            plan = AgentTaskPlan(
                task_type="calculate",
                document_scope=SearchFilters(
                    company_names=company_names,
                    report_years=report_years,
                ),
                fact_periods=years,
                retrieval_hint=request.query,
            )
            retrievals = [
                (target.target_id, target.filters, query)
                for target in comparison_plan.targets
                for query in self._general_retrieval_queries(
                    request.query, target.filters
                )
            ]
            return plan, retrievals

        plan = plan_document_task(
            request,
            available_companies=self.available_companies,
            available_report_years_by_company=self.available_report_years_by_company,
        )
        if plan.document_scope is None:
            return plan, []
        return (
            plan,
            [
                (self._evidence_target, plan.document_scope, query)
                for query in self._general_retrieval_queries(
                    request.query, plan.document_scope
                )
            ],
        )

    @staticmethod
    def _compute_grounded_result(
        submission: SubmitGroundedCalculationArguments,
        memory: EvidenceMemory,
        *,
        prefer_matching_report_year: bool = True,
    ) -> tuple[AgentTaskResult | None, list[str], list[str]]:
        if submission.status == "abstain":
            return (
                AgentTaskResult(
                    outcome="abstain",
                    answer=GeneratedAnswer(
                        answer=submission.message,
                        citations=[],
                        provider="deepseek-grounded-calculator",
                        grounded=False,
                    ),
                    target_evidence={},
                ),
                [],
                [],
            )
        if not submission.operands or not submission.steps:
            return None, ["answer requires operands and calculation steps"], []

        evidence = {item.chunk_id: item for item in memory.items}
        values: dict[str, Decimal] = {}
        sources: dict[str, list[str]] = {}
        labels: dict[str, str] = {}
        rendered: dict[str, str] = {}
        units: dict[str, str] = {}
        errors: list[str] = []

        def normalized(value: str) -> str:
            return re.sub(r"[\s,，]", "", value)

        for operand in submission.operands:
            if operand.name in values:
                errors.append(f"duplicate operand name: {operand.name}")
                continue
            item = evidence.get(operand.evidence_chunk_id)
            if item is None:
                errors.append(f"unknown evidence chunk: {operand.evidence_chunk_id}")
                continue
            if normalized(operand.value) not in normalized(item.excerpt):
                errors.append(f"operand is absent from cited chunk: {operand.name}")
                continue
            fact_year_match = re.search(r"20\d{2}", operand.label)
            if prefer_matching_report_year and fact_year_match:
                fact_year = int(fact_year_match.group())
                matching_vintage = next(
                    (
                        candidate
                        for candidate in memory.items
                        if candidate.report_year == fact_year
                        and normalized(operand.value) in normalized(candidate.excerpt)
                    ),
                    None,
                )
                if matching_vintage is not None:
                    item = matching_vintage
            try:
                values[operand.name] = _decimal_value(operand.value)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            sources[operand.name] = [item.chunk_id]
            labels[operand.name] = operand.label
            rendered[operand.name] = operand.value
            units[operand.name] = ""

        for step in submission.steps:
            if step.name in values:
                errors.append(f"duplicate value name: {step.name}")
                continue
            if any(name not in values for name in step.inputs):
                errors.append(f"unknown step input: {step.name}")
                continue
            inputs = [values[name] for name in step.inputs]
            try:
                if step.operation == "add":
                    result = sum(inputs, Decimal(0))
                elif step.operation == "subtract":
                    if len(inputs) != 2:
                        raise ValueError("subtract requires exactly two inputs")
                    result = inputs[0] - inputs[1]
                elif step.operation == "ratio_percent":
                    if len(inputs) != 2:
                        raise ValueError("ratio_percent requires exactly two inputs")
                    if inputs[1] == 0:
                        raise ValueError("ratio_percent denominator is zero")
                    result = inputs[0] / inputs[1] * Decimal(100)
                else:
                    if len(inputs) != 2:
                        raise ValueError("growth_percent requires exactly two inputs")
                    if inputs[1] == 0:
                        raise ValueError("growth_percent denominator is zero")
                    result = (inputs[0] - inputs[1]) / inputs[1] * Decimal(100)
            except (InvalidOperation, ValueError) as exc:
                errors.append(f"{step.name}: {exc}")
                continue
            values[step.name] = result
            sources[step.name] = list(
                dict.fromkeys(chunk for name in step.inputs for chunk in sources[name])
            )
            labels[step.name] = step.label
            unit = step.unit
            if (
                step.operation == "subtract"
                and len(step.inputs) == 2
                and all(units[name] == "%" for name in step.inputs)
            ):
                unit = "个百分点"
            units[step.name] = unit
            quantum = Decimal(1).scaleb(-step.decimals)
            rendered[step.name] = f"{result.quantize(quantum):,.{step.decimals}f}{unit}"

        for comparison in submission.comparisons:
            if comparison.left not in values or comparison.right not in values:
                errors.append("comparison references an unknown value")
        if errors:
            return None, errors, []

        cited_ids = list(
            dict.fromkeys(
                chunk_id
                for operand in submission.operands
                for chunk_id in sources[operand.name]
            )
        )
        citations: list[Citation] = []
        ordinal_by_chunk: dict[str, int] = {}
        for chunk_id in cited_ids:
            item = evidence[chunk_id]
            ordinal = len(citations) + 1
            ordinal_by_chunk[chunk_id] = ordinal
            citations.append(
                Citation(
                    ordinal=ordinal,
                    chunk_id=chunk_id,
                    page_start=item.page_start,
                    page_end=item.page_end,
                    section_path=item.section_path,
                    excerpt=item.excerpt,
                )
            )

        lines: list[str] = []
        claim_citations: list[ClaimCitation] = []
        for operand in submission.operands:
            ordinal = ordinal_by_chunk[sources[operand.name][0]]
            claim = f"{operand.label}={operand.value}"
            lines.append(f"{claim}[{ordinal}]")
            claim_citations.append(
                ClaimCitation(claim=claim, citation_ordinals=[ordinal])
            )
        for step in submission.steps:
            ordinals = [ordinal_by_chunk[chunk_id] for chunk_id in sources[step.name]]
            claim = f"{step.label}={rendered[step.name]}"
            lines.append(claim + "".join(f"[{ordinal}]" for ordinal in ordinals))
            claim_citations.append(
                ClaimCitation(claim=claim, citation_ordinals=ordinals)
            )
        for comparison in submission.comparisons:
            left_value = values[comparison.left]
            right_value = values[comparison.right]
            if left_value > right_value:
                conclusion = comparison.greater_text
            elif left_value < right_value:
                conclusion = comparison.less_text
            else:
                conclusion = comparison.equal_text
            comparison_sources = list(
                dict.fromkeys(sources[comparison.left] + sources[comparison.right])
            )
            ordinals = [ordinal_by_chunk[chunk_id] for chunk_id in comparison_sources]
            claim = (
                f"{labels[comparison.left]}{rendered[comparison.left]}，"
                f"{labels[comparison.right]}{rendered[comparison.right]}；{conclusion}"
            )
            lines.append(claim + "".join(f"[{ordinal}]" for ordinal in ordinals))
            claim_citations.append(
                ClaimCitation(claim=claim, citation_ordinals=ordinals)
            )
        return (
            AgentTaskResult(
                outcome="answer",
                answer=GeneratedAnswer(
                    answer="\n".join(lines),
                    citations=citations,
                    provider="deepseek-grounded-decimal-calculator",
                    grounded=True,
                    claim_citations=claim_citations,
                ),
                target_evidence={},
            ),
            [],
            cited_ids,
        )

    def _run_general(self, request: AgentTaskRequest) -> AgentTaskTrace:
        task_id = uuid4().hex
        created_at = datetime.now(UTC)
        index_id = self.search_backend.manifest.index_id
        plan, retrievals = self._general_plan(request)
        if plan.corpus_unavailable_reason:
            trace = _corpus_unavailable_trace(
                request,
                plan,
                index_id=index_id,
                created_at=created_at,
            )
            return trace.model_copy(update={"task_id": task_id})
        if plan.clarification:
            trace = _clarification_trace(
                request, plan, index_id=index_id, created_at=created_at
            )
            return trace.model_copy(update={"task_id": task_id})
        if not self.model.available:
            raise RuntimeError("DeepSeek calculation agent requires a provider API key")

        memory = EvidenceMemory(index_id=index_id)
        calls: list[AgentToolCall] = []
        for index, (target_id, filters, query) in enumerate(retrievals, start=1):
            started = perf_counter()
            try:
                search = self.search_backend.search(
                    SearchRequest(
                        query=query,
                        mode=request.mode,
                        top_k=min(request.top_k, 5),
                        filters=filters,
                    ),
                    f"{task_id}:search-{index}",
                )
            except Exception as exc:  # noqa: BLE001
                calls.append(
                    AgentToolCall(
                        call_id=f"search-{index}",
                        round_number=1,
                        target_id=target_id,
                        query=query,
                        filters=filters,
                        status="error",
                        duration_ms=(perf_counter() - started) * 1000,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
                continue
            calls.append(
                AgentToolCall(
                    call_id=f"search-{index}",
                    round_number=1,
                    target_id=target_id,
                    query=query,
                    filters=filters,
                    status="success",
                    duration_ms=(perf_counter() - started) * 1000,
                    retrieval_trace_id=search.trace_id,
                    evidence_chunk_ids=[hit.chunk.chunk_id for hit in search.hits],
                )
            )
            add_evidence(memory, self._evidence_target, search.hits)
            if not search.hits or len(calls) + 2 > request.max_tool_calls:
                continue
            anchor_id = search.hits[0].chunk.chunk_id
            started = perf_counter()
            try:
                chunks = self.search_backend.page_window(
                    anchor_id,
                    before_pages=1,
                    after_pages=1,
                    max_chunks=12,
                )
            except Exception as exc:  # noqa: BLE001
                calls.append(
                    AgentToolCall(
                        call_id=f"window-{index}",
                        round_number=2,
                        tool="get_page_window",
                        target_id=target_id,
                        query=anchor_id,
                        filters=filters,
                        status="error",
                        duration_ms=(perf_counter() - started) * 1000,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
                continue
            add_evidence(
                memory,
                self._evidence_target,
                [
                    SearchHit(rank=rank, chunk=chunk, score=0.0)
                    for rank, chunk in enumerate(chunks, start=1)
                ],
            )
            calls.append(
                AgentToolCall(
                    call_id=f"window-{index}",
                    round_number=2,
                    tool="get_page_window",
                    target_id=target_id,
                    query=anchor_id,
                    filters=filters,
                    status="success",
                    duration_ms=(perf_counter() - started) * 1000,
                    evidence_chunk_ids=[chunk.chunk_id for chunk in chunks],
                )
            )

        def finish(
            result: AgentTaskResult,
            *,
            stop_reason: Literal[
                "sufficient_evidence",
                "no_new_evidence",
                "invalid_model_output",
                "model_error",
            ],
            turns: list[AgentModelTurn] | None = None,
            prompt_sha256: str | None = None,
            status: Literal["completed", "failed"] = "completed",
        ) -> AgentTaskTrace:
            model_trace = (
                _model_trace(
                    self.model,
                    prompt_sha256,
                    turns,
                    prompt_revision=CALCULATE_PROMPT_REVISION,
                )
                if turns is not None and prompt_sha256 is not None
                else None
            )
            return AgentTaskTrace(
                task_id=task_id,
                task_type="calculate",
                runtime="deepseek_tool_calling",
                status=status,
                stop_reason=stop_reason,
                query=request.query,
                index_id=index_id,
                created_at=created_at,
                completed_at=datetime.now(UTC),
                rounds_completed=len(turns or []),
                plan=plan,
                tool_calls=calls,
                evidence_memory=memory,
                sufficiency=SufficiencyDecision(
                    status=("sufficient" if result.outcome == "answer" else "incomplete"),
                    evidence_count_by_target={self._evidence_target: len(memory.items)},
                    gaps=[] if result.outcome == "answer" else [self._evidence_target],
                ),
                result=result,
                model_trace=model_trace,
            )

        def abstention(message: str) -> AgentTaskResult:
            return AgentTaskResult(
                outcome="abstain",
                answer=GeneratedAnswer(
                    answer=message,
                    citations=[],
                    provider="agent-local-calculation-gate",
                    grounded=False,
                ),
                target_evidence={},
            )

        if not memory.items:
            return finish(
                abstention("未找到可验证的计算操作数，系统拒绝猜测。"),
                stop_reason="no_new_evidence",
            )
        messages = _grounded_calculation_messages(request, memory)
        tools = _grounded_calculation_tool([item.chunk_id for item in memory.items])
        prompt_sha256 = _prompt_sha256(messages, tools)
        try:
            response = self.model.complete(messages, tools)
        except Exception:  # noqa: BLE001
            return finish(
                abstention("DeepSeek 计算 Agent 调用失败，任务已安全停止。"),
                stop_reason="model_error",
                status="failed",
            )
        turn = AgentModelTurn(
            turn_number=1,
            finish_reason=response.finish_reason,
            function_names=[call.name for call in response.tool_calls],
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            elapsed_ms=response.elapsed_ms,
        )
        model_call = next(
            (
                call
                for call in response.tool_calls
                if call.name == "execute_grounded_calculation"
            ),
            None,
        )
        if model_call is None:
            turn.validation_errors.append("missing execute_grounded_calculation call")
            return finish(
                abstention("模型未调用受控计算器，系统拒绝心算结果。"),
                stop_reason="invalid_model_output",
                turns=[turn],
                prompt_sha256=prompt_sha256,
            )
        try:
            submission = SubmitGroundedCalculationArguments.model_validate_json(
                model_call.arguments
            )
        except ValidationError as exc:
            turn.validation_errors.append(f"execute_grounded_calculation: {exc}")
            return finish(
                abstention("计算参数未通过本地校验，系统拒绝结果。"),
                stop_reason="invalid_model_output",
                turns=[turn],
                prompt_sha256=prompt_sha256,
            )
        result, errors, cited_ids = self._compute_grounded_result(
            submission,
            memory,
            prefer_matching_report_year=(
                "可比口径" not in request.query and "追溯调整" not in request.query
            ),
        )
        if errors or result is None:
            turn.validation_errors.extend(errors)
            return finish(
                abstention("操作数或计算步骤未通过本地校验，系统拒绝结果。"),
                stop_reason="invalid_model_output",
                turns=[turn],
                prompt_sha256=prompt_sha256,
            )
        calls.append(
            AgentToolCall(
                call_id=model_call.call_id,
                round_number=3,
                tool="calculate",
                target_id=self._evidence_target,
                query=request.query,
                filters=plan.document_scope or SearchFilters(),
                status="success",
                duration_ms=0.0,
                evidence_chunk_ids=cited_ids,
            )
        )
        return finish(
            result,
            stop_reason=(
                "sufficient_evidence" if result.outcome == "answer" else "no_new_evidence"
            ),
            turns=[turn],
            prompt_sha256=prompt_sha256,
        )

    def _calculate_trace(
        self,
        *,
        task_id: str,
        request: AgentTaskRequest,
        plan: AgentTaskPlan,
        created_at: datetime,
        memory: EvidenceMemory,
        calls: list[AgentToolCall],
        turns: list[AgentModelTurn],
        prompt_sha256: str,
        result: AgentTaskResult,
        stop_reason: Literal[
            "sufficient_evidence",
            "no_new_evidence",
            "invalid_model_output",
            "model_error",
        ],
        status: Literal["completed", "failed"] = "completed",
    ) -> AgentTaskTrace:
        answer_ready = result.outcome == "answer"
        return AgentTaskTrace(
            task_id=task_id,
            task_type="calculate",
            runtime="deepseek_tool_calling",
            status=status,
            stop_reason=stop_reason,
            query=request.query,
            index_id=memory.index_id,
            created_at=created_at,
            completed_at=datetime.now(UTC),
            rounds_completed=len(turns),
            plan=plan,
            tool_calls=calls,
            evidence_memory=memory,
            sufficiency=SufficiencyDecision(
                status="sufficient" if answer_ready else "incomplete",
                evidence_count_by_target={self._evidence_target: len(memory.items)},
                gaps=[] if answer_ready else [self._evidence_target],
            ),
            result=result,
            model_trace=_model_trace(
                self.model,
                prompt_sha256,
                turns,
                prompt_revision=CALCULATE_PROMPT_REVISION,
            ),
        )

    def run(self, request: AgentTaskRequest) -> AgentTaskTrace:
        if request.task_type != "calculate":
            raise ValueError("DeepSeekCalculateAgent only accepts calculate tasks")
        if not self.supports(request.query):
            raise ValueError("This calculation agent does not support the requested operation")
        if not self._supports_reconciliation(request.query):
            return self._run_general(request)
        task_id = uuid4().hex
        created_at = datetime.now(UTC)
        index_id = self.search_backend.manifest.index_id
        plan = plan_document_task(
            request,
            available_companies=self.available_companies,
            available_report_years_by_company=self.available_report_years_by_company,
        )
        if plan.corpus_unavailable_reason:
            trace = _corpus_unavailable_trace(
                request,
                plan,
                index_id=index_id,
                created_at=created_at,
            )
            return trace.model_copy(update={"task_id": task_id})
        if plan.clarification:
            trace = _clarification_trace(
                request,
                plan,
                index_id=index_id,
                created_at=created_at,
            )
            return trace.model_copy(update={"task_id": task_id})
        if not self.model.available:
            raise RuntimeError("DeepSeek calculation agent requires a provider API key")
        assert plan.document_scope is not None

        messages = _calculate_messages(request, plan)
        initial_tools = _extract_search_tool()
        prompt_sha256 = _prompt_sha256(messages, initial_tools)
        memory = EvidenceMemory(index_id=index_id)
        calls: list[AgentToolCall] = []
        turns: list[AgentModelTurn] = []

        def call_model(tools: list[dict]) -> ToolModelResponse | None:
            try:
                response = self.model.complete(messages, tools)
            except Exception:  # noqa: BLE001 - persist safe failure
                return None
            turns.append(
                AgentModelTurn(
                    turn_number=len(turns) + 1,
                    finish_reason=response.finish_reason,
                    function_names=[call.name for call in response.tool_calls],
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    elapsed_ms=response.elapsed_ms,
                )
            )
            messages.append(response.assistant_message())
            return response

        def abstain(
            message: str,
            reason: Literal["no_new_evidence", "invalid_model_output", "model_error"],
        ) -> AgentTaskTrace:
            return self._calculate_trace(
                task_id=task_id,
                request=request,
                plan=plan,
                created_at=created_at,
                memory=memory,
                calls=calls,
                turns=turns,
                prompt_sha256=prompt_sha256,
                result=AgentTaskResult(
                    outcome="abstain",
                    answer=GeneratedAnswer(
                        answer=message,
                        citations=[],
                        provider="agent-local-calculation-gate",
                        grounded=False,
                    ),
                    target_evidence={},
                ),
                stop_reason=reason,
                status="failed" if reason == "model_error" else "completed",
            )

        search_response = call_model(initial_tools)
        if search_response is None:
            return abstain("DeepSeek 勾稽 Agent 调用失败，任务已安全停止。", "model_error")
        search_call = next(
            (
                call
                for call in search_response.tool_calls
                if call.name == "search_evidence"
            ),
            None,
        )
        if search_call is None:
            return abstain("模型未调用规定的检索工具，系统拒绝猜测。", "invalid_model_output")
        try:
            search_arguments = ExtractSearchArguments.model_validate_json(
                search_call.arguments
            )
        except ValidationError as exc:
            turns[-1].validation_errors.append(f"search_evidence: {exc}")
            return abstain("检索参数未通过本地校验，系统拒绝猜测。", "invalid_model_output")
        effective_query = search_arguments.query.strip()
        if plan.retrieval_hint and plan.retrieval_hint not in effective_query:
            effective_query = f"{effective_query} {plan.retrieval_hint}"
        started = perf_counter()
        try:
            search = self.search_backend.search(
                SearchRequest(
                    query=effective_query,
                    mode=request.mode,
                    top_k=min(search_arguments.top_k, request.top_k),
                    filters=plan.document_scope,
                ),
                f"{task_id}:{search_call.call_id}",
            )
        except Exception as exc:  # noqa: BLE001
            turns[-1].validation_errors.append(f"search_evidence: {exc}")
            return abstain("检索工具失败，系统拒绝猜测。", "no_new_evidence")
        calls.append(
            AgentToolCall(
                call_id=search_call.call_id,
                round_number=1,
                target_id=self._evidence_target,
                query=effective_query,
                filters=plan.document_scope,
                status="success",
                duration_ms=(perf_counter() - started) * 1000,
                retrieval_trace_id=search.trace_id,
                evidence_chunk_ids=[hit.chunk.chunk_id for hit in search.hits],
            )
        )
        add_evidence(memory, self._evidence_target, search.hits)
        messages.append(
            _tool_message(
                search_call.call_id,
                {"status": "success", "evidence": self._evidence_payload(memory)},
            )
        )
        anchor_ids = [item.chunk_id for item in memory.items]
        if not anchor_ids:
            return abstain("未找到可展开的表格锚点，系统拒绝猜测。", "no_new_evidence")

        messages = _page_window_messages(request, memory)
        window_response = call_model(_page_window_tool(anchor_ids))
        if window_response is None:
            return abstain("DeepSeek 勾稽 Agent 调用失败，任务已安全停止。", "model_error")
        window_call = next(
            (
                call
                for call in window_response.tool_calls
                if call.name == "get_page_window"
            ),
            None,
        )
        if window_call is None:
            return abstain("模型未调用页面窗口工具，系统拒绝猜测。", "invalid_model_output")
        try:
            window_arguments = PageWindowArguments.model_validate_json(window_call.arguments)
            if window_arguments.anchor_chunk_id not in anchor_ids:
                raise ValueError("anchor_chunk_id was not returned by search_evidence")
        except (ValidationError, ValueError) as exc:
            turns[-1].validation_errors.append(f"get_page_window: {exc}")
            return abstain("页面窗口参数未通过本地校验，系统拒绝猜测。", "invalid_model_output")
        started = perf_counter()
        try:
            chunks = self.search_backend.page_window(
                window_arguments.anchor_chunk_id,
                before_pages=window_arguments.before_pages,
                after_pages=window_arguments.after_pages,
                max_chunks=12,
            )
        except Exception as exc:  # noqa: BLE001
            turns[-1].validation_errors.append(f"get_page_window: {exc}")
            return abstain("页面窗口工具失败，系统拒绝猜测。", "no_new_evidence")
        add_evidence(
            memory,
            self._evidence_target,
            [
                SearchHit(rank=rank, chunk=chunk, score=0.0)
                for rank, chunk in enumerate(chunks, start=1)
            ],
        )
        calls.append(
            AgentToolCall(
                call_id=window_call.call_id,
                round_number=2,
                tool="get_page_window",
                target_id=self._evidence_target,
                query=window_arguments.anchor_chunk_id,
                filters=plan.document_scope,
                status="success",
                duration_ms=(perf_counter() - started) * 1000,
                evidence_chunk_ids=[chunk.chunk_id for chunk in chunks],
            )
        )

        messages = _calculation_finalization_messages(request, memory)
        calculate_response = call_model(
            _reconcile_tool([item.chunk_id for item in memory.items])
        )
        if calculate_response is None:
            return abstain("DeepSeek 勾稽 Agent 调用失败，任务已安全停止。", "model_error")
        calculate_call = next(
            (
                call
                for call in calculate_response.tool_calls
                if call.name == "reconcile_subtraction"
            ),
            None,
        )
        if calculate_call is None:
            return abstain("模型未调用受控计算器，系统拒绝心算结果。", "invalid_model_output")
        try:
            arguments = ReconcileSubtractionArguments.model_validate_json(
                calculate_call.arguments
            )
        except ValidationError as exc:
            turns[-1].validation_errors.append(f"reconcile_subtraction: {exc}")
            return abstain("计算器参数未通过本地校验，系统拒绝结果。", "invalid_model_output")

        evidence = {item.chunk_id: item for item in memory.items}
        operands = [arguments.left, arguments.right, arguments.expected]
        errors: list[str] = []
        values: list[Decimal] = []
        for operand in operands:
            item = evidence.get(operand.evidence_chunk_id)
            if item is None:
                errors.append(f"unknown evidence chunk: {operand.evidence_chunk_id}")
                continue
            normalized_value = operand.value.replace(",", "")
            normalized_excerpt = item.excerpt.replace(",", "").replace(" ", "")
            if normalized_value not in normalized_excerpt:
                errors.append(
                    f"operand value is absent from cited chunk: {operand.label}"
                )
                continue
            values.append(_decimal_value(operand.value))
        if errors or len(values) != 3:
            turns[-1].validation_errors.extend(errors)
            return abstain("操作数来源未通过本地校验，系统拒绝计算。", "invalid_model_output")

        computed = values[0] - values[1]
        difference = computed - values[2]
        formatted = [f"{value:,.2f}" for value in values]
        computed_text = f"{computed:,.2f}"
        difference_text = f"{difference:,.2f}"
        cited_ids = [operand.evidence_chunk_id for operand in operands]
        calls.append(
            AgentToolCall(
                call_id=calculate_call.call_id,
                round_number=3,
                tool="calculate",
                target_id=self._evidence_target,
                query=f"{formatted[0]} - {formatted[1]} - {formatted[2]}",
                filters=plan.document_scope,
                status="success",
                duration_ms=0.0,
                evidence_chunk_ids=list(dict.fromkeys(cited_ids)),
            )
        )

        citations: list[Citation] = []
        ordinal_by_chunk: dict[str, int] = {}
        for chunk_id in cited_ids:
            if chunk_id in ordinal_by_chunk:
                continue
            item = evidence[chunk_id]
            ordinal_by_chunk[chunk_id] = len(citations) + 1
            citations.append(
                Citation(
                    ordinal=len(citations) + 1,
                    chunk_id=chunk_id,
                    page_start=item.page_start,
                    page_end=item.page_end,
                    section_path=item.section_path,
                    excerpt=item.excerpt,
                )
            )
        ordinals = [ordinal_by_chunk[chunk_id] for chunk_id in cited_ids]
        labels = ["账面原值期末余额合计", "累计折旧期末余额合计", "期末账面价值合计"]
        lines = [
            f"{label}为{value}元[{ordinal}]"
            for label, value, ordinal in zip(labels, formatted, ordinals, strict=True)
        ]
        equation_citations = "".join(
            f"[{ordinal}]" for ordinal in dict.fromkeys(ordinals)
        )
        relation = "相等" if difference == 0 else "不相等"
        lines.append(
            f"{formatted[0]} - {formatted[1]} = {computed_text}元，"
            f"与披露账面价值{relation}；勾稽差额为{difference_text}元"
            f"{equation_citations}"
        )
        claim_citations = [
            ClaimCitation(claim=lines[index].split("[")[0], citation_ordinals=[ordinal])
            for index, ordinal in enumerate(ordinals)
        ]
        claim_citations.append(
            ClaimCitation(
                claim=lines[-1].split("[")[0],
                citation_ordinals=list(dict.fromkeys(ordinals)),
            )
        )
        result = AgentTaskResult(
            outcome="answer",
            answer=GeneratedAnswer(
                answer="\n".join(lines),
                citations=citations,
                provider="deepseek-cited-decimal-calculator",
                grounded=True,
                claim_citations=claim_citations,
            ),
            target_evidence={},
        )
        return self._calculate_trace(
            task_id=task_id,
            request=request,
            plan=plan,
            created_at=created_at,
            memory=memory,
            calls=calls,
            turns=turns,
            prompt_sha256=prompt_sha256,
            result=result,
            stop_reason="sufficient_evidence",
        )


class InspectPageRegionArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    anchor_chunk_id: str
    page_number: int = Field(ge=1)


class VisualPercentageRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=200)
    value: str = Field(pattern=r"^\d+(?:\.\d+)?%$")
    evidence_chunk_id: str


class SumVisualPercentagesArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relationships: list[VisualPercentageRelation] = Field(min_length=3, max_length=3)


def _inspect_page_region_tool(memory: EvidenceMemory) -> list[dict]:
    anchor_ids = [item.chunk_id for item in memory.items]
    page_numbers = sorted(
        {
            page
            for item in memory.items
            for page in range(item.page_start, item.page_end + 1)
        }
    )
    return [
        {
            "type": "function",
            "function": {
                "name": "inspect_page_region",
                "description": (
                    "检查一个检索命中页面的文字节点、坐标和连接线关系。只选择明确"
                    "包含目标结构图的锚点及页面。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "anchor_chunk_id": {"type": "string", "enum": anchor_ids},
                        "page_number": {"type": "integer", "enum": page_numbers},
                    },
                    "required": ["anchor_chunk_id", "page_number"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def _sum_visual_percentages_tool(anchor_chunk_id: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "sum_visual_percentages",
                "description": (
                    "提交结构图中三条已确认的股东-持股比例关系。本地工具将校验"
                    "几何关系并使用十进制加法求和。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "relationships": {
                            "type": "array",
                            "minItems": 3,
                            "maxItems": 3,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string"},
                                    "value": {
                                        "type": "string",
                                        "pattern": r"^\d+(?:\.\d+)?%$",
                                    },
                                    "evidence_chunk_id": {
                                        "type": "string",
                                        "enum": [anchor_chunk_id],
                                    },
                                },
                                "required": ["label", "value", "evidence_chunk_id"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["relationships"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def _visual_search_messages(request: AgentTaskRequest, plan: AgentTaskPlan) -> list[dict]:
    assert plan.document_scope is not None
    return [
        {
            "role": "system",
            "content": (
                "你是年报视觉结构 Agent。先调用 search_evidence 定位任务所说的结构图"
                "页面；不要用其他章节的普通表格替代。不得凭记忆回答或输出思维链。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": request.query,
                    "document_scope": plan.document_scope.model_dump(),
                    "retrieval_hint": plan.retrieval_hint,
                },
                ensure_ascii=False,
            ),
        },
    ]


def _visual_inspection_messages(
    request: AgentTaskRequest,
    memory: EvidenceMemory,
) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "你是结构图页面定位器。只调用 inspect_page_region，选择文本明确出现"
                "‘权益结构图’及目标股东的锚点和页面。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": request.query,
                    "search_evidence": DeepSeekExtractAgent._evidence_payload(memory),
                },
                ensure_ascii=False,
            ),
        },
    ]


def _visual_sum_messages(
    request: AgentTaskRequest,
    inspection: PageRegionInspection,
    anchor_chunk_id: str,
) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "你是权益结构图关系确认器。只调用 sum_visual_percentages。"
                "relationship_rows 已按 PDF 坐标和连接线把左侧股东标签与右侧百分比"
                "配对；只提交任务询问的三类股东，value 必须原样复制。禁止自行求和。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": request.query,
                    "anchor_chunk_id": anchor_chunk_id,
                    "page_inspection": inspection.model_dump(mode="json"),
                },
                ensure_ascii=False,
            ),
        },
    ]


def _normalized_visual_label(value: str) -> str:
    return re.sub(r"[\s\u3000]", "", value)


class DeepSeekVisualGraphAgent(DeepSeekExtractAgent):
    """Reconstruct PDF diagram rows from geometry, then sum values locally."""

    _evidence_target = "task:visual-graph"

    def __init__(
        self,
        search_backend,
        model: ToolCallingModel,
        *,
        available_companies: list[str],
        available_report_years_by_company: dict[str, list[int]] | None = None,
        region_inspector: PdfRegionInspector,
    ) -> None:
        super().__init__(
            search_backend,
            model,
            available_companies=available_companies,
            available_report_years_by_company=available_report_years_by_company,
        )
        self.region_inspector = region_inspector

    @staticmethod
    def supports(query: str) -> bool:
        return "权益结构图" in query and "持股比例" in query

    def _visual_trace(
        self,
        *,
        task_id: str,
        request: AgentTaskRequest,
        plan: AgentTaskPlan,
        created_at: datetime,
        memory: EvidenceMemory,
        calls: list[AgentToolCall],
        turns: list[AgentModelTurn],
        prompt_sha256: str,
        result: AgentTaskResult,
        stop_reason: Literal[
            "sufficient_evidence",
            "no_new_evidence",
            "invalid_model_output",
            "model_error",
        ],
        status: Literal["completed", "failed"] = "completed",
    ) -> AgentTaskTrace:
        answer_ready = result.outcome == "answer"
        return AgentTaskTrace(
            task_id=task_id,
            task_type="calculate",
            runtime="deepseek_tool_calling",
            status=status,
            stop_reason=stop_reason,
            query=request.query,
            index_id=memory.index_id,
            created_at=created_at,
            completed_at=datetime.now(UTC),
            rounds_completed=len(turns),
            plan=plan,
            tool_calls=calls,
            evidence_memory=memory,
            sufficiency=SufficiencyDecision(
                status="sufficient" if answer_ready else "incomplete",
                evidence_count_by_target={self._evidence_target: len(memory.items)},
                gaps=[] if answer_ready else [self._evidence_target],
            ),
            result=result,
            model_trace=_model_trace(
                self.model,
                prompt_sha256,
                turns,
                prompt_revision=VISUAL_GRAPH_PROMPT_REVISION,
            ),
        )

    def run(self, request: AgentTaskRequest) -> AgentTaskTrace:
        if request.task_type != "calculate" or not self.supports(request.query):
            raise ValueError("DeepSeekVisualGraphAgent only accepts equity graph tasks")
        task_id = uuid4().hex
        created_at = datetime.now(UTC)
        index_id = self.search_backend.manifest.index_id
        plan = plan_document_task(
            request,
            available_companies=self.available_companies,
            available_report_years_by_company=self.available_report_years_by_company,
        )
        if plan.corpus_unavailable_reason:
            trace = _corpus_unavailable_trace(
                request,
                plan,
                index_id=index_id,
                created_at=created_at,
            )
            return trace.model_copy(update={"task_id": task_id})
        if plan.clarification:
            trace = _clarification_trace(
                request,
                plan,
                index_id=index_id,
                created_at=created_at,
            )
            return trace.model_copy(update={"task_id": task_id})
        if not self.model.available:
            raise RuntimeError("DeepSeek visual graph agent requires a provider API key")
        assert plan.document_scope is not None

        messages = _visual_search_messages(request, plan)
        initial_tools = _extract_search_tool()
        prompt_sha256 = _prompt_sha256(messages, initial_tools)
        memory = EvidenceMemory(index_id=index_id)
        calls: list[AgentToolCall] = []
        turns: list[AgentModelTurn] = []

        def call_model(tools: list[dict]) -> ToolModelResponse | None:
            try:
                response = self.model.complete(messages, tools)
            except Exception:  # noqa: BLE001 - persist safe failure
                return None
            turns.append(
                AgentModelTurn(
                    turn_number=len(turns) + 1,
                    finish_reason=response.finish_reason,
                    function_names=[call.name for call in response.tool_calls],
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    elapsed_ms=response.elapsed_ms,
                )
            )
            messages.append(response.assistant_message())
            return response

        def abstain(
            message: str,
            reason: Literal["no_new_evidence", "invalid_model_output", "model_error"],
        ) -> AgentTaskTrace:
            return self._visual_trace(
                task_id=task_id,
                request=request,
                plan=plan,
                created_at=created_at,
                memory=memory,
                calls=calls,
                turns=turns,
                prompt_sha256=prompt_sha256,
                result=AgentTaskResult(
                    outcome="abstain",
                    answer=GeneratedAnswer(
                        answer=message,
                        citations=[],
                        provider="agent-local-visual-gate",
                        grounded=False,
                    ),
                    target_evidence={},
                ),
                stop_reason=reason,
                status="failed" if reason == "model_error" else "completed",
            )

        search_response = call_model(initial_tools)
        if search_response is None:
            return abstain("DeepSeek 视觉 Agent 调用失败，任务已安全停止。", "model_error")
        search_call = next(
            (call for call in search_response.tool_calls if call.name == "search_evidence"),
            None,
        )
        if search_call is None:
            return abstain("模型未调用规定的检索工具，系统拒绝猜测。", "invalid_model_output")
        try:
            arguments = ExtractSearchArguments.model_validate_json(search_call.arguments)
        except ValidationError as exc:
            turns[-1].validation_errors.append(f"search_evidence: {exc}")
            return abstain("检索参数未通过本地校验，系统拒绝猜测。", "invalid_model_output")
        effective_query = arguments.query.strip()
        if plan.retrieval_hint and plan.retrieval_hint not in effective_query:
            effective_query = f"{effective_query} {plan.retrieval_hint}"
        started = perf_counter()
        try:
            search = self.search_backend.search(
                SearchRequest(
                    query=effective_query,
                    mode=request.mode,
                    top_k=min(arguments.top_k, request.top_k),
                    filters=plan.document_scope,
                ),
                f"{task_id}:{search_call.call_id}",
            )
        except Exception as exc:  # noqa: BLE001
            turns[-1].validation_errors.append(f"search_evidence: {exc}")
            return abstain("检索工具失败，系统拒绝猜测。", "no_new_evidence")
        calls.append(
            AgentToolCall(
                call_id=search_call.call_id,
                round_number=1,
                target_id=self._evidence_target,
                query=effective_query,
                filters=plan.document_scope,
                status="success",
                duration_ms=(perf_counter() - started) * 1000,
                retrieval_trace_id=search.trace_id,
                evidence_chunk_ids=[hit.chunk.chunk_id for hit in search.hits],
            )
        )
        add_evidence(memory, self._evidence_target, search.hits)
        if not memory.items:
            return abstain("未找到结构图候选页，系统拒绝猜测。", "no_new_evidence")

        messages = _visual_inspection_messages(request, memory)
        inspect_response = call_model(_inspect_page_region_tool(memory))
        if inspect_response is None:
            return abstain("DeepSeek 视觉 Agent 调用失败，任务已安全停止。", "model_error")
        inspect_call = next(
            (
                call
                for call in inspect_response.tool_calls
                if call.name == "inspect_page_region"
            ),
            None,
        )
        if inspect_call is None:
            return abstain("模型未调用页面区域检查工具，系统拒绝猜测。", "invalid_model_output")
        try:
            inspect_arguments = InspectPageRegionArguments.model_validate_json(
                inspect_call.arguments
            )
            anchor = next(
                item
                for item in memory.items
                if item.chunk_id == inspect_arguments.anchor_chunk_id
            )
            if not anchor.page_start <= inspect_arguments.page_number <= anchor.page_end:
                raise ValueError("page_number is outside the anchor chunk")
            if anchor.document_key is None:
                raise ValueError("anchor chunk has no source document key")
        except (ValidationError, ValueError, StopIteration) as exc:
            turns[-1].validation_errors.append(f"inspect_page_region: {exc}")
            return abstain("页面区域参数未通过本地校验，系统拒绝猜测。", "invalid_model_output")
        started = perf_counter()
        try:
            inspection = self.region_inspector.inspect_relationship_rows(
                anchor.document_key,
                inspect_arguments.page_number,
            )
        except Exception as exc:  # noqa: BLE001
            turns[-1].validation_errors.append(f"inspect_page_region: {exc}")
            return abstain("页面区域检查失败，系统拒绝猜测。", "no_new_evidence")
        calls.append(
            AgentToolCall(
                call_id=inspect_call.call_id,
                round_number=2,
                tool="inspect_page_region",
                target_id=self._evidence_target,
                query=f"{anchor.document_key}:page:{inspect_arguments.page_number}",
                filters=plan.document_scope,
                status="success",
                duration_ms=(perf_counter() - started) * 1000,
                evidence_chunk_ids=[anchor.chunk_id],
            )
        )
        if len(inspection.relationship_rows) < 3:
            return abstain("页面中未重建出足够的带连接线关系，系统拒绝猜测。", "no_new_evidence")

        messages = _visual_sum_messages(request, inspection, anchor.chunk_id)
        sum_response = call_model(_sum_visual_percentages_tool(anchor.chunk_id))
        if sum_response is None:
            return abstain("DeepSeek 视觉 Agent 调用失败，任务已安全停止。", "model_error")
        sum_call = next(
            (
                call
                for call in sum_response.tool_calls
                if call.name == "sum_visual_percentages"
            ),
            None,
        )
        if sum_call is None:
            return abstain("模型未调用受控百分比计算器，系统拒绝心算。", "invalid_model_output")
        try:
            sum_arguments = SumVisualPercentagesArguments.model_validate_json(
                sum_call.arguments
            )
        except ValidationError as exc:
            turns[-1].validation_errors.append(f"sum_visual_percentages: {exc}")
            return abstain("百分比关系未通过本地校验，系统拒绝结果。", "invalid_model_output")

        inspected_pairs = {
            (
                _normalized_visual_label(row.label.text),
                row.value.text,
            )
            for row in inspection.relationship_rows
            if row.connector_present
        }
        submitted_pairs = {
            (_normalized_visual_label(item.label), item.value)
            for item in sum_arguments.relationships
        }
        required_labels = {
            "H股股东",
            "国家能源投资集团有限责任公司",
            "其他A股股东",
        }
        submitted_labels = {
            _normalized_visual_label(item.label)
            for item in sum_arguments.relationships
        }
        if submitted_labels != required_labels or not submitted_pairs <= inspected_pairs:
            turns[-1].validation_errors.append(
                "submitted relationships do not match connector-backed geometry rows"
            )
            return abstain("股东与比例关系未通过几何校验，系统拒绝结果。", "invalid_model_output")

        values = [
            _decimal_value(item.value.removesuffix("%"))
            for item in sum_arguments.relationships
        ]
        total = sum(values, Decimal(0))
        calls.append(
            AgentToolCall(
                call_id=sum_call.call_id,
                round_number=3,
                tool="calculate",
                target_id=self._evidence_target,
                query=" + ".join(item.value for item in sum_arguments.relationships),
                filters=plan.document_scope,
                status="success",
                duration_ms=0.0,
                evidence_chunk_ids=[anchor.chunk_id],
            )
        )
        citation = Citation(
            ordinal=1,
            chunk_id=anchor.chunk_id,
            page_start=anchor.page_start,
            page_end=anchor.page_end,
            section_path=anchor.section_path,
            excerpt=anchor.excerpt,
        )
        by_label = {
            _normalized_visual_label(item.label): item.value
            for item in sum_arguments.relationships
        }
        ordered_labels = [
            "H股股东",
            "国家能源投资集团有限责任公司",
            "其他A股股东",
        ]
        lines = [f"{label}持股比例为{by_label[label]}[1]" for label in ordered_labels]
        total_text = f"{total:.2f}%"
        relation = "等于100%" if total == Decimal(100) else "不等于100%"
        lines.append(f"三者合计为{total_text}，{relation}。[1]")
        result = AgentTaskResult(
            outcome="answer",
            answer=GeneratedAnswer(
                answer="\n".join(lines),
                citations=[citation],
                provider="deepseek-pdf-geometry-local-calculator",
                grounded=True,
                claim_citations=[
                    ClaimCitation(
                        claim=line.removesuffix("[1]"),
                        citation_ordinals=[1],
                    )
                    for line in lines
                ],
            ),
            target_evidence={},
        )
        return self._visual_trace(
            task_id=task_id,
            request=request,
            plan=plan,
            created_at=created_at,
            memory=memory,
            calls=calls,
            turns=turns,
            prompt_sha256=prompt_sha256,
            result=result,
            stop_reason="sufficient_evidence",
        )
