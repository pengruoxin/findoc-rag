import json
import re
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ExpectedFact(BaseModel):
    fact_id: str
    description: str
    subject: str
    predicate: str
    required: bool = True
    canonical_value: str
    value_type: Literal["number", "percentage", "percentage_point", "text", "boolean"]
    acceptable_values: list[str] = Field(default_factory=list)
    unit: str | None = None
    currency: str | None = None
    period: str
    scope: str
    tolerance: str = "0"
    derivation: str | None = None
    evidence_chunk_ids: list[str] = Field(min_length=1)


class GoldEvidence(BaseModel):
    evidence_id: str
    chunk_id: str
    document_version_id: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    section_path: list[str]
    verbatim_quote: str = Field(min_length=1)
    supports_fact_ids: list[str] = Field(min_length=1)
    pdf_visual_verified: bool = False


class HardNegative(BaseModel):
    chunk_id: str
    negative_type: Literal[
        "wrong_company",
        "wrong_period",
        "wrong_scope",
        "partial_evidence",
        "unsupported_causality",
    ]
    reason: str


class AnswerContract(BaseModel):
    expected_behavior: Literal["answer", "abstain", "clarify"]
    required_format: Literal["short", "list", "table", "comparison", "abstention"]
    require_citations: bool = True
    require_units: bool = True
    forbid_external_knowledge: bool = True


class AnnotationProvenance(BaseModel):
    created_by: Literal["assistant_curated", "human"]
    review_status: Literal["assistant_verified", "human_verified"]
    confidence: Literal["high", "medium", "low"]
    source_pdf_sha256: list[str]
    notes: str = ""


class QueryVariant(BaseModel):
    variant_id: str
    query: str
    variant_types: list[str] = Field(default_factory=list)
    query_regime: str
    as_of_date: str | None = None


class GenerationEvaluationItem(BaseModel):
    query_id: str
    family_id: str
    split: Literal["calibration", "dev", "frozen_test"]
    query: str
    company_ids: list[str]
    company_names: list[str]
    company_aliases: list[str]
    report_years: list[int]
    category: Literal[
        "single_fact",
        "multi_fact_table",
        "comparison",
        "calculation",
        "narrative",
        "accounting_policy",
        "unanswerable",
    ]
    difficulty: Literal["easy", "medium", "hard"]
    answerability: Literal["answerable", "unanswerable", "needs_clarification"]
    reference_answer: str
    expected_facts: list[ExpectedFact] = Field(default_factory=list)
    gold_chunk_ids: list[str] = Field(default_factory=list)
    gold_evidence: list[GoldEvidence] = Field(default_factory=list)
    hard_negatives: list[HardNegative] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    query_variants: list[QueryVariant] = Field(default_factory=list)
    answer_contract: AnswerContract
    required_citation_count: int = Field(default=1, ge=0)
    abstention_reason: str | None = None
    annotation: AnnotationProvenance
    notes: str = ""

    @model_validator(mode="after")
    def validate_answerability_contract(self) -> "GenerationEvaluationItem":
        if self.answerability == "answerable":
            if not self.expected_facts or not self.gold_chunk_ids or not self.gold_evidence:
                raise ValueError("Answerable items require facts and gold chunks")
            if self.abstention_reason:
                raise ValueError("Answerable items cannot define an abstention reason")
            if self.required_citation_count != len(set(self.gold_chunk_ids)):
                raise ValueError("Citation count must equal the number of unique gold chunks")
        else:
            if self.expected_facts or self.gold_chunk_ids or self.gold_evidence:
                raise ValueError("Unanswerable items cannot contain unsupported gold evidence")
            if not self.abstention_reason or self.required_citation_count != 0:
                raise ValueError("Unanswerable items require an abstention reason and zero citations")
        if self.answer_contract.expected_behavior == "answer" and self.answerability != "answerable":
            raise ValueError("Only answerable items may require an answer")
        evidence_claims = {
            fact_id for evidence in self.gold_evidence for fact_id in evidence.supports_fact_ids
        }
        all_claims = {fact.fact_id for fact in self.expected_facts}
        unknown_claims = evidence_claims - all_claims
        if unknown_claims:
            raise ValueError(f"Evidence references unknown facts: {sorted(unknown_claims)}")
        required_claims = {fact.fact_id for fact in self.expected_facts if fact.required}
        if not required_claims.issubset(evidence_claims):
            raise ValueError("Every required fact must be bound to gold evidence")
        return self


class GenerationEvaluationDataset(BaseModel):
    schema_version: int = 1
    dataset_id: str
    corpus_index_id: str
    independent_gold: bool = False
    reviewer: str
    status: Literal["assistant_curated_provisional", "human_frozen"]
    tracks: list[Literal["oracle_context", "retrieved_context", "robustness"]]
    item_count: int
    items: list[GenerationEvaluationItem]

    @model_validator(mode="after")
    def validate_dataset(self) -> "GenerationEvaluationDataset":
        if self.item_count != len(self.items):
            raise ValueError("item_count does not match items")
        if len({item.query_id for item in self.items}) != len(self.items):
            raise ValueError("query_id values must be unique")
        return self


class DatasetValidationReport(BaseModel):
    dataset_id: str
    item_count: int
    answerability_counts: dict[str, int]
    split_counts: dict[str, int]
    category_counts: dict[str, int]
    difficulty_counts: dict[str, int]
    company_counts: dict[str, int]
    tag_counts: dict[str, int]
    family_count: int
    unique_gold_chunk_count: int
    fact_count: int
    evidence_count: int
    multi_evidence_item_count: int
    derived_fact_count: int
    robustness_item_count: int
    robustness_split_counts: dict[str, int]
    hard_negative_count: int
    hard_negative_type_counts: dict[str, int]
    warning_count: int
    warnings: list[str]


class GenerationRunItem(BaseModel):
    query_id: str
    response: str
    retrieved_contexts: list[str]
    retrieved_chunk_ids: list[str]
    context_labels: list[str] = Field(default_factory=list)
    provider: str
    model: str
    api_model: str | None = None
    index_id: str
    prompt_sha256: str
    latency_ms: float = Field(ge=0)
    grounded: bool
    as_of_date: str | None = None
    resolved_query: str | None = None
    time_cues: list[str] = Field(default_factory=list)
    context_tokens: int | None = None
    observed_behavior: Literal["answer", "abstain", "clarify"] | None = None
    error: str | None = None

    @model_validator(mode="after")
    def validate_context_provenance(self) -> "GenerationRunItem":
        if len(self.retrieved_contexts) != len(self.retrieved_chunk_ids):
            raise ValueError("Context text and chunk ID counts must match")
        if self.context_labels and len(self.context_labels) != len(self.retrieved_chunk_ids):
            raise ValueError("Context labels and chunk ID counts must match")
        return self


GenerationLane = Literal["oracle_context", "retrieved_context", "robustness"]


class RagasRunSelection(BaseModel):
    """Auditable scope selected from one immutable generation run for RAGAS."""

    lane: GenerationLane
    scope_policy: Literal["full_dataset", "hard_negative_subset"]
    dataset_query_count: int
    dataset_answerable_count: int
    lane_query_count: int
    lane_answerable_count: int
    run_item_count: int
    eligible_count: int
    coverage: float = Field(ge=0, le=1)
    lane_coverage: float = Field(ge=0, le=1)
    run_query_ids: list[str]
    lane_query_ids: list[str]
    eligible_query_ids: list[str]
    excluded_non_answerable_query_ids: list[str]
    behavior_mismatch_query_ids: list[str] = Field(default_factory=list)
    run_error_query_ids: list[str] = Field(default_factory=list)


def select_ragas_run_items(
    dataset: GenerationEvaluationDataset,
    run_items: list[GenerationRunItem],
    lane: GenerationLane,
) -> tuple[list[GenerationEvaluationItem], RagasRunSelection]:
    """Validate a lane run and return its answerable, semantically judgeable cases.

    Oracle and retrieved runs must contain the complete dataset. A robustness run
    must contain exactly the dataset items carrying controlled hard negatives. Run
    errors are rejected instead of being silently removed from the semantic-metric
    denominator. Behavior mismatches remain eligible and are recorded explicitly,
    because excluding wrong answers or false abstentions would create survivor bias.
    """

    dataset_by_id = {item.query_id: item for item in dataset.items}
    run_query_ids = [item.query_id for item in run_items]
    duplicate_ids = sorted(
        query_id for query_id, count in Counter(run_query_ids).items() if count > 1
    )
    if duplicate_ids:
        raise ValueError(f"Run contains duplicate query IDs: {duplicate_ids}")

    unknown_ids = sorted(set(run_query_ids) - set(dataset_by_id))
    if unknown_ids:
        raise ValueError(f"Run contains query IDs absent from the dataset: {unknown_ids}")

    lane_items = (
        [item for item in dataset.items if item.hard_negatives]
        if lane == "robustness"
        else list(dataset.items)
    )
    lane_query_ids = [item.query_id for item in lane_items]
    missing_ids = sorted(set(lane_query_ids) - set(run_query_ids))
    out_of_lane_ids = sorted(set(run_query_ids) - set(lane_query_ids))
    if missing_ids or out_of_lane_ids:
        raise ValueError(
            f"Run query scope does not match lane {lane!r}; "
            f"missing={missing_ids}, out_of_lane={out_of_lane_ids}"
        )

    wrong_index_ids = sorted(
        item.query_id for item in run_items if item.index_id != dataset.corpus_index_id
    )
    if wrong_index_ids:
        raise ValueError(
            "Run items use an index other than the dataset corpus index: "
            f"{wrong_index_ids}"
        )

    run_errors = sorted(item.query_id for item in run_items if item.error is not None)
    if run_errors:
        raise ValueError(f"Run contains failed items: {run_errors}")

    run_by_id = {item.query_id: item for item in run_items}
    behavior_mismatches: list[str] = []
    invalid_answer_payloads: list[str] = []
    for item in lane_items:
        run_item = run_by_id[item.query_id]
        observed = run_item.observed_behavior or (
            "answer" if run_item.grounded else "abstain"
        )
        if observed != item.answer_contract.expected_behavior:
            behavior_mismatches.append(item.query_id)
        if (observed == "answer") != run_item.grounded:
            invalid_answer_payloads.append(item.query_id)
        if observed == "answer" and (
            not run_item.response.strip() or not run_item.retrieved_contexts
        ):
            invalid_answer_payloads.append(item.query_id)
    if invalid_answer_payloads:
        raise ValueError(
            "Run answer behavior is inconsistent with grounded response/context data: "
            f"{sorted(set(invalid_answer_payloads))}"
        )

    eligible_items = [item for item in lane_items if item.answerability == "answerable"]
    eligible_ids = [item.query_id for item in eligible_items]
    excluded_ids = [item.query_id for item in lane_items if item.answerability != "answerable"]
    dataset_answerable_count = sum(
        item.answerability == "answerable" for item in dataset.items
    )
    lane_answerable_count = len(eligible_items)
    selection = RagasRunSelection(
        lane=lane,
        scope_policy=(
            "hard_negative_subset" if lane == "robustness" else "full_dataset"
        ),
        dataset_query_count=dataset.item_count,
        dataset_answerable_count=dataset_answerable_count,
        lane_query_count=len(lane_items),
        lane_answerable_count=lane_answerable_count,
        run_item_count=len(run_items),
        eligible_count=len(eligible_items),
        coverage=(
            len(eligible_items) / dataset_answerable_count
            if dataset_answerable_count
            else 0
        ),
        lane_coverage=(
            len(eligible_items) / lane_answerable_count if lane_answerable_count else 0
        ),
        run_query_ids=run_query_ids,
        lane_query_ids=lane_query_ids,
        eligible_query_ids=eligible_ids,
        excluded_non_answerable_query_ids=excluded_ids,
        behavior_mismatch_query_ids=sorted(behavior_mismatches),
        run_error_query_ids=[],
    )
    return eligible_items, selection


class DeterministicCaseScore(BaseModel):
    query_id: str
    expected_behavior_correct: bool
    strict_success: bool
    strict_success_eligible: bool = True
    semantic_review_required: bool = False
    gold_fact_recall: float | None
    numeric_accuracy: float | None
    unit_accuracy: float | None
    context_recall: float | None
    citation_validity: float | None
    false_abstention: bool


NUMBER_PATTERN = re.compile(r"(?<![\d])[-−]?\d[\d,]*(?:\.\d+)?")
CITATION_PATTERN = re.compile(r"\[(\d+)]")
NUMBER_TOKEN = r"(?<![\d])[-−]?\d[\d,]*(?:\.\d+)?"


def _numbers(text: str) -> set[Decimal]:
    values: set[Decimal] = set()
    for raw in NUMBER_PATTERN.findall(text):
        normalized = raw.replace(",", "").replace("−", "-")
        try:
            values.add(Decimal(normalized))
        except InvalidOperation:
            continue
    return values


def _decimal(raw: str) -> Decimal | None:
    try:
        return Decimal(raw.replace(",", "").replace("−", "-"))
    except InvalidOperation:
        return None


def _contains_value_with_unit(text: str, value: Decimal, unit: str) -> bool:
    """Match a fact's value and unit without collapsing financial unit scales."""
    if unit == "元/股":
        patterns = (
            rf"(?P<value>{NUMBER_TOKEN})\s*元\s*[/／]\s*股",
            rf"每\s*股[^\d−-]{{0,16}}(?P<value>{NUMBER_TOKEN})\s*元",
            rf"(?P<value>{NUMBER_TOKEN})\s*元\s*每\s*股",
        )
    else:
        escaped_unit = re.escape(unit)
        if unit == "元":
            escaped_unit += r"(?!\s*[/／]\s*股)"
        patterns = (rf"(?P<value>{NUMBER_TOKEN})\s*{escaped_unit}",)

    return any(
        _decimal(match.group("value")) == value
        for pattern in patterns
        for match in re.finditer(pattern, text)
    )


def score_generation_run_item(
    item: GenerationEvaluationItem,
    run: GenerationRunItem,
) -> DeterministicCaseScore:
    if item.query_id != run.query_id:
        raise ValueError("Dataset item and run item query_id do not match")
    expected_behavior = item.answer_contract.expected_behavior
    observed_behavior = run.observed_behavior or ("answer" if run.grounded else "abstain")
    expects_answer = expected_behavior == "answer"
    behavior_correct = observed_behavior == expected_behavior
    if not expects_answer:
        return DeterministicCaseScore(
            query_id=item.query_id,
            expected_behavior_correct=behavior_correct,
            strict_success=behavior_correct,
            strict_success_eligible=True,
            semantic_review_required=False,
            gold_fact_recall=None,
            numeric_accuracy=None,
            unit_accuracy=None,
            context_recall=None,
            citation_validity=None,
            false_abstention=False,
        )

    response_numbers = _numbers(run.response)
    fact_matches: list[bool] = []
    numeric_matches: list[bool] = []
    unit_matches: list[bool] = []
    context_matches: list[bool] = []
    for expected in item.expected_facts:
        if not expected.required:
            continue
        numeric_value: Decimal | None = None
        if expected.value_type in {"number", "percentage", "percentage_point"}:
            numeric_value = Decimal(expected.canonical_value)
            numeric_match = numeric_value in response_numbers
            numeric_matches.append(numeric_match)
            fact_matches.append(numeric_match)
        elif expected.value_type == "boolean":
            accepted = expected.acceptable_values or [expected.canonical_value]
            fact_matches.append(any(value in run.response for value in accepted))
        if expected.unit:
            unit_matches.append(
                _contains_value_with_unit(run.response, numeric_value, expected.unit)
                if numeric_value is not None
                else expected.unit in run.response
            )
        context_matches.append(
            set(expected.evidence_chunk_ids).issubset(run.retrieved_chunk_ids)
        )

    citations = [int(value) for value in CITATION_PATTERN.findall(run.response)]
    citation_valid = bool(citations) and all(
        1 <= ordinal <= len(run.retrieved_contexts) for ordinal in citations
    )
    fact_recall = sum(fact_matches) / len(fact_matches) if fact_matches else None
    numeric_accuracy = (
        sum(numeric_matches) / len(numeric_matches) if numeric_matches else None
    )
    unit_accuracy = sum(unit_matches) / len(unit_matches) if unit_matches else None
    context_recall = sum(context_matches) / len(context_matches) if context_matches else None
    semantic_review_required = any(
        expected.required and expected.value_type == "text" for expected in item.expected_facts
    )
    strict_success_eligible = not semantic_review_required
    strict_success = strict_success_eligible and (
        run.grounded
        and fact_recall == 1.0
        and (unit_accuracy is None or unit_accuracy == 1)
        and citation_valid
    )
    return DeterministicCaseScore(
        query_id=item.query_id,
        expected_behavior_correct=behavior_correct,
        strict_success=strict_success,
        strict_success_eligible=strict_success_eligible,
        semantic_review_required=semantic_review_required,
        gold_fact_recall=fact_recall,
        numeric_accuracy=numeric_accuracy,
        unit_accuracy=unit_accuracy,
        context_recall=context_recall,
        citation_validity=float(citation_valid),
        false_abstention=not run.grounded,
    )


def validate_generation_dataset(
    dataset_path: Path,
    chunk_paths: list[Path],
) -> DatasetValidationReport:
    dataset = GenerationEvaluationDataset.model_validate_json(
        dataset_path.read_text(encoding="utf-8")
    )
    chunks: dict[str, dict] = {}
    for path in chunk_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            chunk = json.loads(line)
            chunks[chunk["chunk_id"]] = chunk
    family_splits: dict[str, set[str]] = defaultdict(set)
    chunk_splits: dict[str, set[str]] = defaultdict(set)
    warnings: list[str] = []
    for item in dataset.items:
        family_splits[item.family_id].add(item.split)
        seen_negatives: set[str] = set()
        for negative in item.hard_negatives:
            if negative.chunk_id not in chunks:
                raise ValueError(f"Missing hard-negative chunk: {negative.chunk_id}")
            if negative.chunk_id in item.gold_chunk_ids:
                raise ValueError(
                    f"Hard negative duplicates gold for {item.query_id}: {negative.chunk_id}"
                )
            if negative.chunk_id in seen_negatives:
                raise ValueError(
                    f"Duplicate hard negative for {item.query_id}: {negative.chunk_id}"
                )
            seen_negatives.add(negative.chunk_id)
        for evidence in item.gold_evidence:
            chunk_splits[evidence.chunk_id].add(item.split)
            chunk = chunks.get(evidence.chunk_id)
            if chunk is None:
                raise ValueError(f"Missing gold chunk: {evidence.chunk_id}")
            if evidence.verbatim_quote not in chunk["text"]:
                raise ValueError(f"Gold quote mismatch: {evidence.evidence_id}")
            if (evidence.page_start, evidence.page_end) != (
                chunk["page_start"],
                chunk["page_end"],
            ):
                raise ValueError(f"Gold page mismatch: {evidence.evidence_id}")
            if not evidence.pdf_visual_verified:
                warnings.append(f"PDF visual review pending: {evidence.evidence_id}")
        reference_numbers = _numbers(item.reference_answer)
        for fact in item.expected_facts:
            if not fact.required:
                continue
            if fact.value_type in {"number", "percentage", "percentage_point"}:
                canonical = Decimal(fact.canonical_value)
                if canonical not in reference_numbers:
                    raise ValueError(
                        f"Reference answer omits required fact {item.query_id}:{fact.fact_id}"
                    )
                if not fact.derivation:
                    supporting_quotes = "\n".join(
                        evidence.verbatim_quote
                        for evidence in item.gold_evidence
                        if fact.fact_id in evidence.supports_fact_ids
                    )
                    if canonical not in _numbers(supporting_quotes):
                        raise ValueError(
                            "Evidence span omits direct numeric fact "
                            f"{item.query_id}:{fact.fact_id}"
                        )
            elif fact.value_type == "boolean":
                accepted = fact.acceptable_values or [fact.canonical_value]
                if not any(value in item.reference_answer for value in accepted):
                    raise ValueError(
                        f"Reference answer omits required fact {item.query_id}:{fact.fact_id}"
                    )
    leaked_families = sorted(key for key, splits in family_splits.items() if len(splits) > 1)
    leaked_chunks = sorted(key for key, splits in chunk_splits.items() if len(splits) > 1)
    if leaked_families:
        raise ValueError(f"Families cross dataset splits: {leaked_families}")
    if leaked_chunks:
        raise ValueError(f"Gold chunks cross dataset splits: {leaked_chunks}")
    answerability = Counter(item.answerability for item in dataset.items)
    split_counts = Counter(item.split for item in dataset.items)
    category_counts = Counter(item.category for item in dataset.items)
    difficulty_counts = Counter(item.difficulty for item in dataset.items)
    company_counts = Counter("+".join(item.company_names) for item in dataset.items)
    tag_counts = Counter(tag for item in dataset.items for tag in item.tags)
    hard_negative_types = Counter(
        negative.negative_type for item in dataset.items for negative in item.hard_negatives
    )
    robustness_split_counts = Counter(
        item.split for item in dataset.items if item.hard_negatives
    )
    return DatasetValidationReport(
        dataset_id=dataset.dataset_id,
        item_count=dataset.item_count,
        answerability_counts=dict(answerability),
        split_counts=dict(split_counts),
        category_counts=dict(category_counts),
        difficulty_counts=dict(difficulty_counts),
        company_counts=dict(company_counts),
        tag_counts=dict(tag_counts),
        family_count=len({item.family_id for item in dataset.items}),
        unique_gold_chunk_count=len(
            {chunk_id for item in dataset.items for chunk_id in item.gold_chunk_ids}
        ),
        fact_count=sum(len(item.expected_facts) for item in dataset.items),
        evidence_count=sum(len(item.gold_evidence) for item in dataset.items),
        multi_evidence_item_count=sum(len(item.gold_evidence) > 1 for item in dataset.items),
        derived_fact_count=sum(
            fact.derivation is not None
            for item in dataset.items
            for fact in item.expected_facts
        ),
        robustness_item_count=sum(bool(item.hard_negatives) for item in dataset.items),
        robustness_split_counts=dict(robustness_split_counts),
        hard_negative_count=sum(len(item.hard_negatives) for item in dataset.items),
        hard_negative_type_counts=dict(hard_negative_types),
        warning_count=len(warnings),
        warnings=warnings,
    )


def to_ragas_oracle_rows(dataset: GenerationEvaluationDataset) -> list[dict]:
    """Export answerable cases to RAGAS SingleTurnSample-compatible dictionaries."""
    return [
        {
            "user_input": item.query,
            "reference": item.reference_answer,
            "reference_contexts": [evidence.verbatim_quote for evidence in item.gold_evidence],
            "metadata": {
                "query_id": item.query_id,
                "split": item.split,
                "category": item.category,
                "difficulty": item.difficulty,
            },
        }
        for item in dataset.items
        if item.answerability == "answerable"
    ]


class GenerationJudgment(BaseModel):
    query_id: str
    answer: str
    faithfulness: float = Field(ge=0, le=1)
    answer_relevancy: float = Field(ge=0, le=1)
    context_relevancy: float = Field(ge=0, le=1)
    context_recall: float = Field(ge=0, le=1)
    source: Literal["human", "llm_judge"]
    judge_model: str | None = None
    rationale: str = Field(min_length=1)
    citation_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_model_for_llm_judge(self) -> "GenerationJudgment":
        if self.source == "llm_judge" and not self.judge_model:
            raise ValueError("judge_model is required for llm_judge judgments")
        return self


class GenerationEvaluation(BaseModel):
    evaluated_count: int
    faithfulness: float
    answer_relevancy: float
    context_relevancy: float
    context_recall: float
    judgments: list[GenerationJudgment]


def aggregate_generation_judgments(
    judgments: list[GenerationJudgment],
) -> GenerationEvaluation:
    if not judgments:
        raise ValueError("At least one generation judgment is required")
    count = len(judgments)
    return GenerationEvaluation(
        evaluated_count=count,
        faithfulness=sum(item.faithfulness for item in judgments) / count,
        answer_relevancy=sum(item.answer_relevancy for item in judgments) / count,
        context_relevancy=sum(item.context_relevancy for item in judgments) / count,
        context_recall=sum(item.context_recall for item in judgments) / count,
        judgments=judgments,
    )
