import asyncio
import hashlib
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pymupdf
import pytest

from findoc_rag.answer_generation import ClaimCitation, GeneratedAnswer
from findoc_rag.api import create_app
from findoc_rag.config import (
    AppSettings,
    IngestionSettings,
    ObservabilitySettings,
    RetrievalSettings,
    ScopeRoutingSettings,
)
from findoc_rag.corpus import CurrentIndexPointer
from findoc_rag.documents.models import BoundingBox, DocumentChunk, ElementReference
from findoc_rag.indexing import PersistentIndex
from findoc_rag.query_rewriting import LLMQueryRewriter
from findoc_rag.query_routing import infer_finance_filters, prepare_finance_query


def test_infer_finance_filters_maps_aliases_and_tickers() -> None:
    companies, years = infer_finance_filters("600519 2024 年营收是多少")
    assert companies == ["贵州茅台"]
    assert years == [2024]
    companies, years = infer_finance_filters("伊利去年营收和净利")
    assert companies == ["伊利股份"]
    assert years == []


def test_prepare_finance_query_resolves_relative_time_and_synonyms() -> None:
    resolved = prepare_finance_query(
        "贵州茅台去年营收是多少",
        as_of_date=date(2025, 4, 30),
        rewrite_mode="deterministic",
    )
    assert "2024年" in resolved
    assert "营业收入" in resolved


def test_prepare_finance_query_llm_mode_falls_back_without_key() -> None:
    rewriter = LLMQueryRewriter(api_key="")
    resolved = prepare_finance_query(
        "贵州茅台去年营收是多少",
        as_of_date=date(2025, 4, 30),
        rewrite_mode="llm",
        rewriter=rewriter,
    )
    assert "2024年" in resolved
    assert "营业收入" in resolved


def build_test_index(tmp_path: Path) -> Path:
    chunks = [
        DocumentChunk(
            chunk_id="chunk-0",
            document_id="doc-0",
            chunk_index=0,
            text="公司2024年营业收入为一百亿元。",
            section_path=["主要财务指标"],
            page_start=5,
            page_end=5,
            element_references=[
                ElementReference(
                    element_id="element-0",
                    page_number=5,
                    bbox=BoundingBox(x0=10, y0=20, x1=200, y1=80),
                )
            ],
            character_count=20,
            estimated_token_count=20,
        )
    ]
    source = tmp_path / "chunks.jsonl"
    source.write_text(chunks[0].model_dump_json() + "\n", encoding="utf-8")
    index_dir = tmp_path / "index"
    PersistentIndex.build(index_dir, chunks, source)
    return index_dir


def app_settings(index_dir: Path, tmp_path: Path) -> AppSettings:
    return AppSettings(
        retrieval=RetrievalSettings(
            index_dir=index_dir,
            default_mode="lexical",
            top_k=5,
            candidate_k=20,
        ),
        observability=ObservabilitySettings(trace_db=tmp_path / "traces.sqlite3"),
    )


async def request_app(app, requests: list[tuple[str, str, dict | None, dict | None]]):
    responses = []
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            for method, path, payload, headers in requests:
                responses.append(await client.request(method, path, json=payload, headers=headers))
    return responses


async def upload_and_process(app, pdf_bytes: bytes):
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            uploaded = await client.post(
                "/v1/uploads",
                content=pdf_bytes,
                headers={"X-Filename": "report.pdf"},
            )
            job_id = uploaded.json()["job_id"]
            started = await client.post(
                f"/v1/uploads/{job_id}:process",
                json={
                    "document_key": "company:annual:2024",
                    "metadata": {
                        "company_name": "甲公司",
                        "report_year": 2024,
                        "document_type": "annual",
                    },
                },
            )
            status = await client.get(f"/v1/uploads/{job_id}")
    return uploaded, started, status


def test_health_index_and_search_endpoints(tmp_path: Path) -> None:
    index_dir = build_test_index(tmp_path)
    settings = app_settings(index_dir, tmp_path)

    live, ready, info, search = asyncio.run(
        request_app(
            create_app(settings),
            [
                ("GET", "/health/live", None, None),
                ("GET", "/health/ready", None, None),
                ("GET", "/v1/index", None, None),
                (
                    "POST",
                    "/v1/search",
                    {"query": "营业收入", "top_k": 1},
                    {"X-Request-ID": "test-request"},
                ),
            ],
        )
    )

    assert live.status_code == 200
    assert ready.json()["status"] == "ready"
    assert info.json()["chunk_count"] == 1
    assert search.status_code == 200
    assert search.headers["X-Request-ID"] == "test-request"
    assert search.json()["request_id"] == "test-request"
    assert search.json()["hits"][0]["chunk"]["chunk_id"] == "chunk-0"
    assert search.json()["hits"][0]["chunk"]["page_start"] == 5

    trace_id = search.json()["trace_id"]
    trace, metrics = asyncio.run(
        request_app(
            create_app(settings),
            [
                ("GET", f"/v1/traces/{trace_id}", None, None),
                ("GET", "/v1/metrics", None, None),
            ],
        )
    )
    assert trace.json()["query_text"] is None
    assert trace.json()["stages"][0]["stage"] == "lexical"
    assert metrics.json()["success_count"] == 1


def test_upload_is_explicitly_disabled_by_default(tmp_path: Path) -> None:
    app = create_app(app_settings(build_test_index(tmp_path), tmp_path))

    [response] = asyncio.run(
        request_app(
            app,
            [("POST", "/v1/uploads", None, {"X-Filename": "report.pdf"})],
        )
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "ingestion_disabled"


def test_upload_job_persists_and_reaches_index_bound_ready_state(
    tmp_path: Path,
) -> None:
    index_dir = build_test_index(tmp_path)
    settings = app_settings(index_dir, tmp_path).model_copy(
        update={
            "ingestion": IngestionSettings(
                enabled=True,
                upload_root=tmp_path / "uploads",
                registry_path=tmp_path / "catalog" / "registry.sqlite3",
                storage_dir=tmp_path / "versions",
                index_root=tmp_path / "corpus-index",
            )
        }
    )
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Annual revenue was 100 million in 2024. " * 10)
    pdf_bytes = pdf.tobytes()
    pdf.close()

    uploaded, started, status = asyncio.run(upload_and_process(create_app(settings), pdf_bytes))

    body = status.json()
    assert uploaded.status_code == 202
    assert started.status_code == 202
    assert started.json()["status"] == "validating"
    assert status.status_code == 200
    assert body["status"] == "ready"
    assert body["document_version_id"]
    assert body["index_id"]
    assert (tmp_path / "uploads" / body["job_id"] / "job.json").is_file()

    [reopened] = asyncio.run(
        request_app(
            create_app(settings),
            [("GET", f"/v1/uploads/{body['job_id']}", None, None)],
        )
    )
    assert reopened.json()["index_id"] == body["index_id"]


def test_capabilities_match_lexical_only_runtime(tmp_path: Path) -> None:
    index_dir = build_test_index(tmp_path)
    settings = app_settings(index_dir, tmp_path)

    [response] = asyncio.run(
        request_app(create_app(settings), [("GET", "/v1/capabilities", None, None)])
    )

    body = response.json()
    assert response.status_code == 200
    assert body["contract_version"] == "1.0"
    assert body["index_id"] == PersistentIndex(index_dir).manifest.index_id
    assert body["modes"] == ["lexical"]
    assert body["features"] == {
        "answer": True,
        "rewrite": True,
        "reranker": False,
        "scope": False,
        "tracing": True,
        "deterministic_tables": True,
        "structured_table_artifacts": False,
        "ingestion_jobs": False,
        "claim_citations": True,
        "evidence_resolve": True,
        "agent_tasks": True,
        "human_reviews": True,
        "request_api_keys": True,
    }
    assert body["filter_fields"] == [
        "document_keys",
        "company_names",
        "report_years",
        "document_types",
    ]
    assert body["limits"]["top_k"] == {"default": 5, "minimum": 1, "maximum": 100}
    assert body["limits"]["candidate_k"] == {
        "default": 20,
        "minimum": 1,
        "maximum": 1000,
    }
    assert body["limits"]["max_generation_contexts"] == 5


def test_capabilities_scope_flag_follows_runtime_configuration(tmp_path: Path) -> None:
    index_dir = build_test_index(tmp_path)
    settings = app_settings(index_dir, tmp_path).model_copy(
        update={"scope_routing": ScopeRoutingSettings(enabled=True)}
    )

    [response] = asyncio.run(
        request_app(create_app(settings), [("GET", "/v1/capabilities", None, None)])
    )

    assert response.status_code == 200
    assert response.json()["features"]["scope"] is True


def test_evidence_resolve_preserves_order_hash_and_index_binding(tmp_path: Path) -> None:
    index_dir = build_test_index(tmp_path)
    settings = app_settings(index_dir, tmp_path)
    index_id = PersistentIndex(index_dir).manifest.index_id

    success, mismatch, missing = asyncio.run(
        request_app(
            create_app(settings),
            [
                (
                    "POST",
                    "/v1/evidence:resolve",
                    {"index_id": index_id, "chunk_ids": ["chunk-0", "chunk-0"]},
                    None,
                ),
                (
                    "POST",
                    "/v1/evidence:resolve",
                    {"index_id": "stale-index", "chunk_ids": ["chunk-0"]},
                    None,
                ),
                (
                    "POST",
                    "/v1/evidence:resolve",
                    {"index_id": index_id, "chunk_ids": ["missing", "chunk-0"]},
                    None,
                ),
            ],
        )
    )

    body = success.json()
    assert success.status_code == 200
    assert body["index_id"] == index_id
    assert [item["chunk"]["chunk_id"] for item in body["evidence"]] == [
        "chunk-0",
        "chunk-0",
    ]
    chunk = DocumentChunk.model_validate(body["evidence"][0]["chunk"])
    expected_hash = hashlib.sha256(chunk.model_dump_json().encode()).hexdigest()
    assert body["evidence"][0]["sha256"] == expected_hash
    assert body["evidence"][1]["sha256"] == expected_hash
    assert mismatch.status_code == 409
    assert mismatch.json()["error"]["code"] == "index_id_mismatch"
    assert missing.status_code == 404
    assert missing.json()["error"]["details"]["missing_chunk_ids"] == ["missing"]


@pytest.mark.parametrize(
    ("generated", "expected_outcome"),
    [
        (
            GeneratedAnswer(
                answer="答案 [1]",
                citations=[],
                provider="test",
                grounded=True,
                claim_citations=[ClaimCitation(claim="答案", citation_ordinals=[1])],
            ),
            "answer",
        ),
        (
            GeneratedAnswer(answer="证据不足", citations=[], provider="abstention", grounded=False),
            "abstain",
        ),
        (
            GeneratedAnswer(
                answer="请明确口径", citations=[], provider="clarification", grounded=False
            ),
            "clarify",
        ),
        (
            GeneratedAnswer(
                answer="请查看证据", citations=[], provider="evidence-only", grounded=False
            ),
            "evidence_only",
        ),
    ],
)
def test_query_contract_exposes_agent_metadata_and_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    generated: GeneratedAnswer,
    expected_outcome: str,
) -> None:
    index_dir = build_test_index(tmp_path)
    settings = app_settings(index_dir, tmp_path)
    monkeypatch.setattr(
        "findoc_rag.answer_generation.GroundedAnswerGenerator.generate",
        lambda self, query, hits: generated,
    )

    [response] = asyncio.run(
        request_app(
            create_app(settings),
            [
                (
                    "POST",
                    "/v1/query",
                    {"query": "贵州茅台2024年营业收入是多少", "top_k": 1},
                    {"X-Request-ID": "agent-query-test"},
                )
            ],
        )
    )

    body = response.json()
    assert response.status_code == 200
    assert body["contract_version"] == "1.0"
    assert body["request_id"] == "agent-query-test"
    assert body["trace_id"]
    assert body["index_id"] == PersistentIndex(index_dir).manifest.index_id
    assert body["original_query"] == "贵州茅台2024年营业收入是多少"
    assert "营业收入" in body["resolved_query"]
    assert body["applied_filters"]["company_names"] == ["贵州茅台"]
    assert body["applied_filters"]["report_years"] == [2024]
    assert body["route"]["year_filter_policy"] == "fact_period_as_report_year"
    assert body["outcome"] == expected_outcome
    assert body["rewrite_mode"] == "deterministic"
    assert body["rewrite_gate"] == "deterministic"
    assert body["answer"] == generated.answer
    assert body["provider"] == generated.provider
    assert body["grounded"] == generated.grounded
    assert body["claim_citations"] == [item.model_dump() for item in generated.claim_citations]


def test_query_uses_request_scoped_key_without_returning_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_dir = build_test_index(tmp_path)
    settings = app_settings(index_dir, tmp_path)
    observed: dict[str, object] = {}

    def generate(self, query, hits):
        observed.update(
            api_key=self.api_key,
            enabled=self.enabled,
            model=self.model,
        )
        return GeneratedAnswer(
            answer="营业收入为一百亿元 [1]",
            citations=[],
            provider="test",
            grounded=True,
        )

    monkeypatch.setattr(
        "findoc_rag.answer_generation.GroundedAnswerGenerator.generate",
        generate,
    )
    [response] = asyncio.run(
        request_app(
            create_app(settings),
            [
                (
                    "POST",
                    "/v1/query",
                    {"query": "营业收入", "top_k": 1},
                    {"X-DeepSeek-API-Key": "sk-request-only"},
                )
            ],
        )
    )

    assert response.status_code == 200
    assert observed == {
        "api_key": "sk-request-only",
        "enabled": True,
        "model": "deepseek-chat",
    }
    assert "sk-request-only" not in response.text


def test_agent_web_endpoint_requires_provider_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    [response] = asyncio.run(
        request_app(
            create_app(app_settings(build_test_index(tmp_path), tmp_path)),
            [
                (
                    "POST",
                    "/v1/agent/tasks",
                    {"task_type": "compare", "query": "比较甲公司和乙公司营业收入"},
                    None,
                )
            ],
        )
    )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "provider_key_required"


def test_major_operation_ids_are_stable(tmp_path: Path) -> None:
    schema = create_app(app_settings(build_test_index(tmp_path), tmp_path)).openapi()
    assert schema["paths"]["/v1/search"]["post"]["operationId"] == "search_evidence"
    assert schema["paths"]["/v1/query"]["post"]["operationId"] == "query_documents"
    assert schema["paths"]["/v1/capabilities"]["get"]["operationId"] == "get_capabilities"
    assert schema["paths"]["/v1/evidence:resolve"]["post"]["operationId"] == "resolve_evidence"
    assert schema["paths"]["/v1/agent/tasks"]["post"]["operationId"] == "run_agent_task"
    assert schema["paths"]["/v1/reviews"]["get"]["operationId"] == "list_human_reviews"


def test_dense_mode_returns_clear_error_for_lexical_index(tmp_path: Path) -> None:
    index_dir = build_test_index(tmp_path)
    settings = app_settings(index_dir, tmp_path)

    [response] = asyncio.run(
        request_app(
            create_app(settings),
            [("POST", "/v1/search", {"query": "营业收入", "mode": "dense"}, None)],
        )
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_search_request"
    assert "requires an index with dense embeddings" in response.json()["error"]["message"]
    assert response.json()["trace_id"]

    [trace] = asyncio.run(
        request_app(
            create_app(settings),
            [("GET", f"/v1/traces/{response.json()['trace_id']}", None, None)],
        )
    )
    assert trace.json()["status"] == "error"
    assert trace.json()["error_type"] == "ValueError"


def test_invalid_request_id_is_not_reflected(tmp_path: Path) -> None:
    index_dir = build_test_index(tmp_path)
    settings = app_settings(index_dir, tmp_path)

    [response] = asyncio.run(
        request_app(
            create_app(settings),
            [
                (
                    "POST",
                    "/v1/search",
                    {"query": "营业收入"},
                    {"X-Request-ID": "invalid request id"},
                )
            ],
        )
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] != "invalid request id"
    assert len(response.headers["X-Request-ID"]) == 32


def test_request_validation_errors_are_structured(tmp_path: Path) -> None:
    index_dir = build_test_index(tmp_path)
    settings = app_settings(index_dir, tmp_path)

    [response] = asyncio.run(
        request_app(
            create_app(settings),
            [
                (
                    "POST",
                    "/v1/search",
                    {"query": "   "},
                    {"X-Request-ID": "validation-test"},
                )
            ],
        )
    )

    assert response.status_code == 422
    assert response.json()["request_id"] == "validation-test"
    assert response.json()["error"]["code"] == "request_validation_error"


@pytest.mark.parametrize(
    ("path", "payload", "unknown_field"),
    [
        ("/v1/search", {"query": "营业收入", "topK": 1}, "topK"),
        ("/v1/query", {"query": "营业收入", "topK": 1}, "topK"),
        (
            "/v1/search",
            {"query": "营业收入", "filters": {"report_year": [2024]}},
            "report_year",
        ),
        (
            "/v1/query",
            {"query": "营业收入", "filters": {"report_year": [2024]}},
            "report_year",
        ),
        (
            "/v1/evidence:resolve",
            {"index_id": "index", "chunk_ids": ["chunk"], "chunkIds": ["chunk"]},
            "chunkIds",
        ),
    ],
)
def test_agent_endpoints_reject_unknown_request_fields(
    tmp_path: Path,
    path: str,
    payload: dict,
    unknown_field: str,
) -> None:
    index_dir = build_test_index(tmp_path)
    settings = app_settings(index_dir, tmp_path)

    [response] = asyncio.run(request_app(create_app(settings), [("POST", path, payload, None)]))

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "request_validation_error"
    assert any(
        detail["location"][-1] == unknown_field and detail["type"] == "extra_forbidden"
        for detail in response.json()["error"]["details"]
    )


def test_service_resolves_current_generation_from_corpus_root(tmp_path: Path) -> None:
    generation = build_test_index(tmp_path)
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    pointer = CurrentIndexPointer(
        index_id=PersistentIndex(generation).manifest.index_id,
        generation_path="generations/test",
        activated_at=datetime.now(UTC),
        active_version_ids=["version-1"],
    )
    target_generation = corpus_root / pointer.generation_path
    target_generation.parent.mkdir(parents=True)
    generation.rename(target_generation)
    (corpus_root / "current.json").write_text(
        pointer.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )

    [response] = asyncio.run(
        request_app(
            create_app(app_settings(corpus_root, tmp_path)),
            [("POST", "/v1/search", {"query": "revenue", "top_k": 1}, None)],
        )
    )

    assert response.status_code == 200
    assert response.json()["index_id"] == pointer.index_id


def test_service_rejects_current_pointer_that_escapes_corpus_root(tmp_path: Path) -> None:
    generation = build_test_index(tmp_path)
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    pointer = CurrentIndexPointer(
        index_id=PersistentIndex(generation).manifest.index_id,
        generation_path="../index",
        activated_at=datetime.now(UTC),
        active_version_ids=["version-1"],
    )
    (corpus_root / "current.json").write_text(
        pointer.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )

    app = create_app(app_settings(corpus_root, tmp_path))
    with pytest.raises(ValueError, match="escapes the configured index root"):
        asyncio.run(request_app(app, [("GET", "/health/ready", None, None)]))


def test_service_rejects_current_pointer_with_mismatched_index_id(tmp_path: Path) -> None:
    generation = build_test_index(tmp_path)
    corpus_root = tmp_path / "corpus"
    target_generation = corpus_root / "generations" / "test"
    target_generation.parent.mkdir(parents=True)
    generation.rename(target_generation)
    pointer = CurrentIndexPointer(
        index_id="wrong-index-id",
        generation_path="generations/test",
        activated_at=datetime.now(UTC),
        active_version_ids=["version-1"],
    )
    (corpus_root / "current.json").write_text(
        pointer.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )

    app = create_app(app_settings(corpus_root, tmp_path))
    with pytest.raises(ValueError, match="does not match the referenced index manifest"):
        asyncio.run(request_app(app, [("GET", "/health/ready", None, None)]))
