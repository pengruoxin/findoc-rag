import asyncio
from datetime import UTC, datetime
from pathlib import Path

import httpx

from findoc_rag.api import create_app
from findoc_rag.config import AppSettings, ObservabilitySettings, RetrievalSettings
from findoc_rag.corpus import CurrentIndexPointer
from findoc_rag.documents.models import BoundingBox, DocumentChunk, ElementReference
from findoc_rag.indexing import PersistentIndex


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
                responses.append(
                    await client.request(method, path, json=payload, headers=headers)
                )
    return responses


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


def test_service_resolves_current_generation_from_corpus_root(tmp_path: Path) -> None:
    generation = build_test_index(tmp_path)
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    pointer = CurrentIndexPointer(
        index_id=PersistentIndex(generation).manifest.index_id,
        generation_path="../" + generation.relative_to(corpus_root.parent).as_posix(),
        activated_at=datetime.now(UTC),
        active_version_ids=["version-1"],
    )
    # The pointer path is relative to its own root, including a parent traversal here
    # only to keep the test fixture small.
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
