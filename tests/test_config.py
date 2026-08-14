from pathlib import Path

import pytest
from pydantic import ValidationError

from findoc_rag.config import load_settings


def test_toml_and_environment_configuration(tmp_path: Path) -> None:
    config = tmp_path / "findoc-rag.toml"
    config.write_text(
        """
[server]
port = 8100

[retrieval]
index_dir = "indexes/current"
default_mode = "lexical"
top_k = 5
candidate_k = 20

[observability]
trace_db = "traces/retrieval.sqlite3"

[reranker]
enabled = true
model = "test/reranker"
batch_size = 8

[scope_routing]
enabled = true
adaptive_candidate_budget = true
max_candidate_k = 80

[ingestion]
enabled = true
upload_root = "uploads"
registry_path = "catalog/registry.sqlite3"
storage_dir = "catalog/versions"
index_root = "indexes/corpus"
""".strip(),
        encoding="utf-8",
    )

    settings = load_settings(
        config,
        environ={"FINDOC_RAG_PORT": "8200", "FINDOC_RAG_TOP_K": "7"},
    )

    assert settings.server.port == 8200
    assert settings.retrieval.top_k == 7
    assert settings.retrieval.candidate_k == 20
    assert settings.retrieval.index_dir == (tmp_path / "indexes/current").resolve()
    assert settings.observability.trace_db == (tmp_path / "traces/retrieval.sqlite3").resolve()
    assert settings.reranker.enabled is True
    assert settings.reranker.model == "test/reranker"
    assert settings.reranker.batch_size == 8
    assert settings.scope_routing.enabled is True
    assert settings.scope_routing.adaptive_candidate_budget is True
    assert settings.scope_routing.max_candidate_k == 80
    assert settings.ingestion.enabled is True
    assert settings.ingestion.upload_root == (tmp_path / "uploads").resolve()
    assert settings.ingestion.registry_path == (
        tmp_path / "catalog/registry.sqlite3"
    ).resolve()


def test_configuration_rejects_candidate_count_below_top_k() -> None:
    with pytest.raises(ValidationError, match="candidate_k"):
        load_settings(
            environ={
                "FINDOC_RAG_TOP_K": "10",
                "FINDOC_RAG_CANDIDATE_K": "5",
            }
        )


def test_configuration_rejects_unknown_top_level_section(tmp_path: Path) -> None:
    config = tmp_path / "findoc-rag.toml"
    config.write_text("[typo_section]\nenabled = true\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="typo_section"):
        load_settings(config, environ={})


def test_configuration_rejects_unknown_nested_key(tmp_path: Path) -> None:
    config = tmp_path / "findoc-rag.toml"
    config.write_text("[retrieval]\ntopK = 7\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="topK"):
        load_settings(config, environ={})
