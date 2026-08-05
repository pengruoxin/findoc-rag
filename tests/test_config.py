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


def test_configuration_rejects_candidate_count_below_top_k() -> None:
    with pytest.raises(ValidationError, match="candidate_k"):
        load_settings(
            environ={
                "FINDOC_RAG_TOP_K": "10",
                "FINDOC_RAG_CANDIDATE_K": "5",
            }
        )
