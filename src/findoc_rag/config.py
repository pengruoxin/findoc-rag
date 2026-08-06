import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ServerSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: Literal["critical", "error", "warning", "info", "debug", "trace"] = "info"


class RetrievalSettings(BaseModel):
    index_dir: Path = Path("data/indexes/default")
    default_mode: Literal["lexical", "dense", "hybrid"] = "hybrid"
    top_k: int = Field(default=5, ge=1, le=100)
    candidate_k: int = Field(default=50, ge=1, le=1000)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    lexical_weight: float = Field(default=2.0, gt=0)
    dense_weight: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def validate_candidate_count(self) -> "RetrievalSettings":
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")
        return self


class ObservabilitySettings(BaseModel):
    enabled: bool = True
    trace_db: Path = Path("data/traces/retrieval.sqlite3")
    capture_query_text: bool = False
    max_recorded_hits: int = Field(default=20, ge=1, le=100)


class RerankerSettings(BaseModel):
    enabled: bool = False
    model: str = "BAAI/bge-reranker-v2-m3"
    batch_size: int = Field(default=16, ge=1, le=256)


class ScopeRoutingSettings(BaseModel):
    enabled: bool = False
    adaptive_candidate_budget: bool = False
    max_candidate_k: int = Field(default=100, ge=1, le=1000)


class AnswerGenerationSettings(BaseModel):
    enabled: bool = False
    model: str = "deepseek-chat"
    endpoint: str = "https://api.deepseek.com/chat/completions"


class AppSettings(BaseModel):
    server: ServerSettings = Field(default_factory=ServerSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    scope_routing: ScopeRoutingSettings = Field(default_factory=ScopeRoutingSettings)
    answer_generation: AnswerGenerationSettings = Field(default_factory=AnswerGenerationSettings)


ENVIRONMENT_PATHS: dict[str, tuple[str, str]] = {
    "FINDOC_RAG_HOST": ("server", "host"),
    "FINDOC_RAG_PORT": ("server", "port"),
    "FINDOC_RAG_LOG_LEVEL": ("server", "log_level"),
    "FINDOC_RAG_INDEX_DIR": ("retrieval", "index_dir"),
    "FINDOC_RAG_DEFAULT_MODE": ("retrieval", "default_mode"),
    "FINDOC_RAG_TOP_K": ("retrieval", "top_k"),
    "FINDOC_RAG_CANDIDATE_K": ("retrieval", "candidate_k"),
    "FINDOC_RAG_RRF_K": ("retrieval", "rrf_k"),
    "FINDOC_RAG_TRACE_DB": ("observability", "trace_db"),
    "FINDOC_RAG_TRACING_ENABLED": ("observability", "enabled"),
    "FINDOC_RAG_CAPTURE_QUERY_TEXT": ("observability", "capture_query_text"),
    "FINDOC_RAG_MAX_RECORDED_HITS": ("observability", "max_recorded_hits"),
    "FINDOC_RAG_RERANKER_ENABLED": ("reranker", "enabled"),
    "FINDOC_RAG_RERANKER_MODEL": ("reranker", "model"),
    "FINDOC_RAG_RERANKER_BATCH_SIZE": ("reranker", "batch_size"),
    "FINDOC_RAG_SCOPE_ROUTING_ENABLED": ("scope_routing", "enabled"),
    "FINDOC_RAG_ADAPTIVE_CANDIDATE_BUDGET": (
        "scope_routing",
        "adaptive_candidate_budget",
    ),
    "FINDOC_RAG_MAX_CANDIDATE_K": ("scope_routing", "max_candidate_k"),
    "FINDOC_RAG_ANSWER_ENABLED": ("answer_generation", "enabled"),
    "FINDOC_RAG_ANSWER_MODEL": ("answer_generation", "model"),
    "FINDOC_RAG_ANSWER_ENDPOINT": ("answer_generation", "endpoint"),
}


def load_settings(
    config_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> AppSettings:
    """Load defaults, a TOML file, then explicit environment overrides."""
    values: dict = {}
    base_directory = Path.cwd()
    if config_path is not None:
        resolved_config = config_path.resolve(strict=True)
        with resolved_config.open("rb") as source:
            values = tomllib.load(source)
        base_directory = resolved_config.parent

    environment = environ if environ is not None else os.environ
    for variable, (section, field) in ENVIRONMENT_PATHS.items():
        if variable in environment:
            values.setdefault(section, {})[field] = environment[variable]

    settings = AppSettings.model_validate(values)
    index_dir = settings.retrieval.index_dir
    if not index_dir.is_absolute():
        settings.retrieval.index_dir = (base_directory / index_dir).resolve()
    trace_db = settings.observability.trace_db
    if not trace_db.is_absolute():
        settings.observability.trace_db = (base_directory / trace_db).resolve()
    return settings
