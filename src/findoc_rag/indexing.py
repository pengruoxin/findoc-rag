import hashlib
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Literal
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field

from findoc_rag.documents.models import DocumentChunk, StructuredTable
from findoc_rag.structured_tables import (
    STRUCTURED_TABLE_GENERATOR,
    STRUCTURED_TABLE_SCHEMA_VERSION,
    StructuredTableArtifactManifest,
    chunk_payload_sha256,
    load_structured_tables,
    serialize_structured_tables,
)

INDEX_FORMAT_VERSION = 3
DEFAULT_DENSE_MODEL = "intfloat/multilingual-e5-small"
LATIN_OR_NUMBER = re.compile(r"[a-z0-9]+(?:[._%/-][a-z0-9]+)*", re.IGNORECASE)
CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
FINANCIAL_TERMS = (
    "营业收入", "营业成本", "净利润", "归属于上市公司股东的净利润",
    "经营活动产生的现金流量净额", "经营活动现金流量净额", "资产负债表",
    "现金流量表", "利润表", "审计委员会", "董事会报告",
)
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class IndexManifest(BaseModel):
    index_format_version: int = INDEX_FORMAT_VERSION
    index_id: str
    created_at: datetime
    source_chunk_sha256: str
    document_ids: list[str]
    chunk_count: int = Field(ge=1)
    average_document_length: float = Field(gt=0)
    tokenizer: str = "cjk-bigram-latin-v1"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    dense_model: str | None = None
    embedding_dimension: int | None = None
    parent_index_id: str | None = None
    reused_embedding_count: int = 0
    encoded_embedding_count: int = 0
    structured_table_schema_version: int | None = None
    structured_table_generator: str | None = None
    structured_table_count: int = Field(default=0, ge=0)
    structured_table_cell_count: int = Field(default=0, ge=0)
    structured_tables_sha256: str | None = None


class SearchHit(BaseModel):
    rank: int = Field(ge=1)
    chunk: DocumentChunk
    score: float
    lexical_rank: int | None = None
    lexical_score: float | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    original_rank: int | None = None
    rerank_score: float | None = None
    rank_delta: int | None = None
    retrieval_rank: int | None = None
    scope_score: int | None = None
    scope_rank_delta: int | None = None

    @computed_field
    @property
    def text(self) -> str:
        return self.chunk.text

    @computed_field
    @property
    def page_start(self) -> int:
        return self.chunk.page_start

    @computed_field
    @property
    def page_end(self) -> int:
        return self.chunk.page_end

    @computed_field
    @property
    def section_path(self) -> list[str]:
        return self.chunk.section_path


class SearchFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_keys: list[str] = Field(default_factory=list)
    company_names: list[str] = Field(default_factory=list)
    report_years: list[int] = Field(default_factory=list)
    document_types: list[str] = Field(default_factory=list)

    @property
    def active(self) -> bool:
        return any(
            (self.document_keys, self.company_names, self.report_years, self.document_types)
        )


def tokenize_for_search(text: str) -> list[str]:
    """Tokenize mixed CJK, Latin, and numeric text without a domain dictionary."""
    normalized = text.lower()
    tokens = LATIN_OR_NUMBER.findall(CJK_RUN.sub(" ", normalized))
    for match in CJK_RUN.finditer(normalized):
        run = match.group()
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    tokens.extend(term for term in FINANCIAL_TERMS if term in normalized)
    return tokens


def searchable_chunk_text(chunk: DocumentChunk) -> str:
    section = " ".join(chunk.section_path)
    return f"{section} {section} {chunk.text}".strip()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Dense dependencies are missing. Run: uv sync --extra dev --extra dense"
        ) from exc
    return SentenceTransformer(model_name)


def _dense_text(text: str, model_name: str, kind: Literal["query", "passage"]) -> str:
    lowered = model_name.lower()
    if "e5" in lowered:
        return f"{kind}: {text}"
    if "bge" in lowered and "m3" not in lowered and kind == "query":
        return f"{BGE_QUERY_INSTRUCTION}{text}"
    return text


class PersistentIndex:
    def __init__(self, directory: Path) -> None:
        self.directory = directory.resolve()
        manifest_path = self.directory / "manifest.json"
        database_path = self.directory / "lexical.sqlite3"
        if not manifest_path.is_file() or not database_path.is_file():
            raise FileNotFoundError(f"Invalid FinDocRAG index directory: {self.directory}")
        self.manifest = IndexManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.index_format_version != INDEX_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported index format {self.manifest.index_format_version}; "
                f"expected {INDEX_FORMAT_VERSION}"
            )
        self.database_path = database_path
        self._dense_model = None
        self._dense_model_lock = RLock()
        self._dense_embeddings = None
        self._dense_chunk_ids: list[str] | None = None
        self._statement_scope_cache: dict[str, str] | None = None
        self._structured_table_cache: dict[str, list[StructuredTable]] | None = None
        self.validate()

    @classmethod
    def build(
        cls,
        directory: Path,
        chunks: list[DocumentChunk],
        source_chunk_path: Path,
        dense_model: str | None = None,
        reuse_dense_from: "PersistentIndex | None" = None,
        structured_tables: list[StructuredTable] | None = None,
    ) -> "PersistentIndex":
        if not chunks:
            raise ValueError("Cannot build an index without chunks")
        target_directory = directory.resolve()
        if target_directory.exists():
            raise FileExistsError(f"Index target already exists: {target_directory}")
        target_directory.parent.mkdir(parents=True, exist_ok=True)
        directory = target_directory.parent / (
            f".{target_directory.name}.building-{uuid4().hex[:8]}"
        )
        directory.mkdir()

        database_path = directory / "lexical.sqlite3"
        temporary_database = directory / "lexical.sqlite3.part"
        temporary_database.unlink(missing_ok=True)
        lengths: dict[str, int] = {}
        postings: dict[str, dict[str, int]] = defaultdict(dict)

        with closing(sqlite3.connect(temporary_database)) as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    document_length INTEGER NOT NULL,
                    document_key TEXT,
                    company_name TEXT,
                    report_year INTEGER,
                    document_type TEXT,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE terms (
                    term TEXT PRIMARY KEY,
                    document_frequency INTEGER NOT NULL
                );
                CREATE TABLE postings (
                    term TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    term_frequency INTEGER NOT NULL,
                    PRIMARY KEY (term, chunk_id),
                    FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id)
                );
                CREATE INDEX postings_term_idx ON postings(term);
                CREATE INDEX chunks_document_key_idx ON chunks(document_key);
                CREATE INDEX chunks_company_year_idx ON chunks(company_name, report_year);
                CREATE INDEX chunks_document_type_idx ON chunks(document_type);
                """
            )
            for chunk in chunks:
                term_counts = Counter(tokenize_for_search(searchable_chunk_text(chunk)))
                document_length = sum(term_counts.values())
                if document_length == 0:
                    raise ValueError(f"Chunk has no indexable tokens: {chunk.chunk_id}")
                lengths[chunk.chunk_id] = document_length
                connection.execute(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.chunk_index,
                        document_length,
                        chunk.document_key,
                        chunk.company_name,
                        chunk.report_year,
                        chunk.document_type,
                        chunk.model_dump_json(),
                    ),
                )
                for term, frequency in term_counts.items():
                    postings[term][chunk.chunk_id] = frequency

            connection.executemany(
                "INSERT INTO terms VALUES (?, ?)",
                ((term, len(chunk_frequencies)) for term, chunk_frequencies in postings.items()),
            )
            connection.executemany(
                "INSERT INTO postings VALUES (?, ?, ?)",
                (
                    (term, chunk_id, frequency)
                    for term, chunk_frequencies in postings.items()
                    for chunk_id, frequency in chunk_frequencies.items()
                ),
            )
            connection.commit()
        temporary_database.replace(database_path)

        embedding_dimension = None
        reused_embedding_count = 0
        encoded_embedding_count = 0
        if dense_model:
            previous_embeddings = None
            previous_lookup: dict[str, int] = {}
            if (
                reuse_dense_from is not None
                and reuse_dense_from.manifest.dense_model == dense_model
            ):
                previous_embeddings, previous_ids = reuse_dense_from._get_dense_data()
                previous_lookup = {
                    chunk_id: index for index, chunk_id in enumerate(previous_ids)
                }

            missing_indices = [
                index for index, chunk in enumerate(chunks) if chunk.chunk_id not in previous_lookup
            ]
            new_embeddings = None
            if missing_indices:
                model = _load_sentence_transformer(dense_model)
                passages = [
                    _dense_text(searchable_chunk_text(chunks[index]), dense_model, "passage")
                    for index in missing_indices
                ]
                new_embeddings = model.encode(
                    passages,
                    batch_size=16,
                    normalize_embeddings=True,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                ).astype(np.float32)
                embedding_dimension = int(new_embeddings.shape[1])
            elif previous_embeddings is not None:
                embedding_dimension = int(previous_embeddings.shape[1])
            else:
                raise ValueError("Dense indexing produced no embeddings")

            embeddings = np.empty((len(chunks), embedding_dimension), dtype=np.float32)
            new_embedding_index = 0
            for index, chunk in enumerate(chunks):
                previous_index = previous_lookup.get(chunk.chunk_id)
                if previous_index is not None:
                    if previous_embeddings.shape[1] != embedding_dimension:
                        raise ValueError("Previous dense embedding dimension does not match")
                    embeddings[index] = previous_embeddings[previous_index]
                    reused_embedding_count += 1
                else:
                    embeddings[index] = new_embeddings[new_embedding_index]
                    new_embedding_index += 1
                    encoded_embedding_count += 1
            np.save(directory / "dense_embeddings.npy", embeddings)
            (directory / "dense_chunk_ids.json").write_text(
                json.dumps([chunk.chunk_id for chunk in chunks]) + "\n",
                encoding="utf-8",
            )

        source_digest = _sha256_file(source_chunk_path)
        table_records = structured_tables or []
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        table_ids: set[str] = set()
        for table in table_records:
            if table.table_id in table_ids:
                raise ValueError(f"Duplicate structured table ID: {table.table_id}")
            table_ids.add(table.table_id)
            source_chunk = chunk_by_id.get(table.chunk_id)
            if source_chunk is None:
                raise ValueError(
                    f"Structured table references an unknown chunk: {table.chunk_id}"
                )
            expected_chunk_hash = chunk_payload_sha256(source_chunk)
            if table.chunk_sha256 != expected_chunk_hash:
                raise ValueError(
                    "Structured table source chunk hash mismatch: "
                    f"{table.table_id}"
                )
        tables_content = serialize_structured_tables(table_records)
        tables_digest = hashlib.sha256(tables_content.encode()).hexdigest()
        document_ids = list(dict.fromkeys(chunk.document_id for chunk in chunks))
        index_seed = f"{source_digest}:{dense_model or 'lexical'}:{INDEX_FORMAT_VERSION}"
        manifest = IndexManifest(
            index_id=hashlib.sha256(index_seed.encode()).hexdigest()[:20],
            created_at=datetime.now(UTC),
            source_chunk_sha256=source_digest,
            document_ids=document_ids,
            chunk_count=len(chunks),
            average_document_length=sum(lengths.values()) / len(lengths),
            dense_model=dense_model,
            embedding_dimension=embedding_dimension,
            parent_index_id=(reuse_dense_from.manifest.index_id if reuse_dense_from else None),
            reused_embedding_count=reused_embedding_count,
            encoded_embedding_count=encoded_embedding_count,
            structured_table_schema_version=STRUCTURED_TABLE_SCHEMA_VERSION,
            structured_table_generator=STRUCTURED_TABLE_GENERATOR,
            structured_table_count=len(table_records),
            structured_table_cell_count=sum(len(table.cells) for table in table_records),
            structured_tables_sha256=tables_digest,
        )
        tables_path = directory / "structured_tables.jsonl"
        tables_path.write_text(tables_content, encoding="utf-8")
        table_manifest = StructuredTableArtifactManifest(
            index_id=manifest.index_id,
            source_chunk_sha256=source_digest,
            tables_sha256=tables_digest,
            table_count=len(table_records),
            cell_count=sum(len(table.cells) for table in table_records),
        )
        (directory / "structured_tables.manifest.json").write_text(
            table_manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        (directory / "manifest.json").write_text(
            manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        directory.replace(target_directory)
        return cls(target_directory)

    def validate(self) -> None:
        """Fail fast when persisted files disagree with the index manifest."""
        with closing(sqlite3.connect(self.database_path)) as connection:
            database_chunk_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise ValueError(f"SQLite index integrity check failed: {integrity}")
        if database_chunk_count != self.manifest.chunk_count:
            raise ValueError(
                "Manifest and SQLite chunk counts disagree: "
                f"{self.manifest.chunk_count} != {database_chunk_count}"
            )

        dense_path = self.directory / "dense_embeddings.npy"
        dense_ids_path = self.directory / "dense_chunk_ids.json"
        if self.manifest.dense_model:
            if not dense_path.is_file() or not dense_ids_path.is_file():
                raise FileNotFoundError(
                    f"Dense index files are missing: {dense_path}, {dense_ids_path}"
                )
            embeddings = np.load(dense_path, mmap_mode="r")
            chunk_ids = json.loads(dense_ids_path.read_text(encoding="utf-8"))
            expected_shape = (
                self.manifest.chunk_count,
                self.manifest.embedding_dimension,
            )
            if embeddings.shape != expected_shape:
                raise ValueError(
                    f"Dense matrix shape {embeddings.shape} does not match {expected_shape}"
                )
            if len(chunk_ids) != self.manifest.chunk_count or len(set(chunk_ids)) != len(chunk_ids):
                raise ValueError("Dense chunk IDs are missing or duplicated")
        elif dense_path.exists() or dense_ids_path.exists():
            raise ValueError("Dense embeddings exist but the manifest has no dense model")

        tables_path = self.directory / "structured_tables.jsonl"
        tables_manifest_path = self.directory / "structured_tables.manifest.json"
        declared_table_artifact = any(
            value is not None
            for value in (
                self.manifest.structured_table_schema_version,
                self.manifest.structured_table_generator,
                self.manifest.structured_tables_sha256,
            )
        ) or bool(
            self.manifest.structured_table_count
            or self.manifest.structured_table_cell_count
        )
        if tables_path.exists() != tables_manifest_path.exists():
            raise FileNotFoundError(
                "Structured-table data and manifest must either both exist or both be absent"
            )
        if tables_path.is_file():
            artifact = StructuredTableArtifactManifest.model_validate_json(
                tables_manifest_path.read_text(encoding="utf-8")
            )
            if artifact.schema_version != STRUCTURED_TABLE_SCHEMA_VERSION:
                raise ValueError("Unsupported structured-table artifact schema")
            if artifact.generator != STRUCTURED_TABLE_GENERATOR:
                raise ValueError("Unsupported structured-table artifact generator")
            if self.manifest.structured_table_schema_version != artifact.schema_version:
                raise ValueError("Index manifest structured-table schema mismatch")
            if self.manifest.structured_table_generator != artifact.generator:
                raise ValueError("Index manifest structured-table generator mismatch")
            if artifact.index_id != self.manifest.index_id:
                raise ValueError("Structured-table artifact index ID mismatch")
            if artifact.source_chunk_sha256 != self.manifest.source_chunk_sha256:
                raise ValueError("Structured-table artifact source digest mismatch")
            actual_digest = _sha256_file(tables_path)
            if actual_digest != artifact.tables_sha256:
                raise ValueError("Structured-table artifact content digest mismatch")
            tables = load_structured_tables(tables_path.read_text(encoding="utf-8"))
            if len(tables) != artifact.table_count:
                raise ValueError("Structured-table artifact table count mismatch")
            if sum(len(table.cells) for table in tables) != artifact.cell_count:
                raise ValueError("Structured-table artifact cell count mismatch")
            if self.manifest.structured_tables_sha256 != actual_digest:
                raise ValueError("Index manifest structured-table digest mismatch")
            if self.manifest.structured_table_count != artifact.table_count:
                raise ValueError("Index manifest structured-table count mismatch")
            if self.manifest.structured_table_cell_count != artifact.cell_count:
                raise ValueError("Index manifest structured-table cell count mismatch")
            table_ids: set[str] = set()
            with closing(sqlite3.connect(self.database_path)) as connection:
                payloads = dict(
                    connection.execute(
                        "SELECT chunk_id, payload_json FROM chunks"
                    ).fetchall()
                )
            for table in tables:
                if table.table_id in table_ids:
                    raise ValueError(
                        f"Duplicate structured table ID: {table.table_id}"
                    )
                table_ids.add(table.table_id)
                payload = payloads.get(table.chunk_id)
                if payload is None:
                    raise ValueError(
                        "Structured table references an unknown persisted chunk: "
                        f"{table.chunk_id}"
                    )
                persisted_chunk = DocumentChunk.model_validate_json(payload)
                if table.chunk_sha256 != chunk_payload_sha256(persisted_chunk):
                    raise ValueError(
                        "Structured table persisted chunk hash mismatch: "
                        f"{table.table_id}"
                    )
        elif declared_table_artifact:
            raise FileNotFoundError("Index manifest declares a missing structured-table artifact")

    def _load_chunks(self, chunk_ids: list[str]) -> dict[str, DocumentChunk]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        with closing(sqlite3.connect(self.database_path)) as connection:
            rows = connection.execute(
                f"SELECT chunk_id, payload_json FROM chunks WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            ).fetchall()
        scopes = self._statement_scopes()
        tables = self._structured_tables()
        return {
            chunk_id: DocumentChunk.model_validate_json(payload).model_copy(
                update={
                    "statement_scope": scopes.get(chunk_id, "unspecified"),
                    "structured_tables": tables.get(chunk_id, []),
                }
            )
            for chunk_id, payload in rows
        }

    def _structured_tables(self) -> dict[str, list[StructuredTable]]:
        if self._structured_table_cache is not None:
            return self._structured_table_cache
        path = self.directory / "structured_tables.jsonl"
        by_chunk: dict[str, list[StructuredTable]] = defaultdict(list)
        if path.is_file():
            for table in load_structured_tables(path.read_text(encoding="utf-8")):
                by_chunk[table.chunk_id].append(table)
        self._structured_table_cache = dict(by_chunk)
        return self._structured_table_cache

    def _statement_scopes(self) -> dict[str, str]:
        """Propagate explicit filing section boundaries across chunk payloads."""
        if self._statement_scope_cache is not None:
            return self._statement_scope_cache
        with closing(sqlite3.connect(self.database_path)) as connection:
            rows = connection.execute(
                "SELECT chunk_id, document_id, chunk_index, payload_json "
                "FROM chunks ORDER BY document_id, chunk_index"
            ).fetchall()
        scopes: dict[str, str] = {}
        current_document = ""
        scope = "unspecified"
        for chunk_id, document_id, _chunk_index, payload in rows:
            if document_id != current_document:
                current_document = document_id
                scope = "unspecified"
            chunk = DocumentChunk.model_validate_json(payload)
            context = re.sub(r"\s+", "", "".join(chunk.section_path) + chunk.text[:800])
            parent_positions = [
                position
                for marker in (
                    "母公司资产负债表",
                    "母公司利润表",
                    "母公司现金流量表",
                    "母公司所有者权益变动表",
                    "母公司财务报表主要项目注释",
                )
                if (position := context.rfind(marker)) >= 0
            ]
            consolidated_positions = [
                position
                for marker in (
                    "合并资产负债表",
                    "合并利润表",
                    "合并现金流量表",
                    "合并所有者权益变动表",
                    "合并财务报表项目注释",
                )
                if (position := context.rfind(marker)) >= 0
            ]
            parent_position = max(parent_positions, default=-1)
            consolidated_position = max(consolidated_positions, default=-1)
            if parent_position >= 0 or consolidated_position >= 0:
                scope = (
                    "parent"
                    if parent_position > consolidated_position
                    else "consolidated"
                )
            scopes[chunk_id] = scope
        self._statement_scope_cache = scopes
        return scopes

    def resolve_chunks(self, chunk_ids: list[str]) -> list[DocumentChunk | None]:
        """Resolve exact evidence IDs while preserving caller order and duplicates."""
        chunks = self._load_chunks(chunk_ids)
        return [chunks.get(chunk_id) for chunk_id in chunk_ids]

    def _matching_chunk_ids(self, filters: SearchFilters | None) -> set[str] | None:
        if filters is None or not filters.active:
            return None
        clauses: list[str] = []
        parameters: list[str | int] = []
        for column, values in (
            ("document_key", filters.document_keys),
            ("company_name", filters.company_names),
            ("report_year", filters.report_years),
            ("document_type", filters.document_types),
        ):
            if values:
                clauses.append(f"{column} IN ({','.join('?' for _ in values)})")
                parameters.extend(values)
        with closing(sqlite3.connect(self.database_path)) as connection:
            rows = connection.execute(
                f"SELECT chunk_id FROM chunks WHERE {' AND '.join(clauses)}", parameters
            ).fetchall()
        return {str(row[0]) for row in rows}

    def search_lexical(
        self, query: str, top_k: int = 10, filters: SearchFilters | None = None
    ) -> list[SearchHit]:
        query_terms = Counter(tokenize_for_search(query))
        if not query_terms:
            return []
        terms = list(query_terms)
        placeholders = ",".join("?" for _ in terms)
        scores: dict[str, float] = defaultdict(float)
        allowed_ids = self._matching_chunk_ids(filters)
        if allowed_ids == set():
            return []
        with closing(sqlite3.connect(self.database_path)) as connection:
            term_rows = connection.execute(
                f"SELECT term, document_frequency FROM terms WHERE term IN ({placeholders})",
                terms,
            ).fetchall()
            frequencies = dict(term_rows)
            posting_rows = connection.execute(
                f"""
                SELECT p.term, p.chunk_id, p.term_frequency, c.document_length
                FROM postings p JOIN chunks c ON c.chunk_id = p.chunk_id
                WHERE p.term IN ({placeholders})
                """,
                terms,
            ).fetchall()

        total_documents = self.manifest.chunk_count
        average_length = self.manifest.average_document_length
        k1 = self.manifest.bm25_k1
        b = self.manifest.bm25_b
        for term, chunk_id, term_frequency, document_length in posting_rows:
            document_frequency = frequencies[term]
            inverse_document_frequency = math.log(
                1 + (total_documents - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            denominator = term_frequency + k1 * (
                1 - b + b * document_length / average_length
            )
            scores[chunk_id] += (
                query_terms[term]
                * inverse_document_frequency
                * term_frequency
                * (k1 + 1)
                / denominator
            )
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        if allowed_ids is not None:
            ranked = sorted(
                ((chunk_id, score) for chunk_id, score in scores.items() if chunk_id in allowed_ids),
                key=lambda item: (-item[1], item[0]),
            )[:top_k]
        chunks = self._load_chunks([chunk_id for chunk_id, _ in ranked])
        return [
            SearchHit(
                rank=rank,
                chunk=chunks[chunk_id],
                score=score,
                lexical_rank=rank,
                lexical_score=score,
            )
            for rank, (chunk_id, score) in enumerate(ranked, start=1)
        ]

    def search_dense(
        self, query: str, top_k: int = 10, filters: SearchFilters | None = None
    ) -> list[SearchHit]:
        if not self.manifest.dense_model:
            raise RuntimeError("This index was built without dense embeddings")
        embeddings, chunk_ids = self._get_dense_data()
        model = self._get_dense_model()
        query_text = _dense_text(query, self.manifest.dense_model, "query")
        with self._dense_model_lock:
            query_embedding = model.encode(
                [query_text], normalize_embeddings=True, convert_to_numpy=True
            )[0].astype(np.float32)
        scores = embeddings @ query_embedding
        allowed_ids = self._matching_chunk_ids(filters)
        indices = [
            int(index)
            for index in np.argsort(-scores, kind="stable")
            if allowed_ids is None or str(chunk_ids[index]) in allowed_ids
        ][: min(top_k, len(chunk_ids))]
        ranked_ids = [str(chunk_ids[index]) for index in indices]
        chunks = self._load_chunks(ranked_ids)
        return [
            SearchHit(
                rank=rank,
                chunk=chunks[chunk_ids[index]],
                score=float(scores[index]),
                dense_rank=rank,
                dense_score=float(scores[index]),
            )
            for rank, index in enumerate(indices, start=1)
        ]

    def search_dense_batch(
        self,
        queries: list[str],
        top_k: int = 10,
        filters: list[SearchFilters | None] | None = None,
    ) -> list[list[SearchHit]]:
        """Encode a query batch once for efficient offline evaluation."""
        if not self.manifest.dense_model:
            raise RuntimeError("This index was built without dense embeddings")
        if filters is not None and len(filters) != len(queries):
            raise ValueError("filters must have the same length as queries")
        embeddings, chunk_ids = self._get_dense_data()
        model = self._get_dense_model()
        query_texts = [_dense_text(query, self.manifest.dense_model, "query") for query in queries]
        with self._dense_model_lock:
            query_embeddings = model.encode(
                query_texts, normalize_embeddings=True, convert_to_numpy=True
            ).astype(np.float32)
        output: list[list[SearchHit]] = []
        for index, query_embedding in enumerate(query_embeddings):
            allowed_ids = self._matching_chunk_ids(filters[index] if filters else None)
            scores = embeddings @ query_embedding
            selected = [
                int(position)
                for position in np.argsort(-scores, kind="stable")
                if allowed_ids is None or str(chunk_ids[position]) in allowed_ids
            ][: min(top_k, len(chunk_ids))]
            chunks = self._load_chunks([str(chunk_ids[position]) for position in selected])
            output.append(
                [
                    SearchHit(
                        rank=rank,
                        chunk=chunks[str(chunk_ids[position])],
                        score=float(scores[position]),
                        dense_rank=rank,
                        dense_score=float(scores[position]),
                    )
                    for rank, position in enumerate(selected, start=1)
                ]
            )
        return output

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: Literal["lexical", "dense", "hybrid"] = "lexical",
        candidate_k: int = 50,
        rrf_k: int = 60,
        filters: SearchFilters | None = None,
    ) -> list[SearchHit]:
        if top_k < 1 or candidate_k < top_k:
            raise ValueError("Expected candidate_k >= top_k >= 1")
        if mode == "lexical":
            return self.search_lexical(query, top_k, filters)
        if mode == "dense":
            return self.search_dense(query, top_k, filters)

        lexical = self.search_lexical(query, candidate_k, filters)
        dense = self.search_dense(query, candidate_k, filters)
        return reciprocal_rank_fusion(lexical, dense, top_k=top_k, rrf_k=rrf_k)

    def _get_dense_model(self):
        if not self.manifest.dense_model:
            raise RuntimeError("This index was built without dense embeddings")
        with self._dense_model_lock:
            if self._dense_model is None:
                self._dense_model = _load_sentence_transformer(self.manifest.dense_model)
            return self._dense_model

    def _get_dense_data(self):
        with self._dense_model_lock:
            if self._dense_embeddings is None or self._dense_chunk_ids is None:
                self._dense_embeddings = np.load(
                    self.directory / "dense_embeddings.npy", mmap_mode="r"
                )
                self._dense_chunk_ids = json.loads(
                    (self.directory / "dense_chunk_ids.json").read_text(encoding="utf-8")
                )
            return self._dense_embeddings, self._dense_chunk_ids


def reciprocal_rank_fusion(
    lexical: list[SearchHit],
    dense: list[SearchHit],
    top_k: int,
    rrf_k: int = 60,
    lexical_weight: float = 2.0,
    dense_weight: float = 1.0,
) -> list[SearchHit]:
    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")
    if lexical_weight <= 0 or dense_weight <= 0:
        raise ValueError("RRF weights must be positive")
    by_id: dict[str, dict] = {}
    for hit in lexical:
        by_id.setdefault(hit.chunk.chunk_id, {"chunk": hit.chunk, "score": 0.0})
        by_id[hit.chunk.chunk_id].update(
            lexical_rank=hit.rank,
            lexical_score=hit.score,
        )
        by_id[hit.chunk.chunk_id]["score"] += lexical_weight / (rrf_k + hit.rank)
    for hit in dense:
        by_id.setdefault(hit.chunk.chunk_id, {"chunk": hit.chunk, "score": 0.0})
        by_id[hit.chunk.chunk_id].update(dense_rank=hit.rank, dense_score=hit.score)
        by_id[hit.chunk.chunk_id]["score"] += dense_weight / (rrf_k + hit.rank)

    ranked = sorted(
        by_id.values(),
        key=lambda item: (
            -item["score"],
            item["chunk"].chunk_id,
        ),
    )[:top_k]
    return [SearchHit(rank=rank, **item) for rank, item in enumerate(ranked, start=1)]
