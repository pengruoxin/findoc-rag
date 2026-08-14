import hashlib
import json
import re
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

DOCUMENT_KEY_PATTERN = re.compile(r"^[\w.:-]{1,200}$", re.UNICODE)
VersionStatus = Literal["staging", "active", "superseded", "failed", "deleted"]
LEGACY_PROCESSING_FINGERPRINT = "legacy"


class DocumentVersion(BaseModel):
    version_id: str
    content_version_id: str
    document_key: str
    content_sha256: str
    processing_fingerprint: str
    source_path: str
    status: VersionStatus
    created_at: datetime
    activated_at: datetime | None = None
    document_ir_path: str | None = None
    chunks_path: str | None = None
    chunk_count: int | None = Field(default=None, ge=0)
    error_message: str | None = None
    metadata: dict = Field(default_factory=dict)


class RegisteredDocument(BaseModel):
    document_key: str
    current_version_id: str | None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None
    metadata: dict = Field(default_factory=dict)


class RegistrationDecision(BaseModel):
    action: Literal["new", "resume", "reuse", "unchanged"]
    version: DocumentVersion


def validate_document_key(document_key: str) -> str:
    normalized = document_key.strip()
    if not DOCUMENT_KEY_PATTERN.fullmatch(normalized):
        raise ValueError(
            "document_key must contain only Unicode word characters, '.', ':', or '-' "
            "and be at most 200 characters"
        )
    return normalized


def derive_content_version_id(document_key: str, content_sha256: str) -> str:
    """Derive the stable identity of source content, independent of processing."""
    seed = json.dumps(
        {"document_key": document_key, "content_sha256": content_sha256},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(seed).hexdigest()[:24]


def derive_version_id(content_version_id: str, processing_fingerprint: str) -> str:
    """Derive an artifact revision from content identity and processing identity."""
    seed = json.dumps(
        {
            "content_version_id": content_version_id,
            "processing_fingerprint": processing_fingerprint,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(seed).hexdigest()[:24]


class DocumentRegistry:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path.resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_key TEXT PRIMARY KEY,
                    current_version_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted_at TEXT,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(document_versions)").fetchall()
            }
            if columns and not {"content_version_id", "processing_fingerprint"} <= columns:
                self._migrate_legacy_versions(connection)
            self._create_versions_schema(connection)
            connection.commit()

    @staticmethod
    def _create_versions_schema(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS document_versions (
                version_id TEXT PRIMARY KEY,
                content_version_id TEXT NOT NULL,
                document_key TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                processing_fingerprint TEXT NOT NULL,
                source_path TEXT NOT NULL,
                status TEXT NOT NULL CHECK (
                    status IN ('staging','active','superseded','failed','deleted')
                ),
                created_at TEXT NOT NULL,
                activated_at TEXT,
                document_ir_path TEXT,
                chunks_path TEXT,
                chunk_count INTEGER,
                error_message TEXT,
                metadata_json TEXT NOT NULL,
                UNIQUE(document_key, content_sha256, processing_fingerprint),
                FOREIGN KEY(document_key) REFERENCES documents(document_key)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS version_document_status_idx
            ON document_versions(document_key, status)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS version_content_idx
            ON document_versions(content_sha256)
            """
        )

    @staticmethod
    def _migrate_legacy_versions(connection: sqlite3.Connection) -> None:
        """Upgrade the original content-only uniqueness without losing revisions."""
        connection.execute("ALTER TABLE document_versions RENAME TO document_versions_legacy")
        # SQLite keeps index names when a table is renamed, so release the names
        # before creating the replacement indexes.
        connection.execute("DROP INDEX IF EXISTS version_document_status_idx")
        connection.execute("DROP INDEX IF EXISTS version_content_idx")
        DocumentRegistry._create_versions_schema(connection)
        rows = connection.execute("SELECT * FROM document_versions_legacy").fetchall()
        for row in rows:
            content_version_id = derive_content_version_id(
                row["document_key"], row["content_sha256"]
            )
            connection.execute(
                """
                INSERT INTO document_versions (
                    version_id, content_version_id, document_key, content_sha256,
                    processing_fingerprint, source_path, status, created_at,
                    activated_at, document_ir_path, chunks_path, chunk_count,
                    error_message, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["version_id"],
                    content_version_id,
                    row["document_key"],
                    row["content_sha256"],
                    LEGACY_PROCESSING_FINGERPRINT,
                    row["source_path"],
                    row["status"],
                    row["created_at"],
                    row["activated_at"],
                    row["document_ir_path"],
                    row["chunks_path"],
                    row["chunk_count"],
                    row["error_message"],
                    row["metadata_json"],
                ),
            )
        connection.execute("DROP TABLE document_versions_legacy")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _version_from_row(row: sqlite3.Row) -> DocumentVersion:
        return DocumentVersion(
            version_id=row["version_id"],
            content_version_id=row["content_version_id"],
            document_key=row["document_key"],
            content_sha256=row["content_sha256"],
            processing_fingerprint=row["processing_fingerprint"],
            source_path=row["source_path"],
            status=row["status"],
            created_at=datetime.fromisoformat(row["created_at"]),
            activated_at=(
                datetime.fromisoformat(row["activated_at"]) if row["activated_at"] else None
            ),
            document_ir_path=row["document_ir_path"],
            chunks_path=row["chunks_path"],
            chunk_count=row["chunk_count"],
            error_message=row["error_message"],
            metadata=json.loads(row["metadata_json"]),
        )

    def begin_ingestion(
        self,
        document_key: str,
        content_sha256: str,
        source_path: Path,
        metadata: dict | None = None,
        processing_fingerprint: str = LEGACY_PROCESSING_FINGERPRINT,
    ) -> RegistrationDecision:
        document_key = validate_document_key(document_key)
        if not processing_fingerprint.strip():
            raise ValueError("processing_fingerprint must not be empty")
        now = datetime.now(UTC).isoformat()
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
        content_version_id = derive_content_version_id(document_key, content_sha256)
        version_id = derive_version_id(content_version_id, processing_fingerprint)

        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            document = connection.execute(
                "SELECT * FROM documents WHERE document_key = ?", (document_key,)
            ).fetchone()
            if document is None:
                connection.execute(
                    "INSERT INTO documents VALUES (?, NULL, ?, ?, NULL, ?)",
                    (document_key, now, now, metadata_json),
                )
            else:
                connection.execute(
                    """
                    UPDATE documents SET updated_at = ?, deleted_at = NULL
                    WHERE document_key = ?
                    """,
                    (now, document_key),
                )

            existing = connection.execute(
                """
                SELECT * FROM document_versions
                WHERE document_key = ? AND content_sha256 = ?
                    AND processing_fingerprint = ?
                """,
                (document_key, content_sha256, processing_fingerprint),
            ).fetchone()
            current_version_id = document["current_version_id"] if document else None
            if existing:
                version = self._version_from_row(existing)
                if current_version_id == version.version_id and version.status == "active":
                    # Metadata describes the logical document as currently exposed.
                    # On the unchanged path both copies must move atomically.
                    connection.execute(
                        "UPDATE documents SET metadata_json = ? WHERE document_key = ?",
                        (metadata_json, document_key),
                    )
                    connection.execute(
                        "UPDATE document_versions SET metadata_json = ? WHERE version_id = ?",
                        (metadata_json, version.version_id),
                    )
                    action = "unchanged"
                elif version.status == "staging":
                    connection.execute(
                        "UPDATE document_versions SET metadata_json = ? WHERE version_id = ?",
                        (metadata_json, version.version_id),
                    )
                    action = "resume"
                elif version.document_ir_path and version.chunks_path:
                    connection.execute(
                        "UPDATE document_versions SET metadata_json = ? WHERE version_id = ?",
                        (metadata_json, version.version_id),
                    )
                    action = "reuse"
                else:
                    connection.execute(
                        """
                        UPDATE document_versions
                        SET status = 'staging', source_path = ?, error_message = NULL,
                            metadata_json = ?
                        WHERE version_id = ?
                        """,
                        (source_path.resolve().as_posix(), metadata_json, version.version_id),
                    )
                    action = "resume"
                    existing = connection.execute(
                        "SELECT * FROM document_versions WHERE version_id = ?", (version.version_id,)
                    ).fetchone()
                    version = self._version_from_row(existing)
                existing = connection.execute(
                    "SELECT * FROM document_versions WHERE version_id = ?", (version.version_id,)
                ).fetchone()
                version = self._version_from_row(existing)
                connection.commit()
                return RegistrationDecision(action=action, version=version)

            connection.execute(
                """
                INSERT INTO document_versions (
                    version_id, content_version_id, document_key, content_sha256,
                    processing_fingerprint, source_path, status,
                    created_at, activated_at, document_ir_path, chunks_path,
                    chunk_count, error_message, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, 'staging', ?, NULL, NULL, NULL, NULL, NULL, ?)
                """,
                (
                    version_id,
                    content_version_id,
                    document_key,
                    content_sha256,
                    processing_fingerprint,
                    source_path.resolve().as_posix(),
                    now,
                    metadata_json,
                ),
            )
            row = connection.execute(
                "SELECT * FROM document_versions WHERE version_id = ?", (version_id,)
            ).fetchone()
            connection.commit()
        return RegistrationDecision(action="new", version=self._version_from_row(row))

    def activate(
        self,
        version_id: str,
        document_ir_path: Path,
        chunks_path: Path,
        chunk_count: int,
    ) -> DocumentVersion:
        if not document_ir_path.is_file() or not chunks_path.is_file():
            raise FileNotFoundError("Document IR and chunk artifacts must exist before activation")
        now = datetime.now(UTC).isoformat()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            version = connection.execute(
                "SELECT * FROM document_versions WHERE version_id = ?", (version_id,)
            ).fetchone()
            if version is None:
                raise KeyError(f"Unknown version: {version_id}")
            connection.execute(
                """
                UPDATE document_versions SET status = 'superseded'
                WHERE document_key = ? AND status = 'active' AND version_id != ?
                """,
                (version["document_key"], version_id),
            )
            connection.execute(
                """
                UPDATE document_versions
                SET status = 'active', activated_at = ?, document_ir_path = ?,
                    chunks_path = ?, chunk_count = ?, error_message = NULL
                WHERE version_id = ?
                """,
                (
                    now,
                    document_ir_path.resolve().as_posix(),
                    chunks_path.resolve().as_posix(),
                    chunk_count,
                    version_id,
                ),
            )
            connection.execute(
                """
                UPDATE documents
                SET current_version_id = ?, updated_at = ?, deleted_at = NULL,
                    metadata_json = ?
                WHERE document_key = ?
                """,
                (version_id, now, version["metadata_json"], version["document_key"]),
            )
            row = connection.execute(
                "SELECT * FROM document_versions WHERE version_id = ?", (version_id,)
            ).fetchone()
            connection.commit()
        return self._version_from_row(row)

    def mark_failed(self, version_id: str, message: str) -> None:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                """
                SELECT d.current_version_id FROM document_versions v
                JOIN documents d ON d.document_key = v.document_key
                WHERE v.version_id = ?
                """,
                (version_id,),
            ).fetchone()
            if current is None:
                raise KeyError(f"Unknown version: {version_id}")
            if current["current_version_id"] == version_id:
                raise ValueError("Cannot mark the active version as failed")
            connection.execute(
                """
                UPDATE document_versions SET status = 'failed', error_message = ?
                WHERE version_id = ?
                """,
                (message[:2000], version_id),
            )
            connection.commit()

    def soft_delete(self, document_key: str) -> None:
        document_key = validate_document_key(document_key)
        now = datetime.now(UTC).isoformat()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            document = connection.execute(
                "SELECT current_version_id FROM documents WHERE document_key = ?", (document_key,)
            ).fetchone()
            if document is None:
                raise KeyError(f"Unknown document: {document_key}")
            if document["current_version_id"]:
                connection.execute(
                    "UPDATE document_versions SET status = 'deleted' WHERE version_id = ?",
                    (document["current_version_id"],),
                )
            connection.execute(
                """
                UPDATE documents SET current_version_id = NULL, deleted_at = ?, updated_at = ?
                WHERE document_key = ?
                """,
                (now, now, document_key),
            )
            connection.commit()

    def update_metadata(self, document_key: str, metadata: dict) -> DocumentVersion:
        document_key = validate_document_key(document_key)
        now = datetime.now(UTC).isoformat()
        metadata_json = json.dumps(metadata, ensure_ascii=False)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            document = connection.execute(
                "SELECT current_version_id FROM documents WHERE document_key = ?",
                (document_key,),
            ).fetchone()
            if document is None or not document["current_version_id"]:
                raise KeyError(f"Document has no active version: {document_key}")
            connection.execute(
                "UPDATE documents SET metadata_json = ?, updated_at = ? WHERE document_key = ?",
                (metadata_json, now, document_key),
            )
            connection.execute(
                "UPDATE document_versions SET metadata_json = ? WHERE version_id = ?",
                (metadata_json, document["current_version_id"]),
            )
            row = connection.execute(
                "SELECT * FROM document_versions WHERE version_id = ?",
                (document["current_version_id"],),
            ).fetchone()
            connection.commit()
        return self._version_from_row(row)

    def active_versions(self) -> list[DocumentVersion]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT v.* FROM document_versions v
                JOIN documents d ON d.current_version_id = v.version_id
                WHERE v.status = 'active' AND d.deleted_at IS NULL
                ORDER BY v.document_key
                """
            ).fetchall()
        return [self._version_from_row(row) for row in rows]

    def versions(self, document_key: str) -> list[DocumentVersion]:
        document_key = validate_document_key(document_key)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM document_versions WHERE document_key = ? ORDER BY created_at
                """,
                (document_key,),
            ).fetchall()
        return [self._version_from_row(row) for row in rows]
