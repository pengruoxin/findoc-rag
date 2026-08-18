"""Persistent upload/ingestion jobs for Agent-controlled corpus mutation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from findoc_rag.io import write_text_lf

UploadStatus = Literal[
    "uploaded", "validating", "ingesting", "indexing", "ready", "failed"
]


class UploadJob(BaseModel):
    job_id: str
    filename: str
    status: UploadStatus
    bytes_written: int = Field(default=0, ge=0)
    message: str = ""
    document_key: str | None = None
    document_version_id: str | None = None
    index_id: str | None = None
    created_at: datetime
    updated_at: datetime
    error_code: str | None = None


class StartUploadProcessingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    document_key: str = Field(min_length=1, max_length=200)
    metadata: dict | None = None


class UploadJobStore:
    """File-backed store so process restarts do not erase Agent job state."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def job_directory(self, job_id: str) -> Path:
        if not job_id.isalnum():
            raise ValueError("Invalid upload job ID")
        directory = (self.root / job_id).resolve()
        if not directory.is_relative_to(self.root):
            raise ValueError("Upload job path escapes its root")
        return directory

    def create(self, *, job_id: str, filename: str, bytes_written: int) -> UploadJob:
        now = datetime.now(UTC)
        job = UploadJob(
            job_id=job_id,
            filename=filename,
            status="uploaded",
            bytes_written=bytes_written,
            message="Upload received; processing requires an explicit start request",
            created_at=now,
            updated_at=now,
        )
        self.write(job)
        return job

    def get(self, job_id: str) -> UploadJob | None:
        path = self.job_directory(job_id) / "job.json"
        if not path.is_file():
            return None
        return UploadJob.model_validate_json(path.read_text(encoding="utf-8"))

    def claim(
        self, job_id: str, request: StartUploadProcessingRequest
    ) -> UploadJob | None:
        """Atomically move an uploaded job into the processing state."""
        with self._lock:
            job = self.get(job_id)
            if job is None:
                return None
            if job.status != "uploaded":
                raise RuntimeError(job.status)
            updated = job.model_copy(
                update={
                    "status": "validating",
                    "document_key": request.document_key,
                    "message": "Validating PDF and quality gates",
                    "updated_at": datetime.now(UTC),
                }
            )
            self.write_request(updated, request)
            self.write(updated)
            return updated

    def fail_interrupted_jobs(self) -> int:
        """Make a restart visible instead of leaving jobs indefinitely in flight."""
        failed = 0
        for path in self.root.glob("*/job.json"):
            job = UploadJob.model_validate_json(path.read_text(encoding="utf-8"))
            if job.status not in {"validating", "ingesting", "indexing"}:
                continue
            self.update(
                job,
                status="failed",
                error_code="process_restarted",
                message="Processing was interrupted by a service restart; upload again to retry",
            )
            failed += 1
        return failed

    def write(self, job: UploadJob) -> None:
        directory = self.job_directory(job.job_id)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "job.json"
        temporary = directory / "job.json.part"
        with self._lock:
            write_text_lf(temporary, job.model_dump_json(indent=2) + "\n")
            temporary.replace(target)

    def update(self, job: UploadJob, **changes: object) -> UploadJob:
        updated = job.model_copy(
            update={**changes, "updated_at": datetime.now(UTC)}
        )
        self.write(updated)
        return updated

    def source_path(self, job: UploadJob) -> Path:
        path = self.job_directory(job.job_id) / job.filename
        if not path.is_file():
            raise FileNotFoundError(f"Uploaded PDF is missing: {path}")
        return path

    def write_request(
        self, job: UploadJob, request: StartUploadProcessingRequest
    ) -> None:
        path = self.job_directory(job.job_id) / "processing-request.json"
        temporary = path.with_suffix(".json.part")
        write_text_lf(
            temporary,
            json.dumps(request.model_dump(mode="json"), ensure_ascii=False, indent=2)
            + "\n",
        )
        temporary.replace(path)
