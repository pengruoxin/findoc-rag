from datetime import UTC, datetime
from pathlib import Path

import pytest

from findoc_rag.upload_jobs import (
    StartUploadProcessingRequest,
    UploadJob,
    UploadJobStore,
)


def test_claim_is_single_start_and_persists_request(tmp_path: Path) -> None:
    store = UploadJobStore(tmp_path)
    store.create(job_id="abc123", filename="report.pdf", bytes_written=100)
    request = StartUploadProcessingRequest(
        document_key="company:annual:2024", metadata={"report_year": 2024}
    )

    claimed = store.claim("abc123", request)

    assert claimed is not None
    assert claimed.status == "validating"
    assert (tmp_path / "abc123" / "processing-request.json").is_file()
    with pytest.raises(RuntimeError, match="validating"):
        store.claim("abc123", request)


def test_restart_marks_in_flight_jobs_failed(tmp_path: Path) -> None:
    store = UploadJobStore(tmp_path)
    now = datetime.now(UTC)
    store.write(
        UploadJob(
            job_id="abc123",
            filename="report.pdf",
            status="indexing",
            bytes_written=100,
            created_at=now,
            updated_at=now,
        )
    )

    count = store.fail_interrupted_jobs()
    failed = store.get("abc123")

    assert count == 1
    assert failed is not None
    assert failed.status == "failed"
    assert failed.error_code == "process_restarted"


def test_job_id_cannot_escape_upload_root(tmp_path: Path) -> None:
    store = UploadJobStore(tmp_path)

    with pytest.raises(ValueError, match="Invalid upload job ID"):
        store.get("../outside")
