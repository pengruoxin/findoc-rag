"""Acquire, verify, ingest, and index a sealed benchmark corpus plan."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from findoc_rag.corpus import build_active_corpus_index
from findoc_rag.evaluation.governance import (
    BenchmarkCorpusPlan,
    GovernancePolicy,
    PlannedDocument,
    audit_corpus_plan,
)
from findoc_rag.ingestion import file_sha256, ingest_pdf
from findoc_rag.registry import DocumentRegistry
from findoc_rag.sources.cninfo import (
    CninfoClient,
    FilingArtifact,
    select_chinese_annual_report,
    write_artifact_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("data/evaluation/benchmark-v3-corpus-plan.json"),
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("configs/evaluation-governance-p0.json"),
    )
    parser.add_argument(
        "--artifact-dir", type=Path, default=Path("data/artifacts/cninfo")
    )
    parser.add_argument(
        "--registry", type=Path, default=Path("data/catalog/benchmark-v3/registry.sqlite3")
    )
    parser.add_argument(
        "--storage-dir", type=Path, default=Path("data/catalog/benchmark-v3/versions")
    )
    parser.add_argument(
        "--index-root", type=Path, default=Path("data/indexes/benchmark-v3")
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/evaluation/benchmark-v3-source-manifest.json"),
    )
    parser.add_argument(
        "--corpus-manifest",
        type=Path,
        default=Path("data/evaluation/benchmark-v3-version-manifest.json"),
    )
    parser.add_argument("--skip-ingest", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _stable_created_at(path: Path) -> str:
    if path.is_file():
        current = json.loads(path.read_text(encoding="utf-8"))
        existing = current.get("created_at") or current.get("generated_at")
        if existing:
            return str(existing)
    return datetime.now(UTC).isoformat()


def _acquire_document(
    client: CninfoClient,
    document: PlannedDocument,
    artifact_dir: Path,
) -> FilingArtifact:
    announcements = client.search_annual_reports(
        document.company_name, document.report_year
    )
    selected = select_chinese_annual_report(
        announcements, document.company_name, document.report_year
    )
    if selected.security_code != document.security_code:
        raise ValueError(
            f"Security mismatch for {document.document_key}: "
            f"expected {document.security_code}, got {selected.security_code}"
        )

    stem = f"{selected.security_code}_{document.report_year}_{selected.announcement_id}"
    pdf_path = artifact_dir / f"{stem}.pdf"
    artifact_manifest_path = artifact_dir / f"{stem}.manifest.json"
    if pdf_path.is_file() and artifact_manifest_path.is_file():
        artifact = FilingArtifact.model_validate_json(
            artifact_manifest_path.read_text(encoding="utf-8")
        )
        if artifact.announcement.announcement_id != selected.announcement_id:
            raise ValueError(f"Stale artifact manifest for {document.document_key}")
        actual_hash = file_sha256(pdf_path)
        if actual_hash != artifact.sha256 or pdf_path.stat().st_size != artifact.size_bytes:
            raise ValueError(f"Local source verification failed for {document.document_key}")
        return artifact

    artifact = client.download(selected, pdf_path)
    write_artifact_manifest(artifact, artifact_manifest_path)
    return artifact


def _source_record(document: PlannedDocument, artifact: FilingArtifact) -> dict:
    announcement = artifact.announcement
    return {
        **document.model_dump(mode="json"),
        "announcement_id": announcement.announcement_id,
        "announcement_title": announcement.title,
        "security_name": announcement.security_name,
        "published_at": announcement.published_at.isoformat(),
        "source_url": str(artifact.source_url),
        "local_file": artifact.local_file,
        "sha256": artifact.sha256,
        "size_bytes": artifact.size_bytes,
        "downloaded_at": artifact.downloaded_at.isoformat(),
    }


def main() -> None:
    args = parse_args()
    plan = BenchmarkCorpusPlan.model_validate_json(args.plan.read_text(encoding="utf-8"))
    policy_payload = json.loads(args.policy.read_text(encoding="utf-8"))
    policy_payload["split_strategy"] = plan.split_strategy
    policy = GovernancePolicy.model_validate(policy_payload)
    audit = audit_corpus_plan(plan, policy)
    if not audit.ready_for_acquisition:
        raise SystemExit(f"Corpus plan failed governance audit: {audit.blockers}")

    client = CninfoClient()
    source_records: list[dict] = []
    artifacts: dict[str, FilingArtifact] = {}
    for position, document in enumerate(plan.documents, start=1):
        artifact = _acquire_document(client, document, args.artifact_dir)
        artifacts[document.document_key] = artifact
        source_records.append(_source_record(document, artifact))
        print(
            f"[{position}/{len(plan.documents)}] verified {document.document_key} "
            f"sha256={artifact.sha256[:12]}"
        )

    _write_json(
        args.source_manifest,
        {
            "schema_version": "1",
            "plan_id": plan.plan_id,
            "created_at": _stable_created_at(args.source_manifest),
            "source": "cninfo",
            "documents": source_records,
        },
    )
    if args.skip_ingest:
        return

    registry = DocumentRegistry(args.registry)
    version_records: list[dict] = []
    for position, document in enumerate(plan.documents, start=1):
        artifact = artifacts[document.document_key]
        result = ingest_pdf(
            Path(artifact.local_file),
            document.document_key,
            registry,
            args.storage_dir,
            metadata={
                "security_code": document.security_code,
                "company_name": document.company_name,
                "report_year": document.report_year,
                "document_type": "annual",
                "benchmark_split": document.split,
                "source": document.source,
                "source_announcement_id": artifact.announcement.announcement_id,
                "source_url": str(artifact.source_url),
                "source_sha256": artifact.sha256,
            },
        )
        version = result.version
        version_records.append(
            {
                "document_key": document.document_key,
                "split": document.split,
                "version_id": version.version_id,
                "content_version_id": version.content_version_id,
                "content_sha256": version.content_sha256,
                "processing_fingerprint": version.processing_fingerprint,
                "chunk_count": version.chunk_count,
                "status": version.status,
            }
        )
        print(
            f"[{position}/{len(plan.documents)}] {result.action} {document.document_key} "
            f"chunks={version.chunk_count}"
        )

    index_results = {
        "calibration": build_active_corpus_index(
            registry,
            args.index_root / "calibration",
            benchmark_splits={"calibration"},
        ),
        "development": build_active_corpus_index(
            registry,
            args.index_root / "development",
            benchmark_splits={"calibration", "dev"},
        ),
        "frozen_test": build_active_corpus_index(
            registry,
            args.index_root / "frozen_test",
            benchmark_splits={"frozen_test"},
        ),
    }
    _write_json(
        args.corpus_manifest,
        {
            "schema_version": "1",
            "plan_id": plan.plan_id,
            "created_at": _stable_created_at(args.corpus_manifest),
            "registry_document_count": len(version_records),
            "documents": version_records,
            "indexes": {
                name: {
                    "index_id": result.pointer.index_id,
                    "active_version_ids": result.pointer.active_version_ids,
                    "chunk_count": result.manifest["chunk_count"],
                    "dense_model": result.manifest["dense_model"],
                }
                for name, result in index_results.items()
            },
        },
    )
    for name, result in index_results.items():
        print(
            f"{name} index {result.action}: {result.pointer.index_id} "
            f"documents={len(result.pointer.active_version_ids)}"
        )


if __name__ == "__main__":
    main()
