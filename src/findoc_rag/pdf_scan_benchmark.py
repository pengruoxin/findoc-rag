"""Provenance and eligibility checks for genuine scanned-PDF candidates.

The filename or announcement title is not treated as evidence that a PDF is a
genuine scan.  A selected page must be image dominant and have effectively no
usable native text.  This keeps OCR evaluation separate from PDFs that merely
contain a full-page image alongside a usable hidden text layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import pymupdf
from pydantic import BaseModel, Field, model_validator

from findoc_rag.pdf_complex_benchmark import (
    SHA256_PATTERN,
    PdfBenchmarkSplit,
    PdfChallengeStratum,
)
from findoc_rag.pdf_evaluation import file_sha256


class GenuineScanThresholds(BaseModel):
    max_native_characters: int = Field(default=5, ge=0)
    min_image_coverage: float = Field(default=0.85, ge=0, le=1)
    text_layer_rejection_min_characters: int = Field(default=200, ge=1)


class GenuineScanPageSelection(BaseModel):
    candidate_id: str
    source_page_number: int = Field(ge=1)
    assigned_split: PdfBenchmarkSplit
    primary_stratum: Literal["genuine_scan"] = "genuine_scan"
    additional_strata: list[PdfChallengeStratum] = Field(default_factory=list)
    table_title: str
    visual_review_notes: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_selection(self) -> GenuineScanPageSelection:
        if self.assigned_split not in {"calibration", "development"}:
            raise ValueError("Genuine scan candidates must use calibration or development")
        if self.primary_stratum in self.additional_strata:
            raise ValueError("Primary stratum must not be repeated in additional_strata")
        if len(set(self.additional_strata)) != len(self.additional_strata):
            raise ValueError("Additional strata must be unique")
        return self


class GenuineScanSource(BaseModel):
    source_id: str
    security_code: str
    company_name: str
    announcement_id: str
    announcement_title: str
    official_url: str
    local_file: str
    expected_sha256: str = Field(pattern=SHA256_PATTERN)
    expected_size_bytes: int = Field(gt=0)
    disposition: Literal["accepted_genuine_scan", "rejected_usable_text_layer"]
    disposition_reason: str
    selections: list[GenuineScanPageSelection] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_source(self) -> GenuineScanSource:
        parsed = urlparse(self.official_url)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "static.cninfo.com.cn"
            or not parsed.path.lower().endswith(".pdf")
        ):
            raise ValueError("Official scan sources must be HTTPS PDFs on static.cninfo.com.cn")
        if self.disposition == "accepted_genuine_scan" and not self.selections:
            raise ValueError("Accepted genuine scans must select at least one table page")
        if self.disposition != "accepted_genuine_scan" and self.selections:
            raise ValueError("Rejected sources cannot contribute selected pages")
        return self


class GenuineScanRegistry(BaseModel):
    schema_version: Literal["1"] = "1"
    dataset_id: str
    review_status: Literal["assistant_visual_review_provisional"]
    thresholds: GenuineScanThresholds = Field(default_factory=GenuineScanThresholds)
    sources: list[GenuineScanSource] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_registry(self) -> GenuineScanRegistry:
        source_ids = [source.source_id for source in self.sources]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("Genuine scan source IDs must be unique")
        candidate_ids = [
            selection.candidate_id
            for source in self.sources
            for selection in source.selections
        ]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("Genuine scan candidate IDs must be unique")
        return self


def _workspace_path(value: str, workspace: Path) -> Path:
    path = Path(value)
    resolved = (workspace / path).resolve(strict=True) if not path.is_absolute() else path.resolve(strict=True)
    if not resolved.is_relative_to(workspace):
        raise ValueError(f"Source path escapes the workspace: {value}")
    return resolved


def _image_profile(page: pymupdf.Page, raw: dict) -> tuple[int, float, float]:
    page_area = page.rect.width * page.rect.height
    ratios: list[float] = []
    for block in raw.get("blocks", []):
        if block.get("type") != 1 or page_area <= 0:
            continue
        x0, y0, x1, y1 = block["bbox"]
        ratios.append(max(0.0, x1 - x0) * max(0.0, y1 - y0) / page_area)
    return len(ratios), min(1.0, sum(ratios)), max(ratios, default=0.0)


def profile_scan_page(page: pymupdf.Page, page_number: int) -> dict:
    raw = page.get_text("dict", sort=True)
    native_text = page.get_text("text", sort=True)
    native_character_count = len("".join(native_text.split()))
    image_count, image_coverage_sum, image_coverage_max = _image_profile(page, raw)
    return {
        "page_number": page_number,
        "native_character_count": native_character_count,
        "image_count": image_count,
        "image_coverage_sum": round(image_coverage_sum, 6),
        "image_coverage_max": round(image_coverage_max, 6),
        "width": float(page.rect.width),
        "height": float(page.rect.height),
        "rotation": int(page.rotation),
    }


def audit_genuine_scan_registry(registry_path: Path, workspace: Path | None = None) -> dict:
    effective_workspace = (workspace or Path.cwd()).resolve(strict=True)
    registry = GenuineScanRegistry.model_validate_json(
        registry_path.read_text(encoding="utf-8")
    )
    source_reports: list[dict] = []
    all_valid = True
    for source in registry.sources:
        source_path = _workspace_path(source.local_file, effective_workspace)
        actual_sha256 = file_sha256(source_path)
        actual_size = source_path.stat().st_size
        selected_page_numbers = {
            selection.source_page_number for selection in source.selections
        }
        selected_profiles: list[dict] = []
        minimum_native_characters: int | None = None
        maximum_native_characters = 0
        with pymupdf.open(source_path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                profile = profile_scan_page(page, page_number)
                minimum_native_characters = (
                    profile["native_character_count"]
                    if minimum_native_characters is None
                    else min(minimum_native_characters, profile["native_character_count"])
                )
                maximum_native_characters = max(
                    maximum_native_characters, profile["native_character_count"]
                )
                if page_number in selected_page_numbers:
                    selected_profiles.append(profile)
            page_count = pdf.page_count
        pages_exist = selected_page_numbers <= set(range(1, page_count + 1))
        selected_pages_eligible = bool(selected_profiles) and all(
            profile["native_character_count"]
            <= registry.thresholds.max_native_characters
            and profile["image_coverage_max"]
            >= registry.thresholds.min_image_coverage
            for profile in selected_profiles
        )
        if source.disposition == "accepted_genuine_scan":
            disposition_valid = pages_exist and selected_pages_eligible
        else:
            disposition_valid = (
                not selected_page_numbers
                and minimum_native_characters is not None
                and minimum_native_characters
                >= registry.thresholds.text_layer_rejection_min_characters
            )
        source_valid = (
            actual_sha256 == source.expected_sha256
            and actual_size == source.expected_size_bytes
            and disposition_valid
        )
        all_valid = all_valid and source_valid
        source_reports.append(
            {
                "source_id": source.source_id,
                "official_url": source.official_url,
                "local_file": source.local_file,
                "expected_sha256": source.expected_sha256,
                "actual_sha256": actual_sha256,
                "expected_size_bytes": source.expected_size_bytes,
                "actual_size_bytes": actual_size,
                "page_count": page_count,
                "minimum_native_character_count": minimum_native_characters or 0,
                "maximum_native_character_count": maximum_native_characters,
                "disposition": source.disposition,
                "disposition_valid": disposition_valid,
                "selected_pages_eligible": selected_pages_eligible,
                "selected_page_profiles": selected_profiles,
                "source_valid": source_valid,
            }
        )
    return {
        "schema_version": "1",
        "dataset_id": registry.dataset_id,
        "review_status": registry.review_status,
        "source_registry": registry_path.as_posix(),
        "source_registry_sha256": file_sha256(registry_path),
        "thresholds": registry.thresholds.model_dump(mode="json"),
        "all_sources_valid": all_valid,
        "accepted_source_count": sum(
            source.disposition == "accepted_genuine_scan" for source in registry.sources
        ),
        "rejected_text_layer_source_count": sum(
            source.disposition == "rejected_usable_text_layer"
            for source in registry.sources
        ),
        "selected_page_count": sum(len(source.selections) for source in registry.sources),
        "sources": source_reports,
    }


def build_genuine_scan_candidate_pdf(
    registry_path: Path,
    output_pdf: Path,
    *,
    workspace: Path | None = None,
) -> tuple[dict, dict]:
    effective_workspace = (workspace or Path.cwd()).resolve(strict=True)
    audit = audit_genuine_scan_registry(registry_path, effective_workspace)
    if not audit["all_sources_valid"]:
        raise ValueError("Genuine scan source audit failed")
    registry = GenuineScanRegistry.model_validate_json(
        registry_path.read_text(encoding="utf-8")
    )
    resolved_output = output_pdf.resolve()
    if resolved_output.suffix.lower() != ".pdf" or not resolved_output.is_relative_to(
        effective_workspace
    ):
        raise ValueError("Candidate PDF must be a .pdf inside the workspace")
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved_output.with_suffix(".part.pdf")
    temporary.unlink(missing_ok=True)
    output = pymupdf.open()
    pages: list[dict] = []
    try:
        for source in registry.sources:
            if source.disposition != "accepted_genuine_scan":
                continue
            source_path = _workspace_path(source.local_file, effective_workspace)
            with pymupdf.open(source_path) as source_pdf:
                for selection in source.selections:
                    source_index = selection.source_page_number - 1
                    output.insert_pdf(source_pdf, from_page=source_index, to_page=source_index)
                    source_profile = profile_scan_page(
                        source_pdf[source_index], selection.source_page_number
                    )
                    pages.append(
                        {
                            "candidate_id": selection.candidate_id,
                            "challenge_page_number": len(pages) + 1,
                            "document_key": source.source_id,
                            "security_code": source.security_code,
                            "company_name": source.company_name,
                            "announcement_id": source.announcement_id,
                            "announcement_title": source.announcement_title,
                            "official_url": source.official_url,
                            "source_file": source.local_file,
                            "source_sha256": source.expected_sha256,
                            "source_page_number": selection.source_page_number,
                            "assigned_split": selection.assigned_split,
                            "primary_stratum": selection.primary_stratum,
                            "additional_strata": selection.additional_strata,
                            "table_title": selection.table_title,
                            "visual_review_notes": selection.visual_review_notes,
                            "source_page_profile": source_profile,
                            "annotation_status": "unannotated",
                            "counts_toward_target": False,
                        }
                    )
        output.save(temporary, garbage=4, deflate=True)
    finally:
        output.close()
    temporary.replace(resolved_output)
    with pymupdf.open(resolved_output) as check:
        if check.page_count != len(pages):
            raise ValueError("Built scan candidate PDF page count does not match manifest")
    manifest = {
        "schema_version": "1",
        "dataset_id": "pdf-hard-v2-genuine-scan-development-candidates",
        "status": "assistant_visual_reviewed_unannotated",
        "source_registry": registry_path.as_posix(),
        "source_registry_sha256": file_sha256(registry_path),
        "pdf_path": output_pdf.as_posix(),
        "pdf_sha256": file_sha256(resolved_output),
        "page_count": len(pages),
        "source_audit_passed": True,
        "counts_toward_target": False,
        "frozen_gold_present": False,
        "pages": pages,
    }
    return manifest, audit
