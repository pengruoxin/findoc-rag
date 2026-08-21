"""Strict provisional probes for genuine scanned financial tables.

These probes are intentionally smaller than a full cell graph.  They provide a
repeatable development baseline for text/value recovery and same-row
association while the complete annotation is awaiting independent review.
"""

from __future__ import annotations

import importlib.metadata
import re
import time
import unicodedata
from pathlib import Path
from statistics import fmean
from typing import Literal

import pymupdf
from pydantic import BaseModel, Field, model_validator

from findoc_rag.documents.geometry import element_display_bbox, page_display_dimensions
from findoc_rag.documents.models import DocumentElement, DocumentPage
from findoc_rag.documents.pdf import PdfExtractionConfig, parse_pdf
from findoc_rag.pdf_complex_benchmark import SHA256_PATTERN
from findoc_rag.pdf_evaluation import file_sha256
from findoc_rag.pdf_table_interpretation import TableQuestion, score_table_fact

NUMERIC_TOKEN = re.compile(r"[-−—－]?\s*\d[\d,，]*(?:[.．]\d+)?")


class ScanSemanticProbe(BaseModel):
    probe_id: str
    probe_type: Literal["text", "row_value"]
    category: Literal["title", "header", "cell"]
    expected_text: str | None = None
    row_label: str | None = None
    column_label: str | None = None
    expected_value: str | None = None
    annotation_note: str

    @model_validator(mode="after")
    def validate_probe(self) -> ScanSemanticProbe:
        if self.probe_type == "text":
            if (
                not self.expected_text
                or self.row_label
                or self.column_label
                or self.expected_value
            ):
                raise ValueError("Text probes require only expected_text")
        elif (
            not self.row_label
            or not self.column_label
            or not self.expected_value
            or self.expected_text
        ):
            raise ValueError(
                "Row-value probes require row_label, column_label and expected_value"
            )
        return self


class ScanProbePage(BaseModel):
    candidate_id: str
    page_number: int = Field(ge=1)
    expected_route: Literal["full_ocr"] = "full_ocr"
    probes: list[ScanSemanticProbe] = Field(min_length=1)


class ScanProbeBenchmark(BaseModel):
    schema_version: Literal["1"] = "1"
    dataset_id: str
    status: Literal["assistant_curated_provisional"]
    candidate_manifest: str
    candidate_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    pdf_path: str
    pdf_sha256: str = Field(pattern=SHA256_PATTERN)
    counts_toward_formal_target: Literal[False] = False
    independent_review_required: Literal[True] = True
    pages: list[ScanProbePage] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_benchmark(self) -> ScanProbeBenchmark:
        page_numbers = [page.page_number for page in self.pages]
        if page_numbers != sorted(page_numbers) or len(set(page_numbers)) != len(page_numbers):
            raise ValueError("Probe page numbers must be unique and ascending")
        probe_ids = [probe.probe_id for page in self.pages for probe in page.probes]
        if len(set(probe_ids)) != len(probe_ids):
            raise ValueError("Scan probe IDs must be unique")
        return self


def _compact_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return "".join(normalized.split())


def _numeric_tokens(value: str) -> set[str]:
    tokens: set[str] = set()
    for match in NUMERIC_TOKEN.finditer(unicodedata.normalize("NFKC", value)):
        token = match.group(0).replace(",", "").replace("，", "")
        token = token.replace("．", ".").replace("−", "-").replace("—", "-")
        token = token.replace("－", "-")
        token = "".join(token.split())
        if token:
            tokens.add(token)
    return tokens


def _text_elements(page: DocumentPage) -> list[DocumentElement]:
    return [
        element
        for element in page.elements
        if element.element_type == "text" and element.text.strip()
    ]


def _line_clusters(
    page: DocumentPage,
    *,
    y_tolerance_ratio: float = 0.009,
    use_display_geometry: bool = True,
) -> list[str]:
    display_boxes = {
        element.element_id: (
            element_display_bbox(page, element.bbox)
            if use_display_geometry
            else element.bbox
        )
        for element in _text_elements(page)
    }
    elements = sorted(
        _text_elements(page),
        key=lambda element: (
            (
                display_boxes[element.element_id].y0
                + display_boxes[element.element_id].y1
            )
            / 2,
            display_boxes[element.element_id].x0,
        ),
    )
    display_width, display_height = (
        page_display_dimensions(page)
        if use_display_geometry
        else (page.width, page.height)
    )
    tolerance = max(2.0, max(display_width, display_height) * y_tolerance_ratio)
    clusters: list[dict] = []
    for element in elements:
        display_box = display_boxes[element.element_id]
        center = (display_box.y0 + display_box.y1) / 2
        compatible = [
            cluster
            for cluster in clusters
            if abs(center - cluster["center_sum"] / cluster["count"]) <= tolerance
        ]
        if compatible:
            cluster = min(
                compatible,
                key=lambda item: abs(center - item["center_sum"] / item["count"]),
            )
            cluster["elements"].append(element)
            cluster["center_sum"] += center
            cluster["count"] += 1
        else:
            clusters.append(
                {"elements": [element], "center_sum": center, "count": 1}
            )
    lines: list[str] = []
    for cluster in clusters:
        ordered = sorted(
            cluster["elements"],
            key=lambda element: display_boxes[element.element_id].x0,
        )
        lines.append(" ".join(element.text for element in ordered))
    return lines


def evaluate_page_probes(
    page: DocumentPage,
    expected: ScanProbePage,
    *,
    use_display_geometry: bool = True,
    allow_hierarchical_headers: bool = True,
    allow_wrapped_row_labels: bool = True,
) -> dict:
    elements = _text_elements(page)
    page_text = "\n".join(element.text for element in elements)
    compact_page = _compact_text(page_text)
    page_values = set().union(*(_numeric_tokens(element.text) for element in elements))
    lines = _line_clusters(page, use_display_geometry=use_display_geometry)
    predictions: list[dict] = []
    for probe in expected.probes:
        if probe.probe_type == "text":
            success = _compact_text(probe.expected_text or "") in compact_page
            predictions.append(
                {
                    "probe_id": probe.probe_id,
                    "probe_type": probe.probe_type,
                    "category": probe.category,
                    "expected_text": probe.expected_text,
                    "success": success,
                }
            )
            continue
        compact_label = _compact_text(probe.row_label or "")
        expected_value = next(iter(_numeric_tokens(probe.expected_value or "")), "")
        label_found = compact_label in compact_page
        value_found = expected_value in page_values
        matching_lines = [
            line
            for line in lines
            if compact_label in _compact_text(line)
            and expected_value in _numeric_tokens(line)
        ]
        structure = score_table_fact(
            page,
            TableQuestion(
                question_id=probe.probe_id,
                question=(
                    f"{probe.row_label}的{probe.column_label}是多少？"
                ),
                expected_value=probe.expected_value or "",
                row_label=probe.row_label or "",
                column_label=probe.column_label or "",
            ),
            allow_hierarchical_headers=allow_hierarchical_headers,
            allow_wrapped_row_labels=allow_wrapped_row_labels,
        )
        predictions.append(
            {
                "probe_id": probe.probe_id,
                "probe_type": probe.probe_type,
                "category": probe.category,
                "row_label": probe.row_label,
                "column_label": probe.column_label,
                "expected_value": probe.expected_value,
                "label_found": label_found,
                "value_found": value_found,
                "same_row_association": bool(matching_lines),
                "column_header_found": structure["column_header_found"],
                "column_aligned": structure["column_aligned"],
                "structured_cell_recoverable": structure["recoverable"],
                "matching_lines": matching_lines,
                "success": bool(matching_lines),
            }
        )
    coordinate_elements = [
        element for element in elements if element.extraction_source == "ocr"
    ]
    bounds_width, bounds_height = (
        (page.width, page.height)
        if use_display_geometry or page.rotation % 180 == 0
        else (page.height, page.width)
    )
    coordinate_in_bounds = [
        element
        for element in coordinate_elements
        if element.bbox.x0 >= 0
        and element.bbox.y0 >= 0
        and element.bbox.x1 <= bounds_width
        and element.bbox.y1 <= bounds_height
    ]
    return {
        "candidate_id": expected.candidate_id,
        "page_number": expected.page_number,
        "expected_route": expected.expected_route,
        "actual_route": page.extraction_route,
        "route_correct": page.extraction_route == expected.expected_route,
        "native_character_count": page.native_character_count,
        "extracted_character_count": page.extracted_character_count,
        "ocr_character_count": page.ocr_character_count,
        "ocr_attempted": page.ocr_attempted,
        "ocr_succeeded": page.ocr_succeeded,
        "ocr_backend": page.ocr_backend,
        "rotation": page.rotation,
        "width": page.width,
        "height": page.height,
        "ocr_element_count": len(coordinate_elements),
        "ocr_coordinate_in_bounds_count": len(coordinate_in_bounds),
        "ocr_coordinate_bounds_rate": (
            len(coordinate_in_bounds) / len(coordinate_elements)
            if coordinate_elements
            else 0.0
        ),
        "line_cluster_count": len(lines),
        "predictions": predictions,
        "extracted_text": page_text,
    }


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarize_scan_lane(page_reports: list[dict], elapsed_ms: float) -> dict:
    predictions = [
        prediction
        for page in page_reports
        for prediction in page["predictions"]
    ]
    text_predictions = [
        prediction for prediction in predictions if prediction["probe_type"] == "text"
    ]
    row_predictions = [
        prediction for prediction in predictions if prediction["probe_type"] == "row_value"
    ]
    ocr_elements = sum(page["ocr_element_count"] for page in page_reports)
    in_bounds = sum(page["ocr_coordinate_in_bounds_count"] for page in page_reports)
    return {
        "page_count": len(page_reports),
        "probe_count": len(predictions),
        "route_accuracy": _ratio(
            sum(page["route_correct"] for page in page_reports), len(page_reports)
        ),
        "ocr_attempt_rate": _ratio(
            sum(page["ocr_attempted"] for page in page_reports), len(page_reports)
        ),
        "ocr_success_rate": _ratio(
            sum(page["ocr_succeeded"] for page in page_reports), len(page_reports)
        ),
        "native_character_count": sum(
            page["native_character_count"] or 0 for page in page_reports
        ),
        "ocr_character_count": sum(page["ocr_character_count"] for page in page_reports),
        "text_probe_exact_recall": _ratio(
            sum(prediction["success"] for prediction in text_predictions),
            len(text_predictions),
        ),
        "row_label_exact_recall": _ratio(
            sum(prediction["label_found"] for prediction in row_predictions),
            len(row_predictions),
        ),
        "row_value_exact_recall": _ratio(
            sum(prediction["value_found"] for prediction in row_predictions),
            len(row_predictions),
        ),
        "same_row_association_recall": _ratio(
            sum(prediction["same_row_association"] for prediction in row_predictions),
            len(row_predictions),
        ),
        "column_header_exact_recall": _ratio(
            sum(prediction["column_header_found"] for prediction in row_predictions),
            len(row_predictions),
        ),
        "column_alignment_recall": _ratio(
            sum(prediction["column_aligned"] for prediction in row_predictions),
            len(row_predictions),
        ),
        "structured_cell_recall": _ratio(
            sum(
                prediction["structured_cell_recoverable"]
                for prediction in row_predictions
            ),
            len(row_predictions),
        ),
        "strict_probe_recall": _ratio(
            sum(prediction["success"] for prediction in predictions), len(predictions)
        ),
        "ocr_coordinate_bounds_rate": _ratio(in_bounds, ocr_elements),
        "mean_page_coordinate_bounds_rate": (
            fmean(page["ocr_coordinate_bounds_rate"] for page in page_reports)
            if page_reports
            else 0.0
        ),
        "elapsed_ms": elapsed_ms,
    }


def _backend_version(name: str) -> str:
    if name == "native":
        return pymupdf.VersionBind
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def evaluate_scan_lane(
    pdf_path: Path,
    benchmark: ScanProbeBenchmark,
    *,
    lane: Literal["native", "hybrid"],
    ocr_backend: str = "rapidocr",
    ocr_dpi: int = 180,
    use_display_geometry: bool = True,
    allow_hierarchical_headers: bool = True,
    allow_wrapped_row_labels: bool = True,
) -> dict:
    config = PdfExtractionConfig(
        mode="disabled" if lane == "native" else "auto",
        ocr_backend=ocr_backend,
        ocr_dpi=ocr_dpi,
    )
    started = time.perf_counter()
    document = parse_pdf(pdf_path, config)
    elapsed_ms = (time.perf_counter() - started) * 1000
    expected_by_page = {page.page_number: page for page in benchmark.pages}
    page_reports = [
        evaluate_page_probes(
            page,
            expected_by_page[page.page_number],
            use_display_geometry=use_display_geometry,
            allow_hierarchical_headers=allow_hierarchical_headers,
            allow_wrapped_row_labels=allow_wrapped_row_labels,
        )
        for page in document.pages
        if page.page_number in expected_by_page
    ]
    if len(page_reports) != len(benchmark.pages):
        raise ValueError("Parsed PDF pages do not match the scan probe benchmark")
    return {
        "lane": lane,
        "backend": "pymupdf-native" if lane == "native" else ocr_backend,
        "backend_version": _backend_version("native" if lane == "native" else ocr_backend),
        "ocr_dpi": None if lane == "native" else ocr_dpi,
        "metrics": summarize_scan_lane(page_reports, elapsed_ms),
        "pages": page_reports,
    }


def load_scan_probe_benchmark(
    benchmark_path: Path, workspace: Path | None = None
) -> tuple[ScanProbeBenchmark, Path]:
    effective_workspace = (workspace or Path.cwd()).resolve(strict=True)
    benchmark = ScanProbeBenchmark.model_validate_json(
        benchmark_path.read_text(encoding="utf-8")
    )
    manifest_path = (effective_workspace / benchmark.candidate_manifest).resolve(strict=True)
    pdf_path = (effective_workspace / benchmark.pdf_path).resolve(strict=True)
    if file_sha256(manifest_path) != benchmark.candidate_manifest_sha256:
        raise ValueError("Genuine scan candidate manifest SHA-256 mismatch")
    if file_sha256(pdf_path) != benchmark.pdf_sha256:
        raise ValueError("Genuine scan candidate PDF SHA-256 mismatch")
    return benchmark, pdf_path


def compare_scan_lanes(native: dict, hybrid: dict) -> dict:
    metric_names = (
        "text_probe_exact_recall",
        "row_label_exact_recall",
        "row_value_exact_recall",
        "same_row_association_recall",
        "column_header_exact_recall",
        "column_alignment_recall",
        "structured_cell_recall",
        "strict_probe_recall",
        "ocr_coordinate_bounds_rate",
    )
    return {
        f"{name}_delta": hybrid["metrics"][name] - native["metrics"][name]
        for name in metric_names
    } | {
        "elapsed_ms_delta": hybrid["metrics"]["elapsed_ms"]
        - native["metrics"]["elapsed_ms"]
    }
