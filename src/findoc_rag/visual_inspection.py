"""Bounded PDF geometry inspection for diagram-heavy evidence pages."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

import fitz
from pydantic import BaseModel, ConfigDict, Field, model_validator

from findoc_rag.documents.models import BoundingBox
from findoc_rag.documents.ocr import OcrBackend, create_ocr_backend
from findoc_rag.table_cell_proof import TableCellGeometryProof

PERCENTAGE_PATTERN = re.compile(r"^-?\d+(?:\.\d+)?%$")


class VisualTextNode(BaseModel):
    node_id: str
    text: str
    bbox: BoundingBox


class VisualRelationshipRow(BaseModel):
    label: VisualTextNode
    value: VisualTextNode
    vertical_center_delta: float = Field(ge=0)
    connector_present: bool


class PageRegionInspection(BaseModel):
    document_key: str
    page_number: int = Field(ge=1)
    page_width: float = Field(gt=0)
    page_height: float = Field(gt=0)
    source_sha256: str
    relationship_rows: list[VisualRelationshipRow]
    drawing_count: int = Field(ge=0)


class ReconstructedTextColumn(BaseModel):
    """One visually isolated page column reconstructed from the rendered PDF."""

    label: str
    text: str
    average_confidence: float = Field(ge=0, le=1)


class PageLayoutReconstruction(BaseModel):
    """Manifest-bound, rendered-page text kept in explicit column order."""

    document_key: str
    page_number: int = Field(ge=1)
    source_sha256: str
    extraction_source: str = "rapidocr-rendered-two-column"
    columns: list[ReconstructedTextColumn]

    @property
    def evidence_text(self) -> str:
        sections = [
            f"[PDF第{self.page_number}页双栏重建：{column.label}]\n{column.text}"
            for column in self.columns
            if column.text
        ]
        return "\n\n".join(sections)

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.evidence_text.encode("utf-8")).hexdigest()

    @property
    def evidence_chunk_id(self) -> str:
        return f"layout:{self.source_sha256[:12]}:p{self.page_number}:{self.content_sha256[:16]}"


class TableCellRegionProof(BaseModel):
    """A bounded rendered crop bound to one coordinate table-cell proof."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    document_key: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    table_cell_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    page_number: int = Field(ge=1)
    page_width: float = Field(gt=0)
    page_height: float = Field(gt=0)
    value_bbox: BoundingBox
    region_bbox: BoundingBox
    coordinate_space: Literal["pymupdf_unrotated_page"] = "pymupdf_unrotated_page"
    native_value_match_count: int = Field(ge=1)
    rendered_dpi: int = Field(ge=72, le=300)
    rendered_width: int = Field(gt=0)
    rendered_height: int = Field(gt=0)
    rendered_area_ratio: float = Field(gt=0, le=0.2)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_path: str
    binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_binding(self) -> TableCellRegionProof:
        if self.binding_sha256 != table_cell_region_binding_sha256(self):
            raise ValueError("Table-cell region proof binding SHA-256 mismatch")
        return self


def _region_binding_payload(proof: TableCellRegionProof) -> dict:
    return proof.model_dump(mode="json", exclude={"binding_sha256"})


def table_cell_region_binding_sha256(proof: TableCellRegionProof) -> str:
    canonical = json.dumps(
        _region_binding_payload(proof),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _bbox_union(boxes: list[tuple[float, float, float, float]]) -> BoundingBox:
    return BoundingBox(
        x0=min(box[0] for box in boxes),
        y0=min(box[1] for box in boxes),
        x1=max(box[2] for box in boxes),
        y1=max(box[3] for box in boxes),
    )


def _center_y(box: BoundingBox) -> float:
    return (box.y0 + box.y1) / 2


def _normalized_pdf_value(value: str) -> str:
    return re.sub(r"[\s,%]", "", value).replace("+", "").replace("−", "-")


def _rect_intersects_bbox(rect: fitz.Rect, bbox: BoundingBox) -> bool:
    return not (rect.x1 < bbox.x0 or rect.x0 > bbox.x1 or rect.y1 < bbox.y0 or rect.y0 > bbox.y1)


class PdfRegionInspector:
    """Inspect a manifest-bound PDF page without exposing arbitrary file reads."""

    def __init__(
        self,
        source_manifest: Path,
        *,
        workspace: Path,
        ocr_backend: OcrBackend | None = None,
        ocr_dpi: int = 180,
    ) -> None:
        self.workspace = workspace.resolve()
        manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
        self.documents = {item["document_key"]: item for item in manifest.get("documents", [])}
        self._ocr_backend = ocr_backend
        self.ocr_dpi = ocr_dpi

    def _source_path(self, document_key: str) -> tuple[Path, str]:
        document = self.documents.get(document_key)
        if document is None:
            raise KeyError(f"Unknown source document: {document_key}")
        path = (self.workspace / document["local_file"]).resolve()
        if not path.is_relative_to(self.workspace):
            raise ValueError("Source PDF resolves outside the configured workspace")
        if not path.is_file():
            raise FileNotFoundError(f"Source PDF is missing: {path}")
        digest = _sha256_file(path)
        if digest != document["sha256"]:
            raise ValueError(f"Source PDF SHA-256 mismatch: {document_key}")
        return path, digest

    def inspect_relationship_rows(
        self,
        document_key: str,
        page_number: int,
    ) -> PageRegionInspection:
        """Pair left labels with percentage nodes using PDF coordinates and connectors."""

        path, digest = self._source_path(document_key)
        with fitz.open(path) as document:
            if not 1 <= page_number <= document.page_count:
                raise ValueError(f"Page {page_number} is outside 1..{document.page_count}")
            page = document[page_number - 1]
            words = page.get_text("words")
            drawings = page.get_drawings()

            by_line: dict[tuple[int, int], list[tuple]] = {}
            for word in words:
                by_line.setdefault((int(word[5]), int(word[6])), []).append(word)
            line_nodes: list[VisualTextNode] = []
            for index, line_words in enumerate(by_line.values()):
                ordered = sorted(line_words, key=lambda item: (item[0], item[1]))
                text = "".join(str(item[4]).strip() for item in ordered)
                if not text:
                    continue
                line_nodes.append(
                    VisualTextNode(
                        node_id=f"line-{index}",
                        text=text,
                        bbox=_bbox_union(
                            [
                                (
                                    float(item[0]),
                                    float(item[1]),
                                    float(item[2]),
                                    float(item[3]),
                                )
                                for item in ordered
                            ]
                        ),
                    )
                )

            percentages = [node for node in line_nodes if PERCENTAGE_PATTERN.fullmatch(node.text)]
            if not percentages:
                return PageRegionInspection(
                    document_key=document_key,
                    page_number=page_number,
                    page_width=float(page.rect.width),
                    page_height=float(page.rect.height),
                    source_sha256=digest,
                    relationship_rows=[],
                    drawing_count=len(drawings),
                )
            first_percentage_x = min(node.bbox.x0 for node in percentages)
            label_lines = sorted(
                (
                    node
                    for node in line_nodes
                    if node.bbox.x1 < first_percentage_x - 2
                    and 0 <= node.bbox.y0 <= 500
                    and not PERCENTAGE_PATTERN.fullmatch(node.text)
                ),
                key=lambda node: (node.bbox.y0, node.bbox.x0),
            )
            label_groups: list[list[VisualTextNode]] = []
            for node in label_lines:
                if (
                    label_groups
                    and node.bbox.y0 - max(item.bbox.y1 for item in label_groups[-1]) <= 5
                ):
                    label_groups[-1].append(node)
                else:
                    label_groups.append([node])
            labels = [
                VisualTextNode(
                    node_id=f"label-{index}",
                    text="".join(node.text for node in group),
                    bbox=_bbox_union(
                        [(node.bbox.x0, node.bbox.y0, node.bbox.x1, node.bbox.y1) for node in group]
                    ),
                )
                for index, group in enumerate(label_groups)
            ]

            rows: list[VisualRelationshipRow] = []
            for percentage in percentages:
                candidates = [
                    label
                    for label in labels
                    if label.bbox.x1 < percentage.bbox.x0
                    and abs(_center_y(label.bbox) - _center_y(percentage.bbox)) <= 20
                ]
                if not candidates:
                    continue
                label = min(
                    candidates,
                    key=lambda item: abs(_center_y(item.bbox) - _center_y(percentage.bbox)),
                )
                percentage_center = _center_y(percentage.bbox)
                connector_present = any(
                    float(drawing["rect"].x0) >= label.bbox.x1 - 2
                    and float(drawing["rect"].x1) <= percentage.bbox.x0 + 2
                    and float(drawing["rect"].y0) - 1
                    <= percentage_center
                    <= float(drawing["rect"].y1) + 1
                    and float(drawing["rect"].width) >= 2
                    and float(drawing["rect"].height) <= 3
                    for drawing in drawings
                )
                rows.append(
                    VisualRelationshipRow(
                        label=label,
                        value=percentage,
                        vertical_center_delta=abs(_center_y(label.bbox) - percentage_center),
                        connector_present=connector_present,
                    )
                )
            return PageRegionInspection(
                document_key=document_key,
                page_number=page_number,
                page_width=float(page.rect.width),
                page_height=float(page.rect.height),
                source_sha256=digest,
                relationship_rows=rows,
                drawing_count=len(drawings),
            )

    def render_table_cell_region(
        self,
        document_key: str,
        proof: TableCellGeometryProof,
        *,
        output_directory: Path,
        context_above: float = 96.0,
        rendered_dpi: int = 180,
    ) -> TableCellRegionProof:
        """Render a small, hash-bound row/header context for one proven cell.

        The method refuses text-only cells, verifies that the claimed value
        still intersects matching native PDF text, and caps the crop at 20%
        of the page. It never renders another page or the full document.
        """

        if (
            proof.geometry_status != "coordinate"
            or proof.page_number is None
            or proof.value_bbox is None
            or proof.coordinate_space != "pymupdf_unrotated_page"
        ):
            raise ValueError("Region rendering requires a coordinate table-cell proof")
        if rendered_dpi < 72 or rendered_dpi > 300:
            raise ValueError("rendered_dpi must be between 72 and 300")
        path, source_digest = self._source_path(document_key)
        destination = output_directory.resolve()
        if not destination.is_relative_to(self.workspace):
            raise ValueError("Region output resolves outside the configured workspace")
        with fitz.open(path) as document:
            if not 1 <= proof.page_number <= document.page_count:
                raise ValueError(f"Page {proof.page_number} is outside 1..{document.page_count}")
            page = document[proof.page_number - 1]
            value_rect = fitz.Rect(
                proof.value_bbox.x0,
                proof.value_bbox.y0,
                proof.value_bbox.x1,
                proof.value_bbox.y1,
            )
            if value_rect.is_empty or not page.rect.contains(value_rect):
                raise ValueError("Table-cell bbox falls outside the source PDF page")
            expected_value = _normalized_pdf_value(proof.value)
            native_matches = [
                word
                for word in page.get_text("words")
                if _rect_intersects_bbox(fitz.Rect(*word[:4]), proof.value_bbox)
                and _normalized_pdf_value(str(word[4])) == expected_value
            ]
            if not native_matches:
                raise ValueError("Table-cell bbox does not intersect matching PDF text")

            page_width = float(page.rect.width)
            page_height = float(page.rect.height)
            clip = fitz.Rect(
                max(0.0, page_width * 0.04),
                max(0.0, value_rect.y0 - context_above),
                min(page_width, value_rect.x1 + 16.0),
                min(page_height, value_rect.y1 + 24.0),
            )
            area_ratio = float(clip.width * clip.height / (page_width * page_height))
            if area_ratio <= 0 or area_ratio > 0.2:
                raise ValueError(f"Bounded region exceeds page-area contract: {area_ratio:.4f}")
            pixmap = page.get_pixmap(
                clip=clip,
                dpi=rendered_dpi,
                alpha=False,
            )
            image_bytes = pixmap.tobytes("png")

        image_digest = hashlib.sha256(image_bytes).hexdigest()
        destination.mkdir(parents=True, exist_ok=True)
        image_path = destination / (
            f"{source_digest[:12]}-p{proof.page_number}-{proof.binding_sha256[:16]}.png"
        )
        if image_path.is_file():
            if hashlib.sha256(image_path.read_bytes()).hexdigest() != image_digest:
                raise ValueError("Existing region image does not match rendered bytes")
        else:
            with image_path.open("xb") as stream:
                stream.write(image_bytes)
        relative_image_path = image_path.relative_to(self.workspace).as_posix()
        payload = {
            "document_key": document_key,
            "source_sha256": source_digest,
            "table_cell_binding_sha256": proof.binding_sha256,
            "page_number": proof.page_number,
            "page_width": page_width,
            "page_height": page_height,
            "value_bbox": proof.value_bbox,
            "region_bbox": BoundingBox(
                x0=float(clip.x0),
                y0=float(clip.y0),
                x1=float(clip.x1),
                y1=float(clip.y1),
            ),
            "native_value_match_count": len(native_matches),
            "rendered_dpi": rendered_dpi,
            "rendered_width": pixmap.width,
            "rendered_height": pixmap.height,
            "rendered_area_ratio": area_ratio,
            "image_sha256": image_digest,
            "image_path": relative_image_path,
            "binding_sha256": "0" * 64,
        }
        unbound = TableCellRegionProof.model_construct(**payload)
        payload["binding_sha256"] = table_cell_region_binding_sha256(unbound)
        return TableCellRegionProof.model_validate(payload)

    def reconstruct_two_column_page(
        self,
        document_key: str,
        page_number: int,
    ) -> PageLayoutReconstruction:
        """OCR the rendered left and right halves independently.

        Some annual reports contain a stale or unrelated hidden text layer on a
        visually correct audit page.  Cropping before OCR prevents one detected
        line from interleaving text that belongs to two separate columns.
        """

        path, digest = self._source_path(document_key)
        backend = self._ocr_backend
        if backend is None:
            backend = create_ocr_backend("rapidocr")
            self._ocr_backend = backend
        with fitz.open(path) as document:
            if not 1 <= page_number <= document.page_count:
                raise ValueError(f"Page {page_number} is outside 1..{document.page_count}")
            page = document[page_number - 1]
            midpoint = float(page.rect.width) / 2
            clips = (
                ("左栏", fitz.Rect(0, 0, midpoint, page.rect.height)),
                ("右栏", fitz.Rect(midpoint, 0, page.rect.width, page.rect.height)),
            )
            columns: list[ReconstructedTextColumn] = []
            for label, clip in clips:
                pixmap = page.get_pixmap(
                    clip=clip,
                    dpi=self.ocr_dpi,
                    alpha=False,
                )
                result = backend.extract(
                    pixmap.tobytes("png"),
                    width=pixmap.width,
                    height=pixmap.height,
                )
                accepted = [region for region in result.regions if region.confidence >= 0.5]
                columns.append(
                    ReconstructedTextColumn(
                        label=label,
                        text="\n".join(region.text for region in accepted),
                        average_confidence=(
                            sum(region.confidence for region in accepted) / len(accepted)
                            if accepted
                            else 0.0
                        ),
                    )
                )
        if not any(column.text for column in columns):
            raise ValueError(
                f"Rendered-page OCR returned no text: {document_key} page {page_number}"
            )
        return PageLayoutReconstruction(
            document_key=document_key,
            page_number=page_number,
            source_sha256=digest,
            columns=columns,
        )
