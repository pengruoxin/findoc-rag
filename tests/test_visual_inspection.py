import hashlib
import json
from pathlib import Path

import fitz

from findoc_rag.documents.models import BoundingBox
from findoc_rag.documents.ocr import OcrPageResult, OcrRegion
from findoc_rag.table_cell_proof import (
    TableCellGeometryProof,
    table_cell_binding_sha256,
)
from findoc_rag.visual_inspection import PdfRegionInspector, TableCellRegionProof


def _coordinate_proof(*, bbox: BoundingBox, value: str = "360,403") -> TableCellGeometryProof:
    payload = {
        "table_id": "table-1",
        "table_type": "annual_data",
        "chunk_id": "chunk-1",
        "chunk_sha256": "a" * 64,
        "table_source": "coordinate",
        "page_start": 1,
        "page_end": 1,
        "row": "经营活动产生的现金流量净额",
        "row_index": 1,
        "column": "2023年",
        "column_index": 1,
        "value": value.replace(",", ""),
        "geometry_status": "coordinate",
        "page_number": 1,
        "value_bbox": bbox,
        "coordinate_space": "pymupdf_unrotated_page",
        "binding_sha256": "0" * 64,
    }
    unbound = TableCellGeometryProof.model_construct(**payload)
    payload["binding_sha256"] = table_cell_binding_sha256(unbound)
    return TableCellGeometryProof.model_validate(payload)


def test_pdf_region_inspector_pairs_labels_values_and_connectors(tmp_path: Path) -> None:
    pdf_path = tmp_path / "diagram.pdf"
    document = fitz.open()
    page = document.new_page(width=400, height=500)
    rows = [
        ("H shareholders", "17.00%", 100),
        ("Energy group", "69.52%", 180),
        ("Other A shareholders", "13.48%", 260),
    ]
    for label, value, y_position in rows:
        page.insert_text((40, y_position), label, fontsize=10)
        page.insert_text((220, y_position), value, fontsize=10)
        page.draw_line((150, y_position - 3), (210, y_position - 3))
    document.save(pdf_path)
    document.close()
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "document_key": "test:diagram",
                        "local_file": "diagram.pdf",
                        "sha256": digest,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    inspection = PdfRegionInspector(
        manifest_path,
        workspace=tmp_path,
    ).inspect_relationship_rows("test:diagram", 1)

    assert [row.label.text for row in inspection.relationship_rows] == [
        "Hshareholders",
        "Energygroup",
        "OtherAshareholders",
    ]
    assert [row.value.text for row in inspection.relationship_rows] == [
        "17.00%",
        "69.52%",
        "13.48%",
    ]
    assert all(row.connector_present for row in inspection.relationship_rows)


def test_pdf_region_inspector_renders_bounded_table_cell_proof(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "table.pdf"
    document = fitz.open()
    page = document.new_page(width=400, height=500)
    page.insert_text((40, 90), "2023 year", fontsize=10)
    page.insert_text((40, 150), "cash flow", fontsize=10)
    page.insert_text((220, 150), "360,403", fontsize=10)
    document.save(pdf_path)
    document.close()
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "document_key": "test:table",
                        "local_file": "table.pdf",
                        "sha256": digest,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with fitz.open(pdf_path) as source:
        word = next(item for item in source[0].get_text("words") if item[4] == "360,403")
    proof = _coordinate_proof(bbox=BoundingBox(x0=word[0], y0=word[1], x1=word[2], y1=word[3]))

    region = PdfRegionInspector(
        manifest_path,
        workspace=tmp_path,
    ).render_table_cell_region(
        "test:table",
        proof,
        output_directory=tmp_path / "regions",
    )

    assert isinstance(region, TableCellRegionProof)
    assert region.table_cell_binding_sha256 == proof.binding_sha256
    assert region.rendered_area_ratio <= 0.2
    assert region.native_value_match_count == 1
    assert (tmp_path / region.image_path).is_file()


def test_pdf_region_inspector_rejects_bbox_without_matching_value(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "table.pdf"
    document = fitz.open()
    page = document.new_page(width=400, height=500)
    page.insert_text((220, 150), "360,403", fontsize=10)
    document.save(pdf_path)
    document.close()
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "document_key": "test:table",
                        "local_file": "table.pdf",
                        "sha256": digest,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    proof = _coordinate_proof(bbox=BoundingBox(x0=20, y0=20, x1=40, y1=30))

    try:
        PdfRegionInspector(
            manifest_path,
            workspace=tmp_path,
        ).render_table_cell_region(
            "test:table",
            proof,
            output_directory=tmp_path / "regions",
        )
    except ValueError as exc:
        assert "does not intersect matching PDF text" in str(exc)
    else:
        raise AssertionError("Expected a mismatched PDF cell bbox to fail closed")


class _SequentialOcrBackend:
    name = "fake-ocr"

    def __init__(self) -> None:
        self.calls = 0

    def extract(self, image: bytes, *, width: int, height: int) -> OcrPageResult:
        assert image
        self.calls += 1
        text = "左栏风险判断" if self.calls == 1 else "右栏审计应对"
        return OcrPageResult(
            backend=self.name,
            image_width=width,
            image_height=height,
            regions=[
                OcrRegion(
                    text=text,
                    pixel_bbox=BoundingBox(x0=1, y0=1, x1=20, y1=10),
                    confidence=0.99,
                )
            ],
        )


def test_pdf_region_inspector_reconstructs_columns_before_reading_order(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "two-column.pdf"
    document = fitz.open()
    page = document.new_page(width=400, height=500)
    page.insert_text((40, 100), "left", fontsize=10)
    page.insert_text((240, 100), "right", fontsize=10)
    document.save(pdf_path)
    document.close()
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "document_key": "test:two-column",
                        "local_file": "two-column.pdf",
                        "sha256": digest,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    backend = _SequentialOcrBackend()

    reconstruction = PdfRegionInspector(
        manifest_path,
        workspace=tmp_path,
        ocr_backend=backend,
    ).reconstruct_two_column_page("test:two-column", 1)

    assert backend.calls == 2
    assert [column.label for column in reconstruction.columns] == ["左栏", "右栏"]
    assert reconstruction.evidence_text.index("左栏风险判断") < (
        reconstruction.evidence_text.index("右栏审计应对")
    )
    assert reconstruction.evidence_chunk_id.startswith(f"layout:{digest[:12]}:p1:")
