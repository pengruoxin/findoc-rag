import importlib.util
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_coordinate_reconstruction.py"
)
SPEC = importlib.util.spec_from_file_location(
    "evaluate_coordinate_reconstruction", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
render_markdown = MODULE.render_markdown


def _summary(input_mode: str) -> dict:
    return {
        "input_mode": input_mode,
        "gold_cells_total": 1,
        "correct_cells_total": 1,
        "recall_total": 1.0,
        "tables": [],
    }


def test_markdown_identifies_source_pdf_geometry() -> None:
    markdown = render_markdown(
        _summary("source-pdf-whole-pages-no-cropping")
    )

    assert "完整 pymupdf blocks" in markdown
    assert "Document IR v2" not in markdown


def test_markdown_identifies_persisted_ir_geometry() -> None:
    summary = _summary("persisted-ir-v2-whole-pages-no-cropping")
    summary["document_ir_provenance"] = [
        {
            "version_id": "version-1",
            "document_id": "sha256:document",
            "processing_fingerprint": "processing-sha",
            "document_ir_sha256": "document-ir-sha",
        }
    ]
    markdown = render_markdown(summary)

    assert "持久化 Document IR v2 line/span geometry" in markdown
    assert "完整 pymupdf blocks" not in markdown
    assert "`version-1` / `sha256:document`" in markdown
    assert "processing `processing-sha`" in markdown
    assert "document SHA `document-ir-sha`" in markdown
