from findoc_rag.chunking import (
    ChunkingConfig,
    build_chunking_report,
    chunk_document,
    detect_repeated_margin_elements,
    estimate_tokens,
)
from findoc_rag.documents.models import (
    BoundingBox,
    DocumentElement,
    DocumentPage,
    ParsedDocument,
)


def element(
    element_id: str,
    text: str,
    y0: float,
    y1: float,
    *,
    font_size: float = 10,
    bold: bool = False,
) -> DocumentElement:
    return DocumentElement(
        element_id=element_id,
        element_type="text",
        text=text,
        bbox=BoundingBox(x0=50, y0=y0, x1=550, y1=y1),
        reading_order=0,
        font_size=font_size,
        font_name="TestFont",
        is_bold=bold,
    )


def sample_document() -> ParsedDocument:
    pages = []
    for page_number in range(1, 6):
        pages.append(
            DocumentPage(
                page_number=page_number,
                width=600,
                height=800,
                elements=[
                    element(f"h-{page_number}", "示例公司2024年年度报告", 15, 35),
                    element(
                        f"title-{page_number}",
                        "第一节 公司概况" if page_number == 1 else "一、经营情况",
                        100,
                        130,
                        font_size=16,
                        bold=True,
                    ),
                    element(
                        f"body-{page_number}",
                        "报告期内公司经营保持稳定。营业收入持续增长，产品结构进一步优化。" * 8,
                        150,
                        650,
                    ),
                    element(f"f-{page_number}", f"{page_number} / 5", 770, 790),
                ],
                extracted_character_count=300,
                image_count=0,
                needs_ocr=False,
            )
        )
    return ParsedDocument(
        document_id="sha256:abc",
        source_path="sample.pdf",
        filename="sample.pdf",
        content_sha256="abc",
        page_count=5,
        pages=pages,
        parser="test",
        parser_version="1",
    )


def test_repeated_headers_and_page_numbers_are_excluded() -> None:
    document = sample_document()

    excluded = detect_repeated_margin_elements(document, ChunkingConfig())

    assert {f"h-{page}" for page in range(1, 6)} <= excluded
    assert {f"f-{page}" for page in range(1, 6)} <= excluded


def test_chunking_preserves_sections_provenance_and_budget() -> None:
    document = sample_document()
    config = ChunkingConfig(
        target_tokens=120,
        max_tokens=160,
        min_tokens=30,
        overlap_tokens=20,
    )

    chunks = chunk_document(document, config)

    assert len(chunks) > 5
    assert all(chunk.estimated_token_count <= config.max_tokens for chunk in chunks)
    assert all(chunk.element_references for chunk in chunks)
    assert all(chunk.page_start <= chunk.page_end for chunk in chunks)
    assert any("第一节 公司概况" in chunk.section_path for chunk in chunks)
    assert all("示例公司2024年年度报告" not in chunk.text for chunk in chunks)
    assert all(" / 5" not in chunk.text for chunk in chunks)

    report = build_chunking_report(document, chunks, config)
    assert report.source_element_coverage == 1.0
    assert report.chunks_above_max_tokens == 0
    assert report.repeated_margin_element_count == 10


def test_estimate_tokens_handles_chinese_and_english() -> None:
    assert estimate_tokens("营业收入增长") == 6
    assert estimate_tokens("revenue increased 10 percent") == 6
