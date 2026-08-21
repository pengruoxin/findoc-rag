from findoc_rag.documents.geometry import (
    bbox_to_display_coordinates,
    page_display_dimensions,
)
from findoc_rag.documents.models import BoundingBox, DocumentPage


def _page(rotation: int) -> DocumentPage:
    return DocumentPage(
        page_number=1,
        width=600,
        height=800,
        elements=[],
        extracted_character_count=0,
        image_count=0,
        needs_ocr=False,
        rotation=rotation,
    )


def test_page_display_dimensions_swap_only_for_quarter_turns() -> None:
    assert page_display_dimensions(_page(0)) == (600, 800)
    assert page_display_dimensions(_page(90)) == (800, 600)
    assert page_display_dimensions(_page(180)) == (600, 800)
    assert page_display_dimensions(_page(270)) == (800, 600)


def test_rotated_boxes_map_to_visual_rows() -> None:
    first = bbox_to_display_coordinates(
        BoundingBox(x0=100, y0=600, x1=120, y1=700),
        page_width=600,
        page_height=800,
        rotation=90,
    )
    second = bbox_to_display_coordinates(
        BoundingBox(x0=101, y0=300, x1=121, y1=500),
        page_width=600,
        page_height=800,
        rotation=90,
    )

    assert first.y0 == 100
    assert second.y0 == 101
    assert first.x0 < second.x0


def test_270_degree_rotation_keeps_box_inside_display_bounds() -> None:
    rotated = bbox_to_display_coordinates(
        BoundingBox(x0=10, y0=20, x1=110, y1=70),
        page_width=600,
        page_height=800,
        rotation=270,
    )

    assert rotated == BoundingBox(x0=20, y0=490, x1=70, y1=590)
