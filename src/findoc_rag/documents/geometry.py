"""Coordinate transforms shared by PDF extraction and layout reconstruction."""

from __future__ import annotations

from findoc_rag.documents.models import BoundingBox, DocumentPage


def page_display_dimensions(page: DocumentPage) -> tuple[float, float]:
    """Return dimensions after applying the page's clockwise display rotation."""

    if page.rotation % 180:
        return page.height, page.width
    return page.width, page.height


def bbox_to_display_coordinates(
    bbox: BoundingBox,
    *,
    page_width: float,
    page_height: float,
    rotation: int,
) -> BoundingBox:
    """Map an unrotated PyMuPDF box into the visually displayed coordinate space."""

    normalized_rotation = rotation % 360
    if normalized_rotation not in {0, 90, 180, 270}:
        raise ValueError(f"Unsupported PDF page rotation: {rotation}")

    def transform(x: float, y: float) -> tuple[float, float]:
        if normalized_rotation == 0:
            return x, y
        if normalized_rotation == 90:
            return page_height - y, x
        if normalized_rotation == 180:
            return page_width - x, page_height - y
        return y, page_width - x

    corners = [
        transform(bbox.x0, bbox.y0),
        transform(bbox.x0, bbox.y1),
        transform(bbox.x1, bbox.y0),
        transform(bbox.x1, bbox.y1),
    ]
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    return BoundingBox(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys))


def element_display_bbox(page: DocumentPage, bbox: BoundingBox) -> BoundingBox:
    return bbox_to_display_coordinates(
        bbox,
        page_width=page.width,
        page_height=page.height,
        rotation=page.rotation,
    )
