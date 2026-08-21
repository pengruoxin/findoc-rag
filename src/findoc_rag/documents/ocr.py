"""Optional OCR backends used by the page-level PDF extraction router."""

from __future__ import annotations

from io import BytesIO
from typing import Protocol

from pydantic import BaseModel, Field

from findoc_rag.documents.models import BoundingBox


class OcrBackendUnavailable(RuntimeError):
    """Raised when a requested OCR backend is not installed or cannot start."""


class OcrRegion(BaseModel):
    text: str
    pixel_bbox: BoundingBox
    confidence: float = Field(ge=0, le=1)


class OcrPageResult(BaseModel):
    backend: str
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    regions: list[OcrRegion]
    elapsed_ms: float = Field(default=0, ge=0)


class OcrBackend(Protocol):
    name: str

    def extract(self, image: bytes, *, width: int, height: int) -> OcrPageResult: ...


class RapidOcrBackend:
    """CPU-friendly Chinese/English OCR backed by RapidOCR and ONNX Runtime."""

    def __init__(self, *, red_channel: bool = False) -> None:
        self.red_channel = red_channel
        self.name = "rapidocr-red-channel" if red_channel else "rapidocr"
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:  # pragma: no cover - exercised without the optional extra
            raise OcrBackendUnavailable(
                "RapidOCR is unavailable. Install the project with the 'ocr' extra."
            ) from exc
        try:
            self._engine = RapidOCR()
        except Exception as exc:  # pragma: no cover - backend-specific startup failure
            raise OcrBackendUnavailable(f"RapidOCR failed to initialize: {exc}") from exc

    def extract(self, image: bytes, *, width: int, height: int) -> OcrPageResult:
        prepared = _red_channel_png(image) if self.red_channel else image
        try:
            output = self._engine(prepared)
        except Exception as exc:  # pragma: no cover - backend-specific inference failure
            raise RuntimeError(f"RapidOCR inference failed: {exc}") from exc
        boxes = getattr(output, "boxes", None)
        texts = getattr(output, "txts", ()) or ()
        scores = getattr(output, "scores", ()) or ()
        if boxes is None:
            boxes = []
        regions: list[OcrRegion] = []
        for box, text, score in zip(boxes, texts, scores, strict=True):
            if not str(text).strip():
                continue
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
            regions.append(
                OcrRegion(
                    text=str(text).strip(),
                    pixel_bbox=BoundingBox(
                        x0=max(0.0, min(xs)),
                        y0=max(0.0, min(ys)),
                        x1=min(float(width), max(xs)),
                        y1=min(float(height), max(ys)),
                    ),
                    confidence=max(0.0, min(1.0, float(score))),
                )
            )
        regions.sort(key=lambda region: (region.pixel_bbox.y0, region.pixel_bbox.x0))
        return OcrPageResult(
            backend=self.name,
            image_width=width,
            image_height=height,
            regions=regions,
            elapsed_ms=max(0.0, float(getattr(output, "elapse", 0.0)) * 1000),
        )


def create_ocr_backend(name: str) -> OcrBackend:
    if name == "rapidocr":
        return RapidOcrBackend()
    if name == "rapidocr-red-channel":
        return RapidOcrBackend(red_channel=True)
    raise ValueError(f"Unsupported OCR backend: {name}")


def _red_channel_png(image: bytes) -> bytes:
    """Suppress red seals while retaining black text using the RGB red channel."""

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - guarded by the OCR extra
        raise OcrBackendUnavailable(
            "Red-channel OCR requires Pillow from the project's 'ocr' extra."
        ) from exc
    with Image.open(BytesIO(image)) as source:
        red = source.convert("RGB").getchannel("R")
        target = BytesIO()
        red.save(target, format="PNG")
    return target.getvalue()
