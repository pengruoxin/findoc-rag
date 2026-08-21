from io import BytesIO

from PIL import Image

from findoc_rag.documents.ocr import _red_channel_png


def test_red_channel_preprocessing_suppresses_red_but_keeps_black() -> None:
    source = Image.new("RGB", (2, 1))
    source.putdata([(255, 0, 0), (0, 0, 0)])
    raw = BytesIO()
    source.save(raw, format="PNG")

    prepared = Image.open(BytesIO(_red_channel_png(raw.getvalue())))

    assert list(prepared.get_flattened_data()) == [255, 0]
