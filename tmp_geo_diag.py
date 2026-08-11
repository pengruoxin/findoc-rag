import pymupdf

from findoc_rag.table_reconstruction import (
    blocks_from_pymupdf_dict,
    reconstruct_cells,
)


def dump_spans(pdf_path: str, page_no: int, needle: str, limit: int = 12) -> None:
    pdf = pymupdf.open(pdf_path)
    page = pdf[page_no - 1]
    raw = page.get_text("dict", sort=True)
    print(f"--- page {page_no} spans containing {needle!r}")
    count = 0
    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "")
                if needle in text:
                    print(f"    bbox={tuple(round(v, 1) for v in span['bbox'])} text={text!r}")
                    count += 1
                    if count >= limit:
                        return


def show_predicted(pdf_path: str, pages: range, table_type: str) -> None:
    pdf = pymupdf.open(pdf_path)
    blocks: list[dict] = []
    for page_no in pages:
        blocks.extend(pdf[page_no - 1].get_text("dict", sort=True).get("blocks", []))
    model = blocks_from_pymupdf_dict({"blocks": blocks})
    cells = reconstruct_cells(model, table_type)
    rows: dict[str, list[tuple[str, str]]] = {}
    for cell in cells:
        rows.setdefault(cell.row, []).append((cell.column, cell.value))
    print(f"--- {table_type} predicted rows ({len(rows)})")
    for row, cols in sorted(rows.items()):
        print(f"    {row}: {cols[:4]}")


MOUTAI = r"D:\202607\raglab\data\artifacts\cninfo\600519_2024_1222993920.pdf"
YILI = r"D:\202607\raglab\data\artifacts\cninfo\600887_2024_1223421123.pdf"

dump_spans(MOUTAI, 6, "扣除非经常性损益")
dump_spans(MOUTAI, 6, "24,051,471,185.69")
show_predicted(MOUTAI, range(6, 7), "quarterly")
show_predicted(YILI, range(8, 9), "quarterly")
show_predicted(MOUTAI, range(108, 109), "note_cost")
show_predicted(MOUTAI, range(9, 10), "segment")
