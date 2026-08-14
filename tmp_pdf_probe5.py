"""Probe 5: intra-term line breaks (lexical retrieval damage) + table-type coverage."""
import json
import re
from collections import Counter
from pathlib import Path

import pymupdf

ROOT = Path(r"D:\202607\raglab")
DOCS = {
    "moutai": ("7961508deeffb5e66ae88808", ROOT / "data/artifacts/cninfo/600519_2024_1222993920.pdf"),
    "yili": ("e96cf669106c99e4e283ca45", ROOT / "data/artifacts/cninfo/600887_2024_1223421123.pdf"),
}
CJK = r"\u4e00-\u9fff"
FIN_TERMS = [
    "所有者权益合计", "归属于母公司所有者权益合计", "营业总收入", "营业总成本",
    "经营活动产生的现金流量净额", "归属于上市公司股东的净利润",
    "归属于上市公司股东的扣除非经常性损益的净利润", "加权平均净资产收益率",
    "研发费用", "货币资金", "应收账款", "存货", "商誉", "递延所得税资产",
    "少数股东权益", "基本每股收益", "稀释每股收益", "资产总计", "负债合计",
]


def intra_term_breaks(label: str, version_id: str) -> None:
    path = ROOT / "data/catalog/versions" / version_id / "chunks.jsonl"
    chunks = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()]
    # A line ending with CJK followed by a line starting with CJK, where neither
    # side ends in punctuation -> the term was wrapped mid-word by the PDF.
    wrapped = 0
    examples = []
    for chunk in chunks:
        lines = [ln.strip() for ln in chunk["text"].splitlines() if ln.strip()]
        for a, b in zip(lines, lines[1:]):
            if (
                re.search(rf"[{CJK}]$", a)
                and re.match(rf"^[{CJK}）)]", b)
                and not re.search(r"[。；；：:、，,！？]$", a)
                and len(a) <= 30
            ):
                wrapped += 1
                if len(examples) < 6:
                    examples.append(f"{a[-6:]}//{b[:6]}")
    print(f"[{label}] candidate mid-term line wraps in chunk text: {wrapped}")
    print(f"  e.g. {examples}")

    # Direct check: are canonical financial terms searchable as contiguous strings?
    joined_all = "\n".join(c["text"] for c in chunks)
    collapsed = re.sub(r"\s+", "", joined_all)
    print(f"[{label}] term reachability (contiguous vs whitespace-collapsed):")
    for term in FIN_TERMS:
        direct = joined_all.count(term)
        collapsed_hits = collapsed.count(term)
        if collapsed_hits > direct:
            print(f"  {term}: contiguous {direct} vs collapsed {collapsed_hits}  <-- lost {collapsed_hits - direct}")


def table_type_coverage(label: str, pdf_path: Path) -> None:
    pdf = pymupdf.open(pdf_path)
    total_tables = 0
    header_counter: Counter[str] = Counter()
    for page in pdf:
        try:
            tables = page.find_tables().tables
        except Exception:
            continue
        for t in tables:
            total_tables += 1
            try:
                rows = t.extract()
            except Exception:
                continue
            if rows and rows[0]:
                head = "|".join((c or "").strip()[:10] for c in rows[0][:3])
                header_counter[head] += 1
    print(
        f"[{label}] detected tables: {total_tables}; annotated by table-eval: "
        f"{'5 (moutai)' if label == 'moutai' else '5 (yili)'} -> coverage "
        f"{5 / max(1, total_tables):.1%}"
    )
    print(f"  most repeated table headers: {header_counter.most_common(6)}")
    pdf.close()


def main() -> None:
    for label, (version_id, pdf_path) in DOCS.items():
        print("=" * 90)
        intra_term_breaks(label, version_id)
        table_type_coverage(label, pdf_path)


if __name__ == "__main__":
    main()
