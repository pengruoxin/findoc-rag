"""Coordinate-aware reconstruction for Chinese annual-report tables.

Standard-library only. PyMuPDF is optional: pass the result of
``page.get_text("dict")`` to :func:`blocks_from_pymupdf_dict`.

The module keeps the legacy ``extract_cells(text, table_type)`` entry point
available. If a project-local ``table_extraction.extract_cells`` exists it is
used as the text fallback; otherwise a conservative built-in parser covers the
five regulatory table families used by this module.
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

TableType = Literal["quarterly", "note_cost", "segment", "annual_data", "concentration"]
WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class ExtractedCell:
    row: str
    column: str
    value: str
    section: str = ""


def normalize_value(raw: str) -> str:
    return raw.replace(",", "").replace("+", "").replace("−", "-")


def normalize_label(raw: str) -> str:
    return WHITESPACE.sub("", raw)


@dataclass(frozen=True)
class Span:
    text: str
    bbox: tuple[float, float, float, float]
    font: str = ""
    size: float = 0.0
    bold: bool = False


@dataclass(frozen=True)
class Line:
    spans: list[Span]


@dataclass(frozen=True)
class Block:
    lines: list[Line]
    page: int = 0


@dataclass(frozen=True)
class _Token:
    text: str
    bbox: tuple[float, float, float, float]
    numeric: bool
    source_text: str
    page: int = 0

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def x1(self) -> float:
        return self.bbox[2]

    @property
    def y1(self) -> float:
        return self.bbox[3]

    @property
    def xc(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def height(self) -> float:
        return max(0.01, self.y1 - self.y0)


@dataclass
class _Band:
    tokens: list[_Token]
    y0: float
    y1: float
    page: int = 0

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def numeric_tokens(self) -> list[_Token]:
        return [t for t in self.tokens if t.numeric]

    @property
    def text_tokens(self) -> list[_Token]:
        return [t for t in self.tokens if not t.numeric]


_NUM_RE = re.compile(
    r"(?<![\w.])[-+−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?![\w.])"
)
_FULL_NUM_RE = re.compile(r"^[-+−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?$")
_YEAR_RE = re.compile(r"^(20\d{2})年(?:末)?$")
_QUARTER_RE = re.compile(r"^第([一二三四])季度$")
_UNIT_RE = re.compile(r"单位[:：]?\s*(元|万元|亿元)")

_EXPECTED_COLUMNS: dict[TableType, tuple[str, ...]] = {
    "quarterly": ("第一季度", "第二季度", "第三季度", "第四季度"),
    "note_cost": ("本期收入", "本期成本", "上期收入", "上期成本"),
    "segment": ("营业收入", "营业成本", "毛利率"),
    "annual_data": ("2024年", "2023年", "2022年"),
    "concentration": (),
}

_STANDARD_FINANCE_LABELS = (
    "归属于上市公司股东的扣除非经常性损益后的净利润",
    "归属于上市公司股东的扣除非经常性损益的净利润",
    "归属于上市公司股东的净利润",
    "经营活动产生的现金流量净额",
    "归属于上市公司股东的净资产",
    "营业收入",
    "总资产",
    "股本",
)

_HEADER_FRAGMENTS = {
    "项目", "收入", "成本", "本期发生额", "上期发生额",
    "分行业", "分产品", "分地区", "销售模式",
    "营业收入", "营业成本", "毛利率",
    "营业收入比", "营业成本比", "毛利率比",
    "上年增减", "上年增减（%）", "上年增减(%)",
    "减（%）", "（%）", "(%)", "%",
}
_SECTION_PATTERNS = (
    ("分行业", re.compile(r"主营业务分行业情况")),
    ("分产品", re.compile(r"主营业务分产品情况")),
    ("分地区", re.compile(r"主营业务分地区情况")),
    ("分销售模式", re.compile(r"主营业务分销售模式情况")),
)

_REGION_MARKERS: dict[TableType, str] = {
    "quarterly": "分季度主要财务数据",
    "note_cost": "营业收入和营业成本情况",
    "segment": "主营业务分行业情况",
    "annual_data": "主要会计数据",
    "concentration": "主要销售客户及主要供应商情况",
}

_REGION_END_MARKERS: dict[TableType, tuple[str, ...]] = {
    "quarterly": ("季度数据与已披露",),
    "note_cost": ("注：", "注:",),
    "segment": ("其他说明", "情况的说明", "报告期内向单个客户"),
    "annual_data": ("报告期末公司前三年",),
    "concentration": ("其他说明",),
}

_HEADING_START = re.compile(
    r"^(?:[（(]\s*[一二三四五六七八九十\d]+\s*[）)][.、]?|"
    r"[一二三四五六七八九十\d]+(?:[、.](?!\d)))"
)

_PUNCT_ONLY = re.compile(r"^[（()）\-—\s]+$")
_RANGE_TAIL = re.compile(r"^月份?[）)]$")
_PROSE_MARKERS = (
    "：",
    "，",
    "。",
    "；",
    "主要原因",
    "说明",
    "春节",
    "备货",
    "增加",
    "减少",
)
_ANCHOR_EXCLUDE = (
    "报告期末公司前三年",
    "差异说明",
    "情况的说明",
    "其他说明",
)

_QUARTERLY_ROW_LABELS = (
    "营业收入",
    "归属于上市公司股东的净利润",
    "归属于上市公司股东的扣除非经常性损益后的净利润",
    "经营活动产生的现金流量净额",
)

try:  # Optional compatibility with the project-local legacy parser.
    from findoc_rag.table_extraction import (
        extract_cells as _legacy_extract_cells,
    )
except ImportError:  # pragma: no cover - absence is expected in standalone use.
    _legacy_extract_cells = None


def _is_bold(font: str, flags: int = 0, explicit: object = None) -> bool:
    if isinstance(explicit, bool):
        return explicit
    name = (font or "").lower()
    if any(k in name for k in ("bold", "semibold", "demi", "black", "heavy", "黑体")):
        return True
    # PyMuPDF span flags vary by font; bit 4 is commonly "bold" in recent builds.
    return bool(flags & 16)


def blocks_from_pymupdf_dict(
    raw: dict | list[dict], page: int = 0
) -> list[Block]:
    """Convert PyMuPDF ``get_text("dict")`` output or block-like dicts.

    Unknown/non-text blocks are ignored. Missing font/size/bold metadata is
    tolerated so the same constructor also accepts lightweight synthetic tests.
    """
    raw_blocks = raw.get("blocks", []) if isinstance(raw, dict) else raw
    out: list[Block] = []
    for b in raw_blocks or []:
        if b.get("type", 0) not in (0, None):
            continue
        lines: list[Line] = []
        for ln in b.get("lines", []) or []:
            spans: list[Span] = []
            for sp in ln.get("spans", []) or []:
                text = str(sp.get("text", ""))
                bbox = sp.get("bbox")
                if not text.strip() or not bbox or len(bbox) != 4:
                    continue
                font = str(sp.get("font", ""))
                spans.append(
                    Span(
                        text=text,
                        bbox=tuple(float(v) for v in bbox),
                        font=font,
                        size=float(sp.get("size", 0.0) or 0.0),
                        bold=_is_bold(font, int(sp.get("flags", 0) or 0), sp.get("bold")),
                    )
                )
            if spans:
                lines.append(Line(spans))
        if lines:
            out.append(Block(lines, page=page))
    return out


def _coerce_blocks(blocks: Sequence[dict | Block]) -> list[Block]:
    if not blocks:
        return []
    if all(isinstance(b, Block) for b in blocks):
        return list(blocks)  # type: ignore[arg-type]
    return blocks_from_pymupdf_dict(list(blocks))  # type: ignore[arg-type]


def _split_span(span: Span, page: int = 0) -> list[_Token]:
    """Split mixed spans into approximate label/number tokens.

    PyMuPDF often emits a whole table row as one span. Character-proportional
    boxes are sufficient for column clustering without a PDF-font dependency.
    """
    text = span.text
    stripped = normalize_label(text)
    # Year/quarter headers contain digits but are categorical column labels.
    if _YEAR_RE.match(stripped) or _QUARTER_RE.match(stripped):
        return (
            [_Token(text.strip(), span.bbox, False, span.text, page)]
            if text.strip()
            else []
        )
    matches = list(_NUM_RE.finditer(text))
    if not matches:
        return (
            [_Token(text.strip(), span.bbox, False, span.text, page)]
            if text.strip()
            else []
        )

    x0, y0, x1, y1 = span.bbox
    width = max(0.01, x1 - x0)
    n = max(1, len(text))
    out: list[_Token] = []
    last = 0

    def bbox_for(a: int, b: int) -> tuple[float, float, float, float]:
        return (x0 + width * a / n, y0, x0 + width * b / n, y1)

    for m in matches:
        prefix = text[last:m.start()]
        if prefix.strip():
            a = last + len(prefix) - len(prefix.lstrip())
            b = m.start() - (len(prefix) - len(prefix.rstrip()))
            if b > a:
                out.append(
                    _Token(text[a:b].strip(), bbox_for(a, b), False, span.text, page)
                )
        raw = m.group(0).strip()
        if raw:
            out.append(
                _Token(raw, bbox_for(m.start(), m.end()), True, span.text, page)
            )
        last = m.end()
    suffix = text[last:]
    if suffix.strip():
        a = last + len(suffix) - len(suffix.lstrip())
        b = len(text) - (len(suffix) - len(suffix.rstrip()))
        if b > a:
            out.append(
                _Token(text[a:b].strip(), bbox_for(a, b), False, span.text, page)
            )
    return out


def _flatten_tokens(blocks: Sequence[Block]) -> list[_Token]:
    tokens: list[_Token] = []
    for block in blocks:
        for line in block.lines:
            for span in line.spans:
                tokens.extend(_split_span(span, block.page))
    filtered: list[_Token] = []
    for token in tokens:
        text = token.text.strip()
        if not text:
            continue
        if _PUNCT_ONLY.fullmatch(text) or _RANGE_TAIL.fullmatch(text):
            continue
        filtered.append(token)
    return filtered


def _is_anchor_candidate(token: _Token) -> bool:
    normalized = normalize_label(token.text)
    return not any(marker in normalized for marker in _ANCHOR_EXCLUDE)


def _is_prose_label(label: str) -> bool:
    return any(marker in label for marker in _PROSE_MARKERS) or bool(
        re.search(r"20\d{2}", label)
    )


def cluster_row_bands(tokens: Sequence[_Token], gap_factor: float = 0.28) -> list[_Band]:
    """Cluster tokens by overlapping/nearby y intervals.

    Transitive overlap is intentional: a value centered between two wrapped
    label lines can bridge them into one logical row band.
    """
    if not tokens:
        return []
    heights = sorted(t.height for t in tokens)
    median_h = heights[len(heights) // 2]
    gap = max(1.0, median_h * gap_factor)

    ordered = sorted(tokens, key=lambda t: (t.y0, t.y1, t.x0))
    bands: list[_Band] = []
    for tok in ordered:
        candidates: list[tuple[float, int]] = []
        for i, band in enumerate(bands):
            if band.page != tok.page:
                continue
            overlap = min(tok.y1, band.y1) - max(tok.y0, band.y0)
            near = max(tok.y0, band.y0) - min(tok.y1, band.y1)
            if overlap >= -gap or near <= gap:
                candidates.append((abs(tok.yc - band.yc), i))
        if candidates:
            _, i = min(candidates)
            band = bands[i]
            band.tokens.append(tok)
            band.y0 = min(band.y0, tok.y0)
            band.y1 = max(band.y1, tok.y1)
        else:
            bands.append(_Band([tok], tok.y0, tok.y1, tok.page))

    # Merge overlapping bands created before a bridging token arrived.
    changed = True
    while changed:
        changed = False
        bands.sort(key=lambda b: b.y0)
        merged: list[_Band] = []
        for b in bands:
            if merged and b.page == merged[-1].page and b.y0 <= merged[-1].y1 + gap:
                prev = merged[-1]
                prev.tokens.extend(b.tokens)
                prev.y0 = min(prev.y0, b.y0)
                prev.y1 = max(prev.y1, b.y1)
                changed = True
            else:
                merged.append(b)
        bands = merged
    for band in bands:
        band.tokens.sort(key=lambda t: (t.y0, t.x0))
    return bands


def cluster_numeric_columns(tokens: Sequence[_Token]) -> list[float]:
    """Return stable x-centers for repeated numeric columns."""
    nums = sorted((t for t in tokens if t.numeric), key=lambda t: t.xc)
    if not nums:
        return []
    widths = sorted(max(1.0, t.x1 - t.x0) for t in nums)
    median_w = widths[len(widths) // 2]
    tol = max(8.0, median_w * 0.35)
    clusters: list[list[float]] = []
    for t in nums:
        if not clusters or abs(t.xc - sum(clusters[-1]) / len(clusters[-1])) > tol:
            clusters.append([t.xc])
        else:
            clusters[-1].append(t.xc)
    return [sum(c) / len(c) for c in clusters]


def detect_unit(text: str) -> str:
    m = _UNIT_RE.search(text)
    return m.group(1) if m else ""


def _canonical_header(text: str, table_type: TableType) -> str | None:
    s = normalize_label(text)
    if table_type == "quarterly":
        m = _QUARTER_RE.match(s)
        return s if m else None
    if table_type == "annual_data":
        m = _YEAR_RE.match(s)
        return f"{m.group(1)}年" if m else None
    if table_type == "note_cost":
        for name in _EXPECTED_COLUMNS["note_cost"]:
            if s == name:
                return name
        return None
    if table_type == "segment":
        return s if s in _EXPECTED_COLUMNS["segment"] else None
    return None


def _header_positions(tokens: Sequence[_Token], table_type: TableType) -> dict[str, float]:
    """Map canonical headers to x positions when explicit header spans exist."""
    out: dict[str, list[float]] = {}
    for t in tokens:
        if t.numeric:
            continue
        canon = _canonical_header(t.text, table_type)
        if canon:
            out.setdefault(canon, []).append(t.xc)
    # note_cost usually has two-level headers; infer by x order of the four
    # leaf words only when they are explicit spans.
    if table_type == "note_cost":
        income_cost = [t for t in tokens if not t.numeric and normalize_label(t.text) in ("收入", "成本")]
        if len(income_cost) >= 4:
            ordered = sorted(income_cost, key=lambda t: t.xc)
            for name, tok in zip(_EXPECTED_COLUMNS["note_cost"], ordered[:4]):
                out.setdefault(name, []).append(tok.xc)
    return {k: sum(v) / len(v) for k, v in out.items()}


def _section_at_y(tokens: Sequence[_Token], page: int, y: float) -> str:
    latest_y = -math.inf
    latest = ""
    for t in tokens:
        if t.numeric or t.page != page or t.y0 > y:
            continue
        s = normalize_label(t.text)
        for name, pat in _SECTION_PATTERNS:
            if pat.search(s) and t.y0 >= latest_y:
                latest_y = t.y0
                latest = name
    return latest


def _is_headerish(text: str, table_type: TableType) -> bool:
    s = normalize_label(text)
    if not s:
        return True
    if _canonical_header(s, table_type):
        return True
    if s in _HEADER_FRAGMENTS:
        return True
    if "单位：" in text or "单位:" in text or "币种" in text:
        return True
    if any(p.search(s) for _, p in _SECTION_PATTERNS):
        return True
    if s.startswith(("主要会计数据", "季度数据与已披露", "本期比上年", "本期末比上年")):
        return True
    return any(k in s for k in ("比上年增", "上年增减", "毛利率比上年"))


def _is_region_boundary(text: str, table_type: TableType) -> bool:
    s = normalize_label(text)
    if any(marker in s for marker in _REGION_END_MARKERS.get(table_type, ())):
        return True
    if table_type == "annual_data" and "主要财务指标" in s:
        return True
    return bool(_HEADING_START.match(s))


def _is_split_heading_boundary(
    tokens: Sequence[_Token], index: int, table_type: TableType
) -> bool:
    """Detect headings split into separate number and ')' spans."""
    token = tokens[index]
    if not token.numeric or not re.fullmatch(r"\d{1,3}", token.text):
        return False
    for other in tokens[index + 1 :]:
        if other.page != token.page:
            break
        if other.y0 > token.y1 + 2.0:
            break
        if not other.numeric and normalize_label(other.text).startswith(
            (")", "）")
        ):
            return True
    return False


def _localize_tokens(
    tokens: Sequence[_Token],
    region: str,
    table_type: TableType,
) -> list[_Token]:
    """Keep only tokens belonging to the target table's y-region.

    The anchor is the region title (when provided) or a table-type marker;
    the region ends at the first heading-like / type-specific end token after
    it, otherwise at the bottom of the supplied pages.
    """
    if not tokens:
        return []
    anchor: _Token | None = None
    candidates = [t for t in tokens if not t.numeric]
    norm_region = normalize_label(region)
    if norm_region:
        region_matches = [
            t
            for t in candidates
            if norm_region in normalize_label(t.text)
            and _is_anchor_candidate(t)
        ]
        anchor = max(region_matches, key=lambda t: (t.y0, t.x0), default=None)
    if anchor is None:
        marker = _REGION_MARKERS.get(table_type)
        if marker:
            marker_matches = [
                t
                for t in candidates
                if marker in normalize_label(t.text)
                and _is_anchor_candidate(t)
            ]
            anchor = max(
                marker_matches, key=lambda t: (t.y0, t.x0), default=None
            )
    if anchor is None:
        return list(tokens)

    anchor_page = anchor.page
    start_y = anchor.y0
    boundary_page = anchor_page
    end_y = max((t.y1 for t in tokens if t.page == anchor_page), default=0.0)
    ordered = sorted(tokens, key=lambda t: (t.y0, t.x0))
    for index, token in enumerate(ordered):
        if (token.page, token.y0) <= (anchor_page, anchor.y1 + 1.0):
            continue
        if _is_region_boundary(token.text, table_type) or (
            token.numeric and _is_split_heading_boundary(ordered, index, table_type)
        ):
            boundary_page = token.page
            end_y = token.y0
            break
    kept: list[_Token] = []
    for token in tokens:
        if token.page < anchor_page or token.page > boundary_page:
            continue
        if token.page == anchor_page and token.y0 < start_y - 2.0:
            continue
        if token.page == boundary_page and token.y1 > end_y + 1.0:
            continue
        kept.append(token)
    return kept


def _label_from_band(
    band: _Band,
    first_numeric_x: float,
    table_type: TableType,
) -> str:
    parts = [
        t for t in band.text_tokens
        if t.x0 < first_numeric_x and not _is_headerish(t.text, table_type)
    ]
    if not parts and len(band.numeric_tokens) >= 3:
        # Column names can double as row labels (e.g. 营业收入 in quarterly);
        # when the band clearly carries a numeric row, trust the geometry.
        parts = [
            t
            for t in band.text_tokens
            if t.x0 < first_numeric_x and not _is_prose_label(t.text)
        ]
    if not parts:
        return ""
    parts.sort(key=lambda t: (t.y0, t.x0))
    return normalize_label("".join(t.text for t in parts))


def _nearest_pending_label(
    bands: Sequence[_Band],
    i: int,
    first_numeric_x: float,
    table_type: TableType,
) -> str:
    """Absorb immediately preceding label-only bands for wrapped row labels."""
    pieces: list[str] = []
    current_y = bands[i].y0
    heights = [t.height for b in bands for t in b.tokens]
    median_h = sorted(heights)[len(heights)//2] if heights else 9.0
    max_gap = max(3.0, median_h * 1.45)
    j = i - 1
    while j >= 0:
        prev = bands[j]
        if prev.page != bands[i].page or prev.numeric_tokens:
            break
        gap = current_y - prev.y1
        if gap > max_gap:
            break
        label = _label_from_band(prev, first_numeric_x, table_type)
        if not label:
            if any(_is_headerish(t.text, table_type) for t in prev.text_tokens):
                break
            j -= 1
            current_y = prev.y0
            continue
        pieces.append(label)
        current_y = prev.y0
        j -= 1
    return "".join(reversed(pieces))


def _repair_quarterly_labels(cells: list[ExtractedCell]) -> list[ExtractedCell]:
    """Reattach split quarterly row labels (label tail lands in the next band)."""
    by_row: dict[str, list[ExtractedCell]] = {}
    order: list[str] = []
    for cell in cells:
        if cell.row not in by_row:
            by_row[cell.row] = []
            order.append(cell.row)
        by_row[cell.row].append(cell)
    relabel: dict[str, str] = {}
    for row in order:
        compact = normalize_label(row)
        for canonical in _QUARTERLY_ROW_LABELS:
            if not canonical.startswith(compact) or canonical == compact:
                continue
            suffix = canonical[len(compact):]
            for other in order:
                if other == row:
                    continue
                other_compact = normalize_label(other)
                if other_compact.startswith(suffix):
                    relabel[row] = canonical
                    remainder = other[len(suffix):]
                    if remainder:
                        relabel[other] = remainder
                    else:
                        relabel[other] = ""
            break
    if not relabel:
        return cells
    out: list[ExtractedCell] = []
    for cell in cells:
        target = relabel.get(cell.row, cell.row)
        if target:
            out.append(
                ExtractedCell(
                    row=target,
                    column=cell.column,
                    value=cell.value,
                    section=cell.section,
                )
            )
    return out


def _assign_columns(
    nums: Sequence[_Token],
    table_type: TableType,
    header_pos: dict[str, float],
    global_cols: Sequence[float],
) -> list[tuple[str, _Token]]:
    ordered = sorted(nums, key=lambda t: t.xc)
    expected = list(_EXPECTED_COLUMNS[table_type])
    if not expected:
        return []

    # Segment sub-tables have a stable regulatory row schema: the first three
    # value columns are 营业收入/营业成本/毛利率. Header geometry is unreliable
    # here because "营业收入比/上年增" fragments pollute header positions.
    if table_type == "segment":
        return list(zip(expected, ordered[:3]))

    # Prefer actual header geometry if enough canonical positions were observed.
    if len(header_pos) >= min(2, len(expected)):
        pairs: list[tuple[str, _Token]] = []
        used: set[int] = set()
        for col in expected:
            if col not in header_pos:
                continue
            candidates = [(abs(tok.xc - header_pos[col]), k, tok) for k, tok in enumerate(ordered) if k not in used]
            if not candidates:
                continue
            _, k, tok = min(candidates)
            used.add(k)
            pairs.append((col, tok))
        if len(pairs) >= min(len(ordered), len(expected), 2):
            return pairs

    # Annual regulatory tables usually place YoY between 2023 and 2022.
    if table_type == "annual_data" and len(ordered) >= 4:
        return list(zip(expected, [ordered[0], ordered[1], ordered[-1]]))

    # Segment tables contain additional YoY/delta metrics after gross margin;
    # the first three value columns are the stable revenue/cost/margin schema.
    if table_type == "segment":
        return list(zip(expected, ordered[:3]))

    return list(zip(expected, ordered[: len(expected)]))


def _text_from_blocks(blocks: Sequence[Block]) -> str:
    lines: list[tuple[float, float, str]] = []
    for b in blocks:
        for ln in b.lines:
            if not ln.spans:
                continue
            y = min(s.bbox[1] for s in ln.spans)
            x = min(s.bbox[0] for s in ln.spans)
            text = " ".join(s.text for s in sorted(ln.spans, key=lambda s: s.bbox[0])).strip()
            if text:
                lines.append((y, x, text))
    return "\n".join(t for _, _, t in sorted(lines))


def reconstruct_cells(
    blocks: list[dict] | list[Block],
    table_type: TableType,
    region: str = "",
) -> list[ExtractedCell]:
    """Reconstruct cells from span/line geometry, with conservative fallback."""
    model = _coerce_blocks(blocks)
    tokens = _flatten_tokens(model)
    if not tokens:
        return []

    text = _text_from_blocks(model)
    if table_type == "concentration":
        return extract_cells(text, table_type)

    tokens = _localize_tokens(tokens, region, table_type)
    if not tokens:
        return extract_cells(text, table_type)

    nums_all = [t for t in tokens if t.numeric]
    text_all = [t for t in tokens if not t.numeric]
    if not nums_all or not text_all:
        return extract_cells(text, table_type)

    bands = cluster_row_bands(tokens)
    header_pos = _header_positions(tokens, table_type)
    global_cols = cluster_numeric_columns(tokens)

    out: list[ExtractedCell] = []
    for i, band in enumerate(bands):
        nums = band.numeric_tokens
        if not nums:
            continue
        first_x = min(t.x0 for t in nums)
        label = _label_from_band(band, first_x, table_type)
        pending = _nearest_pending_label(bands, i, first_x, table_type)
        if pending and (not label or not label.startswith(pending)):
            label = pending + label
        if not label:
            continue

        # Reject sentence-like prose and growth annotations as row labels.
        if (
            len(label) > 80
            or _is_prose_label(label)
            or any(
                k in label
                for k in ("个百分点", "适用", "不适用", "年度销售总额", "年度采购总额")
            )
        ):
            continue

        assignments = _assign_columns(nums, table_type, header_pos, global_cols)
        for column, tok in assignments:
            value = normalize_value(tok.text.rstrip("%"))
            if value:
                section = (
                    _section_at_y(tokens, band.page, band.y0)
                    if table_type == "segment"
                    else ""
                )
                out.append(ExtractedCell(row=label, column=column, value=value, section=section))

    # Geometry can be too sparse (e.g. block-level only or no row labels).
    min_expected = {"quarterly": 2, "note_cost": 2, "segment": 2, "annual_data": 2}.get(table_type, 1)
    if len(out) < min_expected:
        fallback = extract_cells(text, table_type)
        if fallback:
            return fallback

    # Stable de-duplication by evaluation key.
    seen: set[tuple[str, str, str]] = set()
    deduped: list[ExtractedCell] = []
    for c in out:
        key = (normalize_label(c.row), c.column, normalize_value(c.value))
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    if table_type == "quarterly":
        return _repair_quarterly_labels(deduped)
    return deduped


def merge_pages(tables: Sequence[Sequence[ExtractedCell]]) -> list[ExtractedCell]:
    """Merge per-page cell lists.

    TODO: when a production caller supplies explicit table/page identities,
    add continuation-header detection and row-fragment stitching here.
    Current behavior is an order-preserving de-duplication suitable for callers
    that have already segmented one logical table across pages.
    """
    out: list[ExtractedCell] = []
    seen: set[tuple[str, str, str, str]] = set()
    for page in tables:
        for c in page:
            key = (normalize_label(c.row), c.column, normalize_value(c.value), c.section)
            if key not in seen:
                seen.add(key)
                out.append(c)
    return out


# ---------------------------
# Conservative text fallback
# ---------------------------

def _clean_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.replace("\r", "\n").split("\n") if ln.strip()]


def _is_number_line(line: str) -> bool:
    return bool(_FULL_NUM_RE.fullmatch(line.replace(" ", "")))


def _numbers_in_line(line: str) -> list[str]:
    return [m.group(0) for m in _NUM_RE.finditer(line)]


def _coerce_external_cells(cells: Iterable[object]) -> list[ExtractedCell]:
    out: list[ExtractedCell] = []
    for c in cells:
        out.append(
            ExtractedCell(
                row=str(c.row),
                column=str(c.column),
                value=str(c.value),
                section=str(getattr(c, "section", "") or ""),
            )
        )
    return out


def _find_label_segments(text: str, labels: Sequence[str]) -> list[tuple[int, int, str]]:
    """Find canonical labels in raw text while allowing PDF line breaks inside."""
    found: list[tuple[int, int, str]] = []
    for label in labels:
        pattern = re.compile(r"\s*".join(re.escape(ch) for ch in label))
        for m in pattern.finditer(text):
            found.append((m.start(), m.end(), label))
    # Longest label wins at identical starts; discard overlaps from shorter aliases.
    found.sort(key=lambda x: (x[0], -(x[1]-x[0])))
    chosen: list[tuple[int, int, str]] = []
    for f in found:
        if chosen and f[0] < chosen[-1][1]:
            continue
        chosen.append(f)
    return chosen


def _fallback_quarterly(text: str) -> list[ExtractedCell]:
    labels = (
        "归属于上市公司股东的扣除非经常性损益后的净利润",
        "归属于上市公司股东的净利润",
        "经营活动产生的现金流量净额",
        "营业收入",
    )
    positions = _find_label_segments(text, labels)
    out: list[ExtractedCell] = []
    for i, (start, end, label) in enumerate(positions):
        stop = positions[i+1][0] if i + 1 < len(positions) else len(text)
        segment = text[end:stop]
        vals = [normalize_value(v.rstrip("%")) for v in _numbers_in_line(segment)]
        # Require exactly the row's quartet before the next known row label.
        # This intentionally refuses to guess when linearization moved values
        # before their label (the coordinate path is designed to repair that).
        if len(vals) < 4:
            continue
        # Ignore quarter-month ranges and other small integers by taking decimal
        # monetary values first.
        decimals = [v for v in vals if "." in v and abs(float(v.replace(",", ""))) >= 1000]
        if len(decimals) >= 4:
            vals = decimals[:4]
        else:
            vals = vals[:4]
        for col, val in zip(_EXPECTED_COLUMNS["quarterly"], vals):
            out.append(ExtractedCell(label, col, val))
    return out


def _fallback_note_cost(text: str) -> list[ExtractedCell]:
    lines = _clean_lines(text)
    out: list[ExtractedCell] = []
    i = 0
    columns = _EXPECTED_COLUMNS["note_cost"]
    while i < len(lines):
        line = lines[i]
        if _is_number_line(line) or _is_headerish(line, "note_cost"):
            i += 1
            continue
        # Regulatory rows are a short label followed by four monetary values;
        # values can share a physical line in the linearized text.
        if len(normalize_label(line)) > 20 or line.startswith(("注", "(", "√", "□", "单位", "币种")):
            i += 1
            continue
        nums: list[str] = []
        j = i + 1
        while j < len(lines) and len(nums) < 4:
            if not _is_number_line(lines[j]) and not _numbers_in_line(lines[j]):
                break
            nums.extend(_numbers_in_line(lines[j]))
            j += 1
        monetary = [normalize_value(n.rstrip("%")) for n in nums if "." in n]
        if len(monetary) >= 4:
            for col, val in zip(columns, monetary[:4]):
                out.append(ExtractedCell(normalize_label(line), col, val))
            i = j
        else:
            i += 1
    return out


def _segment_sections(text: str) -> list[tuple[str, str]]:
    hits: list[tuple[int, str, re.Match[str]]] = []
    for name, pat in _SECTION_PATTERNS:
        for m in pat.finditer(text):
            hits.append((m.start(), name, m))
    if not hits:
        return [("", text)]
    hits.sort(key=lambda x: x[0])
    out: list[tuple[str, str]] = []
    for i, (pos, name, m) in enumerate(hits):
        end = hits[i+1][0] if i + 1 < len(hits) else len(text)
        out.append((name, text[m.end():end]))
    return out


def _fallback_segment(text: str) -> list[ExtractedCell]:
    out: list[ExtractedCell] = []
    for section, body in _segment_sections(text):
        lines = _clean_lines(body)
        pending: list[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            nums_here = _numbers_in_line(line)
            if _is_number_line(line) or (nums_here and normalize_label(line) == normalize_label(nums_here[0])):
                # A data row starts when at least three numeric values are
                # available before the next prose label.
                nums: list[str] = []
                j = i
                while j < len(lines) and len(nums) < 6:
                    lj = lines[j]
                    if _is_number_line(lj):
                        nums.extend(_numbers_in_line(lj))
                        j += 1
                        continue
                    # Stop on growth prose or the next label.
                    break
                label_parts = [p for p in pending[-2:] if not _is_headerish(p, "segment")]
                label = normalize_label("".join(label_parts))
                if label and len(nums) >= 3:
                    for col, raw in zip(_EXPECTED_COLUMNS["segment"], nums[:3]):
                        out.append(ExtractedCell(label, col, normalize_value(raw.rstrip("%")), section=section))
                pending = []
                i = j
                continue

            s = normalize_label(line)
            if (
                not _is_headerish(line, "segment")
                and not s.startswith(("增加", "减少", "个百分点", "√", "□", "单位", "币种", "(", "（"))
                and not _numbers_in_line(line)
                and len(s) <= 24
            ):
                pending.append(line)
            elif s.startswith(("增加", "减少", "个百分点")):
                pending = []
            i += 1
    return out


def _fallback_annual(text: str) -> list[ExtractedCell]:
    positions = _find_label_segments(text, _STANDARD_FINANCE_LABELS)
    out: list[ExtractedCell] = []
    for i, (start, end, label) in enumerate(positions):
        stop = positions[i+1][0] if i + 1 < len(positions) else len(text)
        segment = text[end:stop]
        vals = [normalize_value(v.rstrip("%")) for v in _numbers_in_line(segment)]
        # Monetary values have decimals; the regulatory YoY percentage is
        # interleaved between 2023 and 2022 and is much smaller.
        decimals = [v for v in vals if "." in v]
        if len(decimals) >= 4:
            chosen = (decimals[0], decimals[1], decimals[-1])
        elif len(decimals) >= 3:
            chosen = decimals[:3]
        else:
            continue
        for col, val in zip(_EXPECTED_COLUMNS["annual_data"], chosen):
            out.append(ExtractedCell(label, col, val))
    return out


def _fallback_concentration(text: str) -> list[ExtractedCell]:
    compact = normalize_label(text)
    out: list[ExtractedCell] = []
    patterns = (
        (
            "前五名客户",
            "销售额(万元)",
            "销售占比(%)",
            re.compile(r"前五名客户销售额([-+−\d,.]+)万元[,，]?占年度销售总额([-+−\d,.]+)%"),
        ),
        (
            "前五名供应商",
            "采购额(万元)",
            "采购占比(%)",
            re.compile(r"前五名供应商采购额([-+−\d,.]+)万元[,，]?占年度采购总额([-+−\d,.]+)%"),
        ),
    )
    for row, amount_col, pct_col, pat in patterns:
        m = pat.search(compact)
        if not m:
            continue
        out.append(ExtractedCell(row, amount_col, normalize_value(m.group(1))))
        out.append(ExtractedCell(row, pct_col, normalize_value(m.group(2))))
    return out


def _builtin_extract_cells(text: str, table_type: TableType) -> list[ExtractedCell]:
    if table_type == "quarterly":
        return _fallback_quarterly(text)
    if table_type == "note_cost":
        return _fallback_note_cost(text)
    if table_type == "segment":
        return _fallback_segment(text)
    if table_type == "annual_data":
        return _fallback_annual(text)
    if table_type == "concentration":
        return _fallback_concentration(text)
    return []


def extract_cells(text: str, table_type: TableType) -> list[ExtractedCell]:
    """Compatibility entry point for the existing text extractor.

    A project-local ``table_extraction.extract_cells`` wins when importable;
    standalone use falls back to the conservative built-in implementation.
    """
    if _legacy_extract_cells is not None:
        try:
            return _coerce_external_cells(_legacy_extract_cells(text, table_type))
        except (AttributeError, TypeError, ValueError):
            return _builtin_extract_cells(text, table_type)
    return _builtin_extract_cells(text, table_type)
