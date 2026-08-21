from findoc_rag.pdf_candidate_discovery import build_page_signals, classify_page


def _kinds(
    text: str,
    *,
    drawings: int = 0,
    images: int = 0,
    coverage: float = 0.0,
    spans: list[str] | None = None,
    layout_items: list[tuple[str, float, float]] | None = None,
):
    effective_spans = spans or ["1,000.00"] * 12
    column_count = 3 if spans is None else 2
    span_items = layout_items or [
        (
            value,
            100.0 + (index % column_count) * 100.0,
            100.0 + (index // column_count) * 20.0,
        )
        for index, value in enumerate(effective_spans)
    ]
    signals = build_page_signals(
        page_number=1,
        text=text,
        span_texts=effective_spans,
        span_items=span_items,
        drawing_count=drawings,
        image_count=images,
        image_coverage_max=coverage,
        rotation=0,
    )
    return {candidate.kind for candidate in classify_page(signals)}


def test_dense_table_page_can_be_native_and_borderless_candidate() -> None:
    text = (
        "单位：元 项目 期初余额 期末余额 合计 "
        + "营业收入 1,000.00 2,000.00 合计 3,000.00 " * 20
    )

    kinds = _kinds(text, drawings=2)

    assert "native_control" in kinds
    assert "borderless_table" in kinds


def test_hierarchical_and_continuation_markers_are_only_candidate_signals() -> None:
    text = (
        "续表 单位：元 项目 本期发生额 上期发生额 调整前 调整后 合计 "
        + "主营业务 1,000.00 900.00 800.00 " * 8
    )

    financial_items = [
        ("1,000.00", 100.0 + (index % 3) * 100.0, 100.0 + (index // 3) * 20.0)
        for index in range(12)
    ]
    header_items = [
        ("本期发生额", 150.0, 20.0),
        ("收入", 100.0, 40.0),
        ("成本", 200.0, 40.0),
        ("上期发生额", 450.0, 20.0),
        ("收入", 400.0, 40.0),
        ("成本", 500.0, 40.0),
    ]
    kinds = _kinds(text, drawings=20, layout_items=header_items + financial_items)

    assert "merged_hierarchical_header" in kinds
    assert "cross_page_continuation" in kinds
    assert "borderless_table" not in kinds


def test_image_and_table_text_marks_mixed_layout_candidate() -> None:
    text = "单位：元 项目 合计 " + "收入 100.00 200.00 " * 30

    kinds = _kinds(text, images=1, coverage=0.4)

    assert "rotated_or_mixed_layout" in kinds


def test_dates_in_prose_do_not_look_like_financial_table_cells() -> None:
    text = "项目合同有效期从2021年1月1日至2023年12月31日，合计续签2次。" * 20

    kinds = _kinds(
        text,
        spans=["2021", "1", "1", "2023", "12", "31"] * 10,
    )

    assert "borderless_table" not in kinds
    assert "native_control" not in kinds
