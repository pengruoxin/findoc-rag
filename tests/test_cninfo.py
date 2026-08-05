from datetime import UTC, datetime

import pytest

from findoc_rag.sources.cninfo import Announcement, select_chinese_annual_report


def announcement(title: str) -> Announcement:
    return Announcement(
        announcement_id="1222993920",
        security_code="600519",
        security_name="贵州茅台",
        organization_id="gssh0600519",
        title=title,
        published_at=datetime(2025, 4, 3, tzinfo=UTC),
        file_path="finalpage/2025-04-03/1222993920.PDF",
        file_type="PDF",
    )


def test_selects_exact_chinese_full_annual_report() -> None:
    candidates = [
        announcement("贵州茅台2024年年度报告（英文版）"),
        announcement("贵州茅台2024年年度报告摘要"),
        announcement("贵州茅台2024年年度报告"),
    ]

    selected = select_chinese_annual_report(candidates, "贵州茅台", 2024)

    assert selected.title == "贵州茅台2024年年度报告"


def test_rejects_unsafe_attachment_path() -> None:
    unsafe = announcement("贵州茅台2024年年度报告").model_copy(
        update={"file_path": "../secret.PDF"}
    )

    with pytest.raises(ValueError, match="Unsafe"):
        _ = unsafe.download_url

