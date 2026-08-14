import hashlib
import html
import json
import re
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from urllib.parse import urljoin

import httpx
from pydantic import BaseModel, HttpUrl

CNINFO_ORIGIN = "https://www.cninfo.com.cn"
CNINFO_QUERY_URL = f"{CNINFO_ORIGIN}/new/hisAnnouncement/query"
CNINFO_DOWNLOAD_BASE = "https://static.cninfo.com.cn/"
HTML_TAG = re.compile(r"<[^>]+>")


class Announcement(BaseModel):
    announcement_id: str
    security_code: str
    security_name: str
    organization_id: str
    title: str
    published_at: datetime
    file_path: str
    file_type: str

    @property
    def download_url(self) -> str:
        path = PurePosixPath(self.file_path)
        if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".pdf":
            raise ValueError(f"Unsafe CNInfo attachment path: {self.file_path}")
        return urljoin(CNINFO_DOWNLOAD_BASE, self.file_path)


class FilingArtifact(BaseModel):
    source: str = "cninfo"
    announcement: Announcement
    source_url: HttpUrl
    local_file: str
    sha256: str
    size_bytes: int
    downloaded_at: datetime


def _plain_text(value: str) -> str:
    return html.unescape(HTML_TAG.sub("", value)).strip()


def _parse_announcement(payload: dict) -> Announcement:
    return Announcement(
        announcement_id=str(payload["announcementId"]),
        security_code=str(payload["secCode"]),
        security_name=_plain_text(payload["secName"]),
        organization_id=str(payload["orgId"]),
        title=_plain_text(payload["announcementTitle"]),
        published_at=datetime.fromtimestamp(payload["announcementTime"] / 1000, tz=UTC),
        file_path=str(payload["adjunctUrl"]),
        file_type=str(payload["adjunctType"]),
    )


def select_chinese_annual_report(
    announcements: list[Announcement], company: str, report_year: int
) -> Announcement:
    expected_title = f"{company}{report_year}年年度报告"
    matches = [item for item in announcements if item.title == expected_title]
    if len(matches) > 1:
        raise LookupError(f"Multiple exact annual reports found for {company} {report_year}")
    if matches:
        return matches[0]

    # CNInfo exposes the exchange security short name separately from the
    # announcement title.  Some issuers use that short name in the title
    # ("贵州茅台"), while others use their full legal name
    # ("内蒙古伊利实业集团股份有限公司").  The search command is documented in
    # terms of the short name, so accept the unique full Chinese annual report
    # belonging to the matching security.  Keep the exact suffix deliberately:
    # summaries, English editions and correction notices must not be selected.
    annual_report_suffix = f"{report_year}年年度报告"
    security_matches = [
        item
        for item in announcements
        if item.security_name == company and item.title.endswith(annual_report_suffix)
    ]
    if len(security_matches) == 1:
        return security_matches[0]
    if len(security_matches) > 1:
        raise LookupError(
            f"Multiple Chinese annual reports found for security {company!r} {report_year}"
        )

    available = ", ".join(item.title for item in announcements[:10]) or "none"
    raise LookupError(
        f"No Chinese annual report for security {company!r} in {report_year}; "
        f"expected {expected_title!r}; found: {available}"
    )


class CninfoClient:
    def __init__(self, client: httpx.Client | None = None) -> None:
        self.client = client or httpx.Client(
            timeout=30,
            follow_redirects=True,
            headers={
                "User-Agent": "FinDocRAG/0.1 (+https://github.com/yiwu17/findoc-rag)",
                "Origin": CNINFO_ORIGIN,
                "Referer": f"{CNINFO_ORIGIN}/",
            },
        )

    def search_annual_reports(self, company: str, report_year: int) -> list[Announcement]:
        publication_year = report_year + 1
        response = self.client.post(
            CNINFO_QUERY_URL,
            data={
                "pageNum": "1",
                "pageSize": "30",
                "column": "szse",
                "tabName": "fulltext",
                "plate": "",
                "stock": "",
                "searchkey": company,
                "secid": "",
                "category": "category_ndbg_szsh",
                "trade": "",
                "seDate": f"{publication_year}-01-01~{publication_year}-12-31",
                "sortName": "",
                "sortType": "",
                "isHLtitle": "true",
            },
        )
        response.raise_for_status()
        payload = response.json()
        return [_parse_announcement(item) for item in payload.get("announcements") or []]

    def download(self, announcement: Announcement, destination: Path) -> FilingArtifact:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.unlink(missing_ok=True)
        hasher = hashlib.sha256()
        size = 0
        with self.client.stream("GET", announcement.download_url) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if "pdf" not in content_type and "octet-stream" not in content_type:
                raise ValueError(f"Unexpected attachment content type: {content_type}")
            with temporary.open("wb") as target:
                for chunk in response.iter_bytes():
                    target.write(chunk)
                    hasher.update(chunk)
                    size += len(chunk)

        if size < 5 or temporary.read_bytes()[:5] != b"%PDF-":
            temporary.unlink(missing_ok=True)
            raise ValueError("Downloaded attachment is not a valid PDF")
        temporary.replace(destination)

        return FilingArtifact(
            announcement=announcement,
            source_url=announcement.download_url,
            local_file=destination.as_posix(),
            sha256=hasher.hexdigest(),
            size_bytes=size,
            downloaded_at=datetime.now(UTC),
        )


def write_artifact_manifest(artifact: FilingArtifact, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
