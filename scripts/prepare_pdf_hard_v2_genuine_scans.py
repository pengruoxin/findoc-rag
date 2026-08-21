"""Audit official scan sources and build the fixed genuine-scan development PDF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from findoc_rag.pdf_scan_benchmark import build_genuine_scan_candidate_pdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path(
            "data/evaluation/pdf-hard-v2/genuine-scan-source-candidates-v1.json"
        ),
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path(
            "data/evaluation/pdf-hard-v2/genuine-scan-development-candidates.pdf"
        ),
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path(
            "data/evaluation/pdf-hard-v2/genuine-scan-development-candidates.json"
        ),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path(
            "reports/pdf-extraction/pdf-hard-v2-genuine-scan-source-audit.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest, audit = build_genuine_scan_candidate_pdf(
        args.registry.resolve(strict=True), args.output_pdf
    )
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"sources_valid={audit['all_sources_valid']}")
    print(f"accepted_sources={audit['accepted_source_count']}")
    print(f"rejected_text_layer_sources={audit['rejected_text_layer_source_count']}")
    print(f"selected_pages={manifest['page_count']}")
    print(f"pdf_sha256={manifest['pdf_sha256']}")
    print(f"pdf={args.output_pdf.resolve()}")
    print(f"manifest={args.output_manifest.resolve()}")
    print(f"audit={args.audit_output.resolve()}")


if __name__ == "__main__":
    main()
