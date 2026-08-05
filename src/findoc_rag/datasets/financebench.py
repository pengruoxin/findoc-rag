import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from findoc_rag.schemas import BenchmarkQuestion, CorpusDocument


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc


def _evidence_id(doc_name: str, page_number: int, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{doc_name}:page-{page_number}:{digest}"


def convert_financebench(source: Path) -> tuple[list[CorpusDocument], list[BenchmarkQuestion]]:
    """Convert the public FinanceBench JSONL into FinDocRAG's internal schema.

    The public QA sample contains gold evidence pages, not every page in each 10-K.
    This adapter therefore creates a bootstrap corpus from all unique gold pages. It
    is useful for validating retrieval code, but must not be reported as full-document
    FinanceBench performance.
    """
    corpus_by_id: dict[str, CorpusDocument] = {}
    questions: list[BenchmarkQuestion] = []

    for record in _read_jsonl(source):
        gold_ids: list[str] = []
        for evidence in record["evidence"]:
            text = evidence["evidence_text_full_page"].strip()
            page_number = int(evidence["evidence_page_num"])
            document_id = _evidence_id(evidence["doc_name"], page_number, text)
            gold_ids.append(document_id)
            corpus_by_id.setdefault(
                document_id,
                CorpusDocument(
                    document_id=document_id,
                    source_document=evidence["doc_name"],
                    page_number=page_number,
                    text=text,
                    metadata={"company": record["company"]},
                ),
            )

        questions.append(
            BenchmarkQuestion(
                question_id=record["financebench_id"],
                question=record["question"],
                answer=record["answer"],
                gold_document_ids=list(dict.fromkeys(gold_ids)),
                question_type=record["question_type"],
                source_document=record["doc_name"],
                justification=record.get("justification") or "",
            )
        )

    return list(corpus_by_id.values()), questions


def write_jsonl(records: Iterable[CorpusDocument | BenchmarkQuestion], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        for record in records:
            if not isinstance(record, (CorpusDocument, BenchmarkQuestion)):
                raise TypeError(f"Unsupported record type: {type(record).__name__}")
            target.write(record.model_dump_json() + "\n")
