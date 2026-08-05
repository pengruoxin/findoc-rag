import json
from pathlib import Path

from findoc_rag.datasets.financebench import convert_financebench


def test_convert_financebench_deduplicates_evidence_pages(tmp_path: Path) -> None:
    evidence = {
        "doc_name": "Example_2024_10K",
        "evidence_page_num": 7,
        "evidence_text_full_page": "Revenue was 100 million.",
    }
    records = [
        {
            "financebench_id": f"q_{index}",
            "company": "Example",
            "doc_name": "Example_2024_10K",
            "question_type": "metrics-generated",
            "question": f"Question {index}?",
            "answer": "100",
            "justification": "Direct extraction.",
            "evidence": [evidence],
        }
        for index in range(2)
    ]
    source = tmp_path / "sample.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")

    corpus, questions = convert_financebench(source)

    assert len(corpus) == 1
    assert len(questions) == 2
    assert questions[0].gold_document_ids == questions[1].gold_document_ids
    assert corpus[0].page_number == 7
