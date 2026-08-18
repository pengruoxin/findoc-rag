import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel


def read_jsonl[ModelT: BaseModel](path: Path, model: type[ModelT]) -> list[ModelT]:
    records: list[ModelT] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                records.append(model.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"Invalid record on line {line_number} of {path}") from exc
    return records


def write_text_lf(path: Path, content: str) -> None:
    """Write text without platform newline translation.

    Several integrity checks hash a string in memory and later re-hash the file
    from disk. On Windows the default ``newline=None`` rewrites every ``\\n`` as
    ``\\r\\n``, so those two digests can never agree. Every hashed payload must
    go through here.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="")


def write_dict_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(data: dict, path: Path) -> None:
    write_text_lf(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
