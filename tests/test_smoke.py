from typer.testing import CliRunner

from findoc_rag.cli import app


def test_doctor_command() -> None:
    result = CliRunner().invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "FinDocRAG 0.1.0 is ready" in result.stdout
