from scripts.evaluate_pdf_hard_v2_deepseek_fallback import (
    score_fallback_answers,
    unresolved_cell_predictions,
)


def test_high_risk_gate_selects_only_unrecoverable_cells() -> None:
    report = {
        "predictions": [
            {"probe_id": "title", "probe_type": "text", "success": False},
            {
                "probe_id": "safe",
                "probe_type": "row_value",
                "structured_cell_recoverable": True,
            },
            {
                "probe_id": "risk",
                "probe_type": "row_value",
                "structured_cell_recoverable": False,
            },
        ]
    }

    assert [item["probe_id"] for item in unresolved_cell_predictions(report)] == [
        "risk"
    ]


def test_model_candidate_is_never_auto_accepted_without_local_structure_proof() -> None:
    predictions = [
        {
            "probe_id": "risk",
            "row_label": "本年年末余额",
            "column_label": "股东权益合计",
            "expected_value": "482,110,062.36",
        }
    ]
    answers = [
        {
            "question_id": "risk",
            "status": "answered",
            "value": "482110062.36",
        }
    ]

    result = score_fallback_answers(
        predictions, answers, "四、本期期末余额 | 482,110,062.36"
    )[0]

    assert result["value_correct"] is True
    assert result["evidence_value_supported"] is True
    assert result["decision"] == "manual_review"
    assert result["auto_accepted"] is False
