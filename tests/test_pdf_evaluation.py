from findoc_rag.documents.models import BoundingBox, DocumentElement, DocumentPage
from findoc_rag.pdf_evaluation import (
    character_error_rate,
    edit_distance,
    normalize_extraction_text,
    numeric_scores,
)
from findoc_rag.pdf_table_interpretation import (
    DeepSeekTableInterpreter,
    TableQuestion,
    score_table_fact,
    score_table_fact_pages,
    serialize_layout_page,
    serialize_layout_pages,
    table_values_equal,
)


def test_pdf_text_metrics_normalize_layout_whitespace() -> None:
    reference = "营业收入 1,234.50 元\n同比 8.2%"
    candidate = "营业收入\n1,234.50元 同比8.2%"

    assert normalize_extraction_text(reference) == normalize_extraction_text(candidate)
    assert edit_distance("收入", "收人") == 1
    assert character_error_rate(reference, candidate) == 0.0
    assert numeric_scores(reference, candidate) == (1.0, 1.0, 1.0)


def test_numeric_metrics_penalize_missing_and_spurious_values() -> None:
    precision, recall, f1 = numeric_scores("收入100，成本40", "收入100，利润60")

    assert precision == 0.5
    assert recall == 0.5
    assert f1 == 0.5


def _text_element(
    element_id: str, text: str, x0: float, y0: float, x1: float, y1: float
) -> DocumentElement:
    return DocumentElement(
        element_id=element_id,
        element_type="text",
        text=text,
        bbox=BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1),
        reading_order=int(element_id.removeprefix("e")),
        extraction_source="ocr",
        confidence=0.99,
    )


def _table_page() -> DocumentPage:
    elements = [
        _text_element("e0", "项目列示", 0, 0, 50, 10),
        _text_element("e1", "项目", 0, 20, 50, 30),
        _text_element("e2", "期末余额", 100, 20, 150, 30),
        _text_element("e3", "期初余额", 200, 20, 250, 30),
        _text_element("e4", "固定资产", 0, 40, 50, 50),
        _text_element("e5", "21,871,446,747.14", 100, 40, 150, 50),
        _text_element("e6", "19,909,280,655.97", 200, 40, 250, 50),
    ]
    return DocumentPage(
        page_number=1,
        width=300,
        height=500,
        elements=elements,
        extracted_character_count=sum(len(element.text) for element in elements),
        image_count=1,
        needs_ocr=False,
        extraction_route="full_ocr",
    )


def test_layout_table_score_requires_row_column_and_value_association() -> None:
    question = TableQuestion(
        question_id="closing",
        question="固定资产期末余额是多少？",
        expected_value="21,871,446,747.14",
        expected_unit="元",
        row_label="固定资产",
        column_label="期末余额",
        section_label="项目列示",
    )

    score = score_table_fact(_table_page(), question)

    assert score["row_value_same_row"] is True
    assert score["column_aligned"] is True
    assert score["recoverable"] is True
    assert "固定资产 | 21,871,446,747.14" in serialize_layout_page(_table_page())
    assert table_values_equal("21,871,446,747.14", "21871446747.140")


def test_layout_rows_follow_visual_orientation_on_rotated_page() -> None:
    page = DocumentPage(
        page_number=1,
        width=600,
        height=800,
        elements=[
            _text_element("e0", "货币资金", 100, 650, 115, 720),
            _text_element("e1", "105,904,442.39", 101, 300, 116, 450),
        ],
        extracted_character_count=20,
        image_count=1,
        needs_ocr=False,
        extraction_route="full_ocr",
        rotation=90,
    )

    assert "货币资金 | 105,904,442.39" in serialize_layout_page(page)


def test_hierarchical_column_header_combines_stacked_exact_fragments() -> None:
    elements = [
        _text_element("e0", "2025年期末", 200, 20, 300, 30),
        _text_element("e1", "往来资金余额", 205, 32, 295, 42),
        _text_element("e2", "海宁中国皮革城投资有限公司", 10, 60, 160, 70),
        _text_element("e3", "76,568.24", 220, 60, 285, 70),
    ]
    page = DocumentPage(
        page_number=1,
        width=400,
        height=500,
        elements=elements,
        extracted_character_count=sum(len(element.text) for element in elements),
        image_count=1,
        needs_ocr=False,
        extraction_route="full_ocr",
    )
    question = TableQuestion(
        question_id="closing",
        question="期末余额是多少？",
        expected_value="76,568.24",
        row_label="海宁中国皮革城投资有限公司",
        column_label="2025年期末往来资金余额",
    )

    score = score_table_fact(page, question)

    assert score["row_value_same_row"] is True
    assert score["column_header_found"] is True
    assert score["column_aligned"] is True
    assert score["recoverable"] is True


def test_financial_period_label_match_allows_one_ocr_character_error() -> None:
    elements = [
        _text_element("e0", "股本", 200, 20, 260, 30),
        _text_element("e1", "、1年期末余额", 10, 60, 150, 70),
        _text_element("e2", "381,512,820.00", 200, 60, 280, 70),
    ]
    page = DocumentPage(
        page_number=1,
        width=400,
        height=500,
        elements=elements,
        extracted_character_count=sum(len(element.text) for element in elements),
        image_count=1,
        needs_ocr=False,
        extraction_route="full_ocr",
    )
    question = TableQuestion(
        question_id="prior-close",
        question="上年年末余额的股本是多少？",
        expected_value="381,512,820.00",
        row_label="上年年末余额",
        column_label="股本",
    )

    score = score_table_fact(page, question)

    assert score["row_found"] is True
    assert score["recoverable"] is True


def test_financial_period_label_match_does_not_swap_current_and_prior_year() -> None:
    elements = [
        _text_element("e0", "股本", 200, 20, 260, 30),
        _text_element("e1", "本年期末余额", 10, 60, 150, 70),
        _text_element("e2", "381,512,820.00", 200, 60, 280, 70),
    ]
    page = DocumentPage(
        page_number=1,
        width=400,
        height=500,
        elements=elements,
        extracted_character_count=sum(len(element.text) for element in elements),
        image_count=1,
        needs_ocr=False,
        extraction_route="full_ocr",
    )
    question = TableQuestion(
        question_id="prior-close",
        question="上年年末余额的股本是多少？",
        expected_value="381,512,820.00",
        row_label="上年年末余额",
        column_label="股本",
    )

    score = score_table_fact(page, question)

    assert score["row_found"] is False
    assert score["recoverable"] is False


def test_hierarchical_header_does_not_join_unrelated_columns() -> None:
    elements = [
        _text_element("e0", "2025年期末", 200, 20, 300, 30),
        _text_element("e1", "往来资金余额", 20, 32, 110, 42),
        _text_element("e2", "公司", 10, 60, 100, 70),
        _text_element("e3", "76,568.24", 220, 60, 285, 70),
    ]
    page = DocumentPage(
        page_number=1,
        width=400,
        height=500,
        elements=elements,
        extracted_character_count=sum(len(element.text) for element in elements),
        image_count=1,
        needs_ocr=False,
    )
    question = TableQuestion(
        question_id="closing",
        question="期末余额是多少？",
        expected_value="76,568.24",
        row_label="公司",
        column_label="2025年期末往来资金余额",
    )

    score = score_table_fact(page, question)

    assert score["column_header_found"] is False
    assert score["recoverable"] is False


def test_wrapped_row_label_joins_only_nearby_left_fragments() -> None:
    elements = [
        _text_element("e0", "2025年期初", 200, 20, 300, 30),
        _text_element("e1", "往来资金余额", 205, 32, 295, 42),
        _text_element("e2", "海宁皮革时尚小镇投资开发", 10, 60, 170, 70),
        _text_element("e3", "144,133.68", 220, 60, 285, 70),
        _text_element("e4", "有限公司", 10, 72, 65, 82),
    ]
    page = DocumentPage(
        page_number=1,
        width=400,
        height=500,
        elements=elements,
        extracted_character_count=sum(len(element.text) for element in elements),
        image_count=1,
        needs_ocr=False,
        extraction_route="full_ocr",
    )
    question = TableQuestion(
        question_id="opening",
        question="期初余额是多少？",
        expected_value="144,133.68",
        row_label="海宁皮革时尚小镇投资开发有限公司",
        column_label="2025年期初往来资金余额",
    )

    score = score_table_fact(page, question)

    assert score["row_found"] is True
    assert score["row_value_same_row"] is True
    assert score["column_aligned"] is True
    assert score["recoverable"] is True


def test_cross_page_table_score_carries_section_into_repeated_header() -> None:
    section_page = DocumentPage(
        page_number=1,
        width=300,
        height=500,
        elements=[_text_element("e0", "二、累计折旧", 0, 450, 80, 460)],
        extracted_character_count=6,
        image_count=0,
        needs_ocr=False,
    )
    target_elements = [
        _text_element("e0", "项目", 0, 20, 50, 30),
        _text_element("e1", "其他", 100, 20, 150, 30),
        _text_element("e2", "合计", 200, 20, 250, 30),
        _text_element("e3", "5.期末余额", 0, 40, 70, 50),
        _text_element("e4", "309,265,284.63", 100, 40, 150, 50),
        _text_element("e5", "3,316,847,042.37", 200, 40, 250, 50),
    ]
    target_page = DocumentPage(
        page_number=2,
        width=300,
        height=500,
        elements=target_elements,
        extracted_character_count=sum(len(element.text) for element in target_elements),
        image_count=0,
        needs_ocr=False,
    )
    question = TableQuestion(
        question_id="depreciation-total",
        question="累计折旧期末余额合计是多少？",
        expected_value="3,316,847,042.37",
        expected_unit="元",
        row_label="5.期末余额",
        column_label="合计",
        section_label="二、累计折旧",
        target_page_offset=1,
        requires_previous_page_context=True,
    )

    score = score_table_fact_pages([section_page, target_page], question)

    assert score["section_active"] is True
    assert score["section_carried_from_previous_page"] is True
    assert score["section_carryover_correct"] is True
    assert score["recoverable"] is True
    assert "[PAGE 2]" in serialize_layout_pages([section_page, target_page])


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"answers":[{"question_id":"closing",'
                            '"status":"answered","value":"21,871,446,747.14",'
                            '"unit":"元","row_label":"固定资产",'
                            '"column_label":"期末余额","section_label":"项目列示",'
                            '"evidence":"L003"}]}'
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 30},
        }


class _FakeClient:
    def __init__(self) -> None:
        self.payload: dict | None = None

    def post(self, *args: object, **kwargs: object) -> _FakeResponse:
        self.payload = kwargs["json"]  # type: ignore[assignment]
        return _FakeResponse()


class _NullAbstentionResponse(_FakeResponse):
    def json(self) -> dict:
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"answers":[{"question_id":"closing",'
                            '"status":"insufficient_evidence","value":null,'
                            '"unit":null,"row_label":null,"column_label":null,'
                            '"section_label":null,"evidence":null}]}'
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 80, "completion_tokens": 20},
        }


class _NullAbstentionClient(_FakeClient):
    def post(self, *args: object, **kwargs: object) -> _NullAbstentionResponse:
        self.payload = kwargs["json"]  # type: ignore[assignment]
        return _NullAbstentionResponse()


def test_deepseek_table_interpreter_batches_questions_without_leaking_gold() -> None:
    client = _FakeClient()
    interpreter = DeepSeekTableInterpreter(
        model="deepseek-test",
        api_key="test-key",
        client=client,  # type: ignore[arg-type]
    )
    question = TableQuestion(
        question_id="closing",
        question="固定资产期末余额是多少？",
        expected_value="secret-gold-value",
        expected_unit="元",
        row_label="固定资产",
        column_label="期末余额",
        section_label="项目列示",
    )

    result = interpreter.interpret_page([question], "L001: 固定资产 | 21,871,446,747.14")

    assert result.answers[0].value == "21,871,446,747.14"
    assert result.input_tokens == 100
    assert client.payload is not None
    user_prompt = client.payload["messages"][1]["content"]
    questions_only = user_prompt.split("页面证据", maxsplit=1)[0]
    assert "secret-gold-value" not in questions_only


def test_deepseek_table_interpreter_normalizes_null_abstention_fields() -> None:
    interpreter = DeepSeekTableInterpreter(
        model="deepseek-test",
        api_key="test-key",
        client=_NullAbstentionClient(),  # type: ignore[arg-type]
    )
    question = TableQuestion(
        question_id="closing",
        question="固定资产期末余额是多少？",
        expected_value="21,871,446,747.14",
        row_label="固定资产",
        column_label="期末余额",
    )

    result = interpreter.interpret_page([question], "L001: OCR 无法确认")

    assert result.answers[0].status == "insufficient_evidence"
    assert result.answers[0].value == ""
    assert result.answers[0].evidence == ""
