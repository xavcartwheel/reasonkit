# Unit tests for the classifier stage: parsing structured output.
from reasonkit.classifier import parse_classification
from tests.fixtures import (
    CLASSIFY_DECISION,
    CLASSIFY_DIRECT,
    CLASSIFY_CODE,
    CLASSIFY_CODE_WITH_INPUT,
    CLASSIFY_CODE_VAGUE,
)


def test_parse_decision_has_goal_assumptions_questions():
    r = parse_classification(CLASSIFY_DECISION)
    assert r["category"] == "DECISION"
    assert r["complexity"] == "COMPLEX"
    assert "profitable" in r["goal"]
    assert len(r["assumptions"]) == 3
    assert len(r["clarifying_questions"]) == 2
    assert r["needs_clarification"] is False


def test_parse_direct_has_no_answer_field():
    # The classifier is ROUTE-ONLY: it must not emit a DIRECT_ANSWER. The final
    # answer is produced by a later, dedicated call (see core._direct_pipeline).
    r = parse_classification(CLASSIFY_DIRECT)
    assert r["category"] == "DIRECT"
    assert "direct_answer" not in r
    assert r["needs_clarification"] is False


def test_parse_unknown_category_falls_back_to_direct():
    r = parse_classification("CATEGORY: WAT\nCOMPLEXITY: SIMPLE\nGOAL:\n")
    assert r["category"] == "DIRECT"


def test_parse_missing_assumptions_is_empty_list():
    r = parse_classification(
        "CATEGORY: DECISION\nCOMPLEXITY: SIMPLE\nGOAL: g\nASSUMPTIONS:\nCLARIFYING_QUESTIONS:\n"
    )
    assert r["assumptions"] == []
    assert r["clarifying_questions"] == []


def test_parse_code_category_detected():
    r = parse_classification(CLASSIFY_CODE)
    assert r["category"] == "CODE"
    assert r["code_input"] == ""
    assert r["needs_clarification"] is True


def test_parse_code_with_input_captures_code():
    r = parse_classification(CLASSIFY_CODE_WITH_INPUT)
    assert r["category"] == "CODE"
    assert "def add" in r["code_input"]
    assert r["needs_clarification"] is False


def test_parse_vague_code_flags_needs_clarification():
    # The misroute case: "generate a C++ app that does X" is CLASSIFIED as CODE
    # but the spec ("X") is undefined, so the classifier flags it. The pipeline
    # must reroute this to DECISION rather than producing a clarification stub.
    r = parse_classification(CLASSIFY_CODE_VAGUE)
    assert r["category"] == "CODE"
    assert r["needs_clarification"] is True
    assert r["code_input"] == ""


def test_parse_needs_clarification_yes_no_variants():
    assert parse_classification("NEEDS_CLARIFICATION: YES\n")["needs_clarification"] is True
    assert parse_classification("NEEDS_CLARIFICATION: NO\n")["needs_clarification"] is False
    assert parse_classification("NEEDS_CLARIFICATION: yes, the request is vague\n")["needs_clarification"] is True
