# Copyright (C) 2026 Microsoft Corporation.
# Licensed under the MIT License

"""Tests for the community report extractor completion contract.

The community report extraction call must work on backends that only accept the
flat subset of JSON Schema for guided decoding. Those backends reject any
schema carrying top level ``$defs``, which the nested ``CommunityReportResponse``
model always produces, so the call must not send a schema at all and instead
parse the JSON body on the client.
"""

import json
from typing import Any, cast

import pytest
from graphrag.index.operations.summarize_communities.community_reports_extractor import (
    CommunityReportResponse,
    CommunityReportsExtractor,
)
from graphrag.index.operations.summarize_communities.summarize_communities import (
    run_extractor,
)
from graphrag_llm.utils import create_completion_response

VALID_REPORT = {
    "title": "Report title",
    "summary": "Executive summary",
    "findings": [
        {"summary": "First insight", "explanation": "First explanation"},
        {"summary": "Second insight", "explanation": "Second explanation"},
    ],
    "rating": 7.5,
    "rating_explanation": "Rating explanation",
}

EXTRACTION_PROMPT = (
    "Summarize this community:\n{input_text}\nLimit: {max_report_length}"
)


class _RecordingCompletion:
    """Replay one completion body and record every request sent to the model."""

    def __init__(self, body: str) -> None:
        self._body = body
        self.requests: list[dict[str, Any]] = []

    async def completion_async(self, **kwargs: Any) -> Any:
        """Record the request and return a completion response with the body."""
        self.requests.append(dict(kwargs))
        return create_completion_response(self._body)


def _extractor(body: str) -> tuple[CommunityReportsExtractor, _RecordingCompletion]:
    """Build an extractor bound to a recording completion."""
    recorder = _RecordingCompletion(body)
    extractor = CommunityReportsExtractor(
        cast("Any", recorder),
        extraction_prompt=EXTRACTION_PROMPT,
        max_report_length=500,
    )
    return extractor, recorder


def _error_tracker() -> tuple[list[BaseException], Any]:
    """Collect errors routed to the extractor error handler."""
    errors: list[BaseException] = []

    def on_error(error: BaseException, _stack: str, _data: Any) -> None:
        errors.append(error)

    return errors, on_error


def _assert_flat_schema_backend_request(request: dict[str, Any]) -> None:
    """Fail when a request carries a schema that a flat schema backend rejects."""
    assert "response_format" not in request or request["response_format"] is None, (
        "community report extraction must not send a response_format schema, "
        f"got {request.get('response_format')!r}"
    )


async def test_request_carries_no_response_format_schema() -> None:
    extractor, recorder = _extractor(json.dumps(VALID_REPORT))

    await extractor("<community context>")

    assert len(recorder.requests) == 1
    _assert_flat_schema_backend_request(recorder.requests[0])


def test_nested_schema_would_have_carried_defs() -> None:
    """Document why the schema cannot be sent to a flat schema backend."""
    schema = CommunityReportResponse.model_json_schema()

    assert "$defs" in schema
    assert schema["properties"]["findings"]["items"] == {"$ref": "#/$defs/FindingModel"}


async def test_parses_report_wrapped_in_json_code_fence() -> None:
    body = f"```json\n{json.dumps(VALID_REPORT, indent=2)}\n```"
    extractor, _ = _extractor(body)

    result = await extractor("<community context>")

    assert result.structured_output is not None
    assert result.structured_output.title == "Report title"
    assert [f.summary for f in result.structured_output.findings] == [
        "First insight",
        "Second insight",
    ]
    assert result.output.startswith("# Report title")
    assert "## First insight\n\nFirst explanation" in result.output


async def test_parses_report_wrapped_in_bare_code_fence() -> None:
    body = f"```\n{json.dumps(VALID_REPORT)}\n```"
    extractor, _ = _extractor(body)

    result = await extractor("<community context>")

    assert result.structured_output is not None
    assert result.structured_output.rating == pytest.approx(7.5)


async def test_parses_report_with_leading_and_trailing_prose() -> None:
    body = (
        "Here is the report you asked for:\n\n"
        f"{json.dumps(VALID_REPORT)}\n\n"
        "Let me know if you need any changes."
    )
    extractor, _ = _extractor(body)

    result = await extractor("<community context>")

    assert result.structured_output is not None
    assert result.structured_output.summary == "Executive summary"


async def test_parses_report_returned_as_bare_json() -> None:
    extractor, _ = _extractor(json.dumps(VALID_REPORT))

    result = await extractor("<community context>")

    assert result.structured_output is not None
    assert result.structured_output.rating_explanation == "Rating explanation"


async def test_unparseable_body_reports_error_and_returns_no_report() -> None:
    errors, on_error = _error_tracker()
    recorder = _RecordingCompletion("I could not produce a report for this community.")
    extractor = CommunityReportsExtractor(
        cast("Any", recorder),
        extraction_prompt=EXTRACTION_PROMPT,
        max_report_length=500,
        on_error=on_error,
    )

    result = await extractor("<community context>")

    assert result.structured_output is None
    assert result.output == ""
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)


async def test_body_missing_required_fields_reports_error() -> None:
    errors, on_error = _error_tracker()
    recorder = _RecordingCompletion(json.dumps({"title": "Only a title"}))
    extractor = CommunityReportsExtractor(
        cast("Any", recorder),
        extraction_prompt=EXTRACTION_PROMPT,
        max_report_length=500,
        on_error=on_error,
    )

    result = await extractor("<community context>")

    assert result.structured_output is None
    assert len(errors) == 1


async def test_run_extractor_maps_parsed_report_to_community_report() -> None:
    body = f"```json\n{json.dumps(VALID_REPORT)}\n```"
    recorder = _RecordingCompletion(body)

    report = await run_extractor(
        12,
        "<community context>",
        1,
        cast("Any", recorder),
        EXTRACTION_PROMPT,
        500,
    )

    assert report is not None
    assert report["community"] == 12
    assert report["level"] == 1
    assert report["rank"] == pytest.approx(7.5)
    assert [f["explanation"] for f in report["findings"]] == [
        "First explanation",
        "Second explanation",
    ]
    assert report["full_content"].startswith("# Report title")
    _assert_flat_schema_backend_request(recorder.requests[0])
