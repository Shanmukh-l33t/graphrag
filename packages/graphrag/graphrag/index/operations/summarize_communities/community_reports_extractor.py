# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing 'CommunityReportsResult' and 'CommunityReportsExtractor' models."""

import json
import logging
import re
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from graphrag.index.typing.error_handler import ErrorHandlerFn

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion
    from graphrag_llm.types import LLMCompletionResponse

logger = logging.getLogger(__name__)

# these tokens are used in the prompt
INPUT_TEXT_KEY = "input_text"
MAX_LENGTH_KEY = "max_report_length"

# a fenced code block, optionally tagged json, wrapping the report body
_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class FindingModel(BaseModel):
    """A model for the expected LLM response shape."""

    summary: str = Field(description="The summary of the finding.")
    explanation: str = Field(description="An explanation of the finding.")


class CommunityReportResponse(BaseModel):
    """A model for the expected LLM response shape."""

    title: str = Field(description="The title of the report.")
    summary: str = Field(description="A summary of the report.")
    findings: list[FindingModel] = Field(
        description="A list of findings in the report."
    )
    rating: float = Field(description="The rating of the report.")
    rating_explanation: str = Field(description="An explanation of the rating.")


@dataclass
class CommunityReportsResult:
    """Community reports result class definition."""

    output: str
    structured_output: CommunityReportResponse | None


def _parse_json_response(content: str) -> dict[str, Any]:
    """Parse the report object out of a completion body.

    The body may be bare JSON, JSON inside a fenced code block, or JSON surrounded
    by prose. Raises ValueError when no JSON object can be decoded.
    """
    text = content.strip()
    candidates = [text]

    fenced = _FENCE_PATTERN.search(text)
    if fenced:
        candidates.append(fenced.group(1))

    first_brace, last_brace = text.find("{"), text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(text[first_brace : last_brace + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    msg = "Completion content did not contain a JSON object."
    raise ValueError(msg)


class CommunityReportsExtractor:
    """Community reports extractor class definition."""

    _model: "LLMCompletion"
    _extraction_prompt: str
    _output_formatter_prompt: str
    _on_error: ErrorHandlerFn
    _max_report_length: int

    def __init__(
        self,
        model: "LLMCompletion",
        extraction_prompt: str,
        max_report_length: int,
        on_error: ErrorHandlerFn | None = None,
    ):
        """Init method definition."""
        self._model = model
        self._extraction_prompt = extraction_prompt
        self._on_error = on_error or (lambda _e, _s, _d: None)
        self._max_report_length = max_report_length

    async def __call__(self, input_text: str):
        """Call method definition."""
        output = None
        try:
            prompt = self._extraction_prompt.format(**{
                INPUT_TEXT_KEY: input_text,
                MAX_LENGTH_KEY: str(self._max_report_length),
            })
            # No response_format: a nested model always emits top level $defs, which
            # guided decoding backends limited to the flat JSON Schema subset reject.
            response: LLMCompletionResponse = await self._model.completion_async(
                messages=prompt,
            )  # type: ignore

            output = CommunityReportResponse(**_parse_json_response(response.content))
        except Exception as e:
            logger.exception("error generating community report")
            self._on_error(e, traceback.format_exc(), None)

        text_output = self._get_text_output(output) if output else ""
        return CommunityReportsResult(
            structured_output=output,
            output=text_output,
        )

    def _get_text_output(self, report: CommunityReportResponse) -> str:
        report_sections = "\n\n".join(
            f"## {f.summary}\n\n{f.explanation}" for f in report.findings
        )
        return f"# {report.title}\n\n{report.summary}\n\n{report_sections}"
