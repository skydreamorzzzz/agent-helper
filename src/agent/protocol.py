from __future__ import annotations

import json
import re
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_call"]
    tool: str
    arguments: dict


class FinalAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["final_answer"]
    content: str


AgentMessage = Annotated[Union[ToolCall, FinalAnswer], Field(discriminator="type")]


class AgentOutput(BaseModel):
    output: AgentMessage


class ModelOutputParseError(ValueError):
    pass


def extract_json_from_markdown(text: str) -> str | None:
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def extract_first_json_object(text: str) -> str | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            _, end = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        return text[match.start() : match.start() + end]
    return None


def parse_model_output(raw: str) -> AgentMessage:
    candidates = [raw]
    extracted = extract_json_from_markdown(raw)
    if extracted:
        candidates.append(extracted)
    embedded = extract_first_json_object(raw)
    if embedded:
        candidates.append(embedded)

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            return AgentOutput.model_validate({"output": data}).output
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc

    raise ModelOutputParseError(f"Model output is not valid agent JSON: {last_error}")
