from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
import pytest

from living_tabletop.harness import (
    HarnessValidationError,
    StructuredHarness,
    strict_json_schema,
)
from living_tabletop.models import LLMResult


class HarnessContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str = Field(min_length=1)
    note: str | None = None


class SequenceLLM:
    enabled = True

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        output = self.outputs.pop(0)
        return LLMResult(
            data=output,
            latency_ms=7,
            input_tokens=11,
            output_tokens=5,
        )


def test_strict_schema_requires_defaulted_fields():
    schema = strict_json_schema(HarnessContract)

    assert schema["required"] == ["value", "note"]
    assert schema["additionalProperties"] is False
    assert "default" not in schema["properties"]["note"]


def test_harness_repairs_invalid_output_once_and_aggregates_usage():
    llm = SequenceLLM(
        [
            {"value": "", "note": None},
            {"value": "fixed", "note": None},
        ]
    )

    outcome = StructuredHarness(llm).run(
        HarnessContract,
        system="Return the contract.",
        user_payload={"input": "hello"},
    )

    assert outcome.value.value == "fixed"
    assert outcome.repaired is True
    assert outcome.attempts == 2
    assert outcome.llm_result.latency_ms == 14
    assert outcome.llm_result.input_tokens == 22
    assert outcome.llm_result.output_tokens == 10
    assert "_harness_repair" in llm.calls[1]["user_payload"]
    assert llm.calls[0]["response_schema"]["required"] == ["value", "note"]


def test_harness_repairs_application_validation_failure():
    llm = SequenceLLM(
        [
            {"value": "unknown", "note": None},
            {"value": "available", "note": None},
        ]
    )

    def require_available(output: HarnessContract) -> None:
        if output.value != "available":
            raise ValueError("value is not currently available")

    outcome = StructuredHarness(llm).run(
        HarnessContract,
        system="Return the contract.",
        user_payload={},
        post_validate=require_available,
    )

    assert outcome.value.value == "available"
    assert outcome.repaired is True
    assert "not currently available" in llm.calls[1]["user_payload"]["_harness_repair"]["validation_errors"]


def test_harness_fails_after_single_repair():
    llm = SequenceLLM(
        [
            {"value": "", "note": None},
            {"value": "", "note": None},
        ]
    )

    with pytest.raises(HarnessValidationError) as caught:
        StructuredHarness(llm).run(
            HarnessContract,
            system="Return the contract.",
            user_payload={},
        )

    assert len(llm.calls) == 2
    assert caught.value.result is not None
    assert caught.value.result.latency_ms == 14


def test_harness_accepts_a_schema_override():
    llm = SequenceLLM([{"value": "ok", "note": None}])
    schema = {
        "type": "object",
        "properties": {
            "value": {"type": "string"},
            "note": {"type": ["string", "null"]},
        },
        "required": ["value", "note"],
        "additionalProperties": False,
    }

    StructuredHarness(llm).run(
        HarnessContract,
        system="Return the contract.",
        user_payload={},
        response_schema=schema,
    )

    assert llm.calls[0]["response_schema"] is schema
