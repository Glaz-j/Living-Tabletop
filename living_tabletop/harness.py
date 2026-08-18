from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from .llm import LLMResult, OpenAICompatibleLLM


StructuredModel = TypeVar("StructuredModel", bound=BaseModel)
PostValidator = Callable[[StructuredModel], None]


class HarnessValidationError(ValueError):
    """Raised after the model fails the structured contract and one repair."""

    def __init__(self, message: str, *, result: LLMResult | None = None):
        super().__init__(message)
        self.result = result


@dataclass(frozen=True, slots=True)
class HarnessResult(Generic[StructuredModel]):
    value: StructuredModel
    llm_result: LLMResult
    attempts: int

    @property
    def repaired(self) -> bool:
        return self.attempts > 1


def strict_json_schema(model: type[BaseModel]) -> dict:
    """Make every model field explicit so local grammar decoders cannot omit defaults."""

    schema = deepcopy(model.model_json_schema())

    def visit(node):
        if isinstance(node, dict):
            node.pop("default", None)
            properties = node.get("properties")
            if isinstance(properties, dict):
                node["additionalProperties"] = False
                node["required"] = list(properties)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(schema)
    return schema


class StructuredHarness:
    """Thin native harness: schema-constrained call, validation, and one repair."""

    def __init__(self, llm: OpenAICompatibleLLM, *, max_repairs: int = 1):
        self.llm = llm
        self.max_repairs = max(0, max_repairs)

    @staticmethod
    def _validation_message(error: Exception) -> str:
        if isinstance(error, ValidationError):
            parts = []
            for item in error.errors(include_url=False)[:8]:
                location = ".".join(str(value) for value in item.get("loc", ())) or "root"
                parts.append(f"{location}: {item.get('msg', 'invalid value')}")
            return "; ".join(parts)
        return str(error)[:1200]

    @staticmethod
    def _combine_results(results: list[LLMResult]) -> LLMResult:
        final = results[-1]
        input_tokens = [item.input_tokens for item in results]
        output_tokens = [item.output_tokens for item in results]
        return LLMResult(
            data=final.data,
            latency_ms=sum(item.latency_ms for item in results),
            input_tokens=sum(input_tokens) if all(item is not None for item in input_tokens) else None,
            output_tokens=sum(output_tokens) if all(item is not None for item in output_tokens) else None,
        )

    def run(
        self,
        output_model: type[StructuredModel],
        *,
        system: str,
        user_payload: dict,
        max_output_tokens: int | None = None,
        post_validate: PostValidator[StructuredModel] | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        response_schema: dict | None = None,
    ) -> HarnessResult[StructuredModel]:
        schema = response_schema or strict_json_schema(output_model)
        results: list[LLMResult] = []
        payload = dict(user_payload)
        repair_note = ""

        for attempt in range(self.max_repairs + 1):
            result = self.llm.complete_json(
                system=system + repair_note,
                user_payload=payload,
                max_output_tokens=max_output_tokens,
                response_schema=schema,
                schema_name=output_model.__name__,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )
            results.append(result)
            try:
                value = output_model.model_validate(result.data)
                if post_validate is not None:
                    post_validate(value)
                return HarnessResult(
                    value=value,
                    llm_result=self._combine_results(results),
                    attempts=attempt + 1,
                )
            except (ValidationError, ValueError, TypeError) as error:
                if attempt >= self.max_repairs:
                    raise HarnessValidationError(
                        self._validation_message(error),
                        result=self._combine_results(results),
                    ) from error
                payload = {
                    **user_payload,
                    "_harness_repair": {
                        "previous_output": result.data,
                        "validation_errors": self._validation_message(error),
                        "instruction": (
                            "Return a complete replacement JSON object. Preserve the player's full intent "
                            "and fix every validation error; do not explain the repair."
                        ),
                    },
                }
                repair_note = (
                    "\nYour previous JSON failed the runtime contract. Follow _harness_repair and return "
                    "one complete corrected JSON object only."
                )

        raise AssertionError("unreachable")
