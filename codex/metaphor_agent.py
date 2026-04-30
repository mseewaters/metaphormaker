"""Core metaphor generation logic for Metaphor Creator."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI


QUALITY_THRESHOLD = 8
MAX_ITERATIONS = 3
DEFAULT_MODEL = "gpt-5.4-mini"
ENV_PATHS = (Path(__file__).resolve().parent / ".env", Path(__file__).resolve().parent.parent / ".env")
MODEL_PRICING_PER_1M = {
    "gpt-5.4-mini": {"input": 0.75, "cached_input": 0.075, "output": 4.50},
    "gpt-5.4-nano": {"input": 0.20, "cached_input": 0.02, "output": 1.25},
    "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
    "gpt-4.1-nano": {"input": 0.10, "cached_input": 0.025, "output": 0.40},
    "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
}


@dataclass
class Evaluation:
    clarity: int
    novelty: int
    accuracy: int
    notes: str

    @property
    def passes(self) -> bool:
        return min(self.clarity, self.novelty, self.accuracy) >= QUALITY_THRESHOLD


def create_metaphors(
    concept: str,
    audience: str,
    *,
    tone: str = "Any",
    category: str = "Any",
    count: int = 3,
    model: str | None = None,
) -> dict[str, Any]:
    """Generate, evaluate, and improve metaphors until quality is acceptable."""
    concept = concept.strip()
    audience = audience.strip()
    tone = _normalize_option(tone)
    category = _normalize_option(category)

    if not concept or not audience:
        raise ValueError("Both concept and audience are required.")

    _load_env_files()
    model_name = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    client = OpenAI()

    metaphors: list[dict[str, str]] = []
    thinking: list[dict[str, Any]] = []
    usage_calls: list[dict[str, Any]] = []
    feedback = ""

    for iteration in range(1, MAX_ITERATIONS + 1):
        metaphors, generate_usage = _generate_metaphors(
            client=client,
            model=model_name,
            concept=concept,
            audience=audience,
            tone=tone,
            category=category,
            count=count,
            prior_metaphors=metaphors,
            feedback=feedback,
        )
        generate_usage["iteration"] = iteration
        generate_usage["phase"] = "generate"
        usage_calls.append(generate_usage)

        evaluation, evaluate_usage = _evaluate_metaphors(
            client=client,
            model=model_name,
            concept=concept,
            audience=audience,
            tone=tone,
            category=category,
            metaphors=metaphors,
        )
        evaluate_usage["iteration"] = iteration
        evaluate_usage["phase"] = "evaluate"
        usage_calls.append(evaluate_usage)

        iteration_usage = _summarize_usage([generate_usage, evaluate_usage])

        decision = _improvement_decision(evaluation, iteration)
        thinking.append(
            {
                "iteration": iteration,
                "scores": {
                    "clarity": evaluation.clarity,
                    "novelty": evaluation.novelty,
                    "accuracy": evaluation.accuracy,
                },
                "notes": evaluation.notes,
                "decision": decision,
                "usage": iteration_usage,
            }
        )

        if evaluation.passes or iteration == MAX_ITERATIONS:
            break

        feedback = evaluation.notes

    return {
        "metaphors": metaphors[:count],
        "thinking": thinking,
        "iterations": len(thinking),
        "usage": {
            "model": model_name,
            "calls": usage_calls,
            "total": _summarize_usage(usage_calls),
        },
    }


def _generate_metaphors(
    *,
    client: OpenAI,
    model: str,
    concept: str,
    audience: str,
    tone: str,
    category: str,
    count: int,
    prior_metaphors: list[dict[str, str]],
    feedback: str,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Ask the model for audience-specific metaphors in structured JSON."""
    revision_context = ""
    if prior_metaphors and feedback:
        revision_context = (
            "Revise the previous attempt using this evaluator feedback:\n"
            f"{feedback}\n\nPrevious metaphors:\n{json.dumps(prior_metaphors, indent=2)}"
        )

    prompt = f"""
Create {count} high-quality metaphors for this concept and audience.

Concept: {concept}
Audience: {audience}
Tone: {_style_instruction(tone, "tone")}
Category: {_style_instruction(category, "category")}

{revision_context}

Each metaphor must be accurate, concrete, audience-appropriate, and not a cliche.
If tone or category is "Any", choose what best serves the audience and concept.
If a category is selected, draw the metaphor from that domain without forcing a bad fit.
Return JSON only with this shape:
{{
  "metaphors": [
    {{
      "metaphor": "string",
      "why_it_works": "string",
      "where_it_breaks_down": "string"
    }}
  ]
}}
"""

    data, usage = _call_json(client, model, _system_prompt(), prompt)
    metaphors = data.get("metaphors", [])
    if not isinstance(metaphors, list) or not metaphors:
        raise ValueError("The model did not return any metaphors.")

    return [_normalize_metaphor(item) for item in metaphors[:count]], usage


def _evaluate_metaphors(
    *,
    client: OpenAI,
    model: str,
    concept: str,
    audience: str,
    tone: str,
    category: str,
    metaphors: list[dict[str, str]],
) -> tuple[Evaluation, dict[str, Any]]:
    """Score the batch so weak generations can be improved."""
    prompt = f"""
Evaluate these metaphors for the target audience.

Concept: {concept}
Audience: {audience}
Requested tone: {_style_instruction(tone, "tone")}
Requested category: {_style_instruction(category, "category")}
Metaphors:
{json.dumps(metaphors, indent=2)}

Score the full set from 1 to 10 for:
- clarity: easy for the audience to understand
- novelty: fresh and memorable, not generic
- accuracy: preserves important truth about the concept
Also consider whether the metaphors respect the requested tone and category.

Return JSON only with this shape:
{{
  "clarity": 1,
  "novelty": 1,
  "accuracy": 1,
  "notes": "specific improvement guidance if any score is below 8"
}}
"""

    data, usage = _call_json(client, model, _evaluator_prompt(), prompt)
    return (
        Evaluation(
            clarity=_clamp_score(data.get("clarity")),
            novelty=_clamp_score(data.get("novelty")),
            accuracy=_clamp_score(data.get("accuracy")),
            notes=str(data.get("notes", "")).strip() or "No evaluator notes returned.",
        ),
        usage,
    )


def _call_json(client: OpenAI, model: str, system: str, prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Call the LLM and parse a JSON object response."""
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("The model returned an empty response.")

    usage = _extract_usage(response.usage, model)
    return json.loads(content), usage


def _extract_usage(raw_usage: Any, model: str) -> dict[str, Any]:
    prompt_tokens = _usage_value(raw_usage, "prompt_tokens")
    completion_tokens = _usage_value(raw_usage, "completion_tokens")
    total_tokens = _usage_value(raw_usage, "total_tokens")
    cached_tokens = _cached_prompt_tokens(raw_usage)

    usage = {
        "model": model,
        "prompt_tokens": prompt_tokens,
        "cached_prompt_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    usage["estimated_cost_usd"] = _estimate_cost_usd(model, usage)
    return usage


def _usage_value(raw_usage: Any, field: str) -> int:
    if raw_usage is None:
        return 0
    value = getattr(raw_usage, field, 0)
    return int(value or 0)


def _cached_prompt_tokens(raw_usage: Any) -> int:
    if raw_usage is None:
        return 0

    details = getattr(raw_usage, "prompt_tokens_details", None)
    if details is None:
        return 0

    if isinstance(details, dict):
        return int(details.get("cached_tokens") or 0)

    return int(getattr(details, "cached_tokens", 0) or 0)


def _estimate_cost_usd(model: str, usage: dict[str, Any]) -> float | None:
    pricing = MODEL_PRICING_PER_1M.get(model)
    if pricing is None:
        return None

    cached_input = usage["cached_prompt_tokens"]
    uncached_input = max(0, usage["prompt_tokens"] - cached_input)
    output = usage["completion_tokens"]

    return (
        (uncached_input * pricing["input"])
        + (cached_input * pricing["cached_input"])
        + (output * pricing["output"])
    ) / 1_000_000


def _summarize_usage(calls: list[dict[str, Any]]) -> dict[str, Any]:
    cost_values = [call.get("estimated_cost_usd") for call in calls]
    known_costs = [cost for cost in cost_values if cost is not None]
    return {
        "prompt_tokens": sum(call.get("prompt_tokens", 0) for call in calls),
        "cached_prompt_tokens": sum(call.get("cached_prompt_tokens", 0) for call in calls),
        "completion_tokens": sum(call.get("completion_tokens", 0) for call in calls),
        "total_tokens": sum(call.get("total_tokens", 0) for call in calls),
        "estimated_cost_usd": sum(known_costs) if len(known_costs) == len(calls) else None,
    }


def _load_env_files() -> None:
    """Load simple KEY=VALUE lines from .env without overriding shell env vars."""
    for path in ENV_PATHS:
        if not path.exists():
            continue

        for line in path.read_text(encoding="utf-8").splitlines():
            key, value = _parse_env_line(line)
            if key and key not in os.environ:
                os.environ[key] = value


def _parse_env_line(line: str) -> tuple[str, str]:
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return "", ""

    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip().strip("\"'")
    return key, value


def _normalize_option(value: str) -> str:
    value = value.strip()
    return value if value else "Any"


def _style_instruction(value: str, label: str) -> str:
    if value.lower() == "any":
        return f"Any {label}; choose the strongest fit."
    return value


def _normalize_metaphor(item: Any) -> dict[str, str]:
    """Keep output stable for the Streamlit renderer."""
    if not isinstance(item, dict):
        item = {}

    return {
        "metaphor": str(item.get("metaphor", "")).strip(),
        "why_it_works": str(item.get("why_it_works", "")).strip(),
        "where_it_breaks_down": str(item.get("where_it_breaks_down", "")).strip(),
    }


def _clamp_score(value: Any) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return 1
    return max(1, min(10, score))


def _improvement_decision(evaluation: Evaluation, iteration: int) -> str:
    if evaluation.passes:
        return "Quality threshold met; stopping."
    if iteration == MAX_ITERATIONS:
        return "Reached maximum iterations; returning best current result."
    return "One or more scores are below 8; generating an improved version."


def _system_prompt() -> str:
    return (
        "You are a precise metaphor designer. You tailor explanations to the "
        "audience, protect conceptual accuracy, and avoid vague comparisons."
    )


def _evaluator_prompt() -> str:
    return (
        "You are a strict editor evaluating metaphor quality. Score honestly and "
        "give concise, actionable guidance for improvement."
    )
