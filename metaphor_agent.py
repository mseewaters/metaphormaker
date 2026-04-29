"""
Core metaphor generation and evaluation logic.
Uses an iterative loop: generate → evaluate → improve → repeat.
"""

import json
import anthropic

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"


def generate_metaphors(concept: str, audience: str, previous_feedback: str = "") -> list[dict]:
    """Ask the model to produce 2–3 metaphors for the given concept and audience."""

    improvement_note = (
        f"\n\nPrevious attempt feedback to improve upon:\n{previous_feedback}"
        if previous_feedback
        else ""
    )

    prompt = f"""You are an expert at creating vivid, memorable metaphors that make complex ideas accessible.

Concept to explain: {concept}
Target audience: {audience}{improvement_note}

Generate exactly 3 metaphors tailored specifically to this audience's background and interests.

Return ONLY valid JSON — no prose, no markdown fences — in this exact structure:
[
  {{
    "metaphor": "The metaphor text itself",
    "why_it_works": "Explanation of why this resonates with the audience",
    "where_it_breaks": "Where this analogy could mislead or fall apart"
  }}
]"""

    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()
    # Strip markdown code fences if the model adds them anyway
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


def evaluate_metaphors(concept: str, audience: str, metaphors: list[dict]) -> list[dict]:
    """Score each metaphor 1–10 on clarity, novelty, and accuracy."""

    metaphors_json = json.dumps(metaphors, indent=2)

    prompt = f"""You are a critical evaluator of explanatory metaphors.

Concept: {concept}
Audience: {audience}

Metaphors to evaluate:
{metaphors_json}

Score each metaphor on three dimensions (1–10 each):
- clarity: How easily does the audience grasp the concept?
- novelty: How fresh and memorable is the comparison?
- accuracy: How faithfully does it map to the concept?

Also provide a brief "feedback" string (1–2 sentences) on what to improve.

Return ONLY valid JSON — no prose, no markdown fences — in this exact structure:
[
  {{
    "clarity": 8,
    "novelty": 6,
    "accuracy": 7,
    "feedback": "Short improvement note"
  }}
]

Return one entry per metaphor in the same order."""

    message = client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    scores = json.loads(raw)

    # Merge scores back into the metaphor dicts
    for metaphor, score in zip(metaphors, scores):
        metaphor["scores"] = score

    return metaphors


def _average_score(metaphor: dict) -> float:
    s = metaphor.get("scores", {})
    values = [s.get("clarity", 0), s.get("novelty", 0), s.get("accuracy", 0)]
    return sum(values) / len(values)


def _build_feedback_summary(metaphors: list[dict]) -> str:
    """Summarise per-metaphor feedback into a single improvement prompt."""
    lines = []
    for i, m in enumerate(metaphors, 1):
        avg = _average_score(m)
        fb = m.get("scores", {}).get("feedback", "")
        lines.append(f"Metaphor {i} (avg score {avg:.1f}/10): {fb}")
    return "\n".join(lines)


def run_metaphor_agent(
    concept: str,
    audience: str,
    max_iterations: int = 3,
    quality_threshold: float = 7.5,
) -> dict:
    """
    Main agent loop:
      1. Generate metaphors
      2. Evaluate them
      3. If average quality is below threshold and iterations remain, improve and repeat
      4. Return the best set along with the full trace for "show thinking"
    """

    trace = []  # Records each iteration for the "show thinking" panel
    metaphors = []
    feedback_summary = ""

    for iteration in range(1, max_iterations + 1):
        # --- Generate ---
        metaphors = generate_metaphors(concept, audience, previous_feedback=feedback_summary)

        # --- Evaluate ---
        metaphors = evaluate_metaphors(concept, audience, metaphors)

        # Compute overall quality for this iteration
        avg_scores = [_average_score(m) for m in metaphors]
        iteration_avg = sum(avg_scores) / len(avg_scores)

        trace.append(
            {
                "iteration": iteration,
                "metaphors": [m.copy() for m in metaphors],
                "iteration_avg": round(iteration_avg, 2),
                "improved": iteration > 1,
            }
        )

        # --- Decide whether to iterate ---
        if iteration_avg >= quality_threshold:
            break

        if iteration < max_iterations:
            feedback_summary = _build_feedback_summary(metaphors)

    return {
        "metaphors": metaphors,
        "trace": trace,
        "iterations_run": len(trace),
    }
