"""Streamlit UI for Metaphor Creator."""

from __future__ import annotations

import streamlit as st
from openai import OpenAIError

from metaphor_agent import create_metaphors


TONE_OPTIONS = [
    "Any",
    "Serious",
    "Professional",
    "Funny",
    "Sarcastic",
    "Warm",
    "Inspirational",
    "Plainspoken",
]

CATEGORY_OPTIONS = [
    "Any",
    "Sports",
    "Food",
    "Nature",
    "Business",
    "Money",
    "Technology",
    "Travel",
    "Music",
    "Everyday Life",
]


st.set_page_config(page_title="Metaphor Creator", page_icon=":material/psychology:", layout="centered")


def main() -> None:
    st.title("Metaphor Creator")
    st.write("Generate audience-tailored metaphors with critique and refinement.")

    with st.form("metaphor-form"):
        concept = st.text_area(
            "Complex concept",
            placeholder="Example: transformer attention, technical debt, compound interest",
            height=110,
        )
        audience = st.text_input(
            "Audience",
            placeholder="Example: middle school students, startup founders, nurses",
        )
        tone = st.selectbox("Tone", TONE_OPTIONS)
        category = st.selectbox("Category", CATEGORY_OPTIONS)
        show_thinking = st.toggle("Show agent thinking", value=False)
        submitted = st.form_submit_button("Generate metaphors", type="primary")

    if submitted:
        if not concept.strip() or not audience.strip():
            st.warning("Enter both a concept and an audience.")
            return

        with st.spinner("Generating, evaluating, and improving metaphors..."):
            try:
                result = create_metaphors(
                    concept,
                    audience,
                    tone=tone,
                    category=category,
                )
            except OpenAIError as exc:
                st.error(f"LLM request failed: {exc}")
                return
            except Exception as exc:
                st.error(f"Could not generate metaphors: {exc}")
                return

        _render_metaphors(result["metaphors"])
        _render_usage(result["usage"])

        if show_thinking:
            _render_thinking(result["thinking"])


def _render_metaphors(metaphors: list[dict[str, str]]) -> None:
    st.subheader("Results")

    for index, item in enumerate(metaphors, start=1):
        with st.container(border=True):
            st.markdown(f"### {index}. {item['metaphor']}")
            st.markdown("**Why it works for this audience**")
            st.write(item["why_it_works"])
            st.markdown("**Where it breaks down or could mislead**")
            st.write(item["where_it_breaks_down"])


def _render_usage(usage: dict) -> None:
    total = usage["total"]
    cost = _format_cost(total["estimated_cost_usd"])

    st.caption(
        f"Usage for this generation: {total['total_tokens']:,} tokens "
        f"({total['prompt_tokens']:,} input, {total['completion_tokens']:,} output) | "
        f"Estimated cost: {cost} | Model: {usage['model']}"
    )


def _render_thinking(thinking: list[dict]) -> None:
    st.subheader("Agent Thinking")

    for step in thinking:
        with st.expander(f"Iteration {step['iteration']}", expanded=True):
            scores = step["scores"]
            st.write(
                f"Clarity: {scores['clarity']} / 10 | "
                f"Novelty: {scores['novelty']} / 10 | "
                f"Accuracy: {scores['accuracy']} / 10"
            )
            st.markdown("**Evaluator notes**")
            st.write(step["notes"])
            st.markdown("**Decision**")
            st.write(step["decision"])
            st.markdown("**Token usage**")
            usage = step["usage"]
            st.write(
                f"{usage['total_tokens']:,} tokens "
                f"({usage['prompt_tokens']:,} input, {usage['completion_tokens']:,} output), "
                f"estimated cost {_format_cost(usage['estimated_cost_usd'])}"
            )


def _format_cost(cost: float | None) -> str:
    if cost is None:
        return "unavailable for this model"
    if cost < 0.01:
        return f"${cost:.6f}"
    return f"${cost:.4f}"


if __name__ == "__main__":
    main()
