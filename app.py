"""Streamlit UI for the Metaphor Creator app."""

import os
import streamlit as st
from metaphor_agent import run_metaphor_agent

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Metaphor Creator",
    page_icon="💡",
    layout="centered",
)

# ── API key (sidebar or env) ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    api_key_input = st.text_input(
        "Anthropic API key",
        type="password",
        placeholder="sk-ant-…",
        help="Get yours at console.anthropic.com",
    )
    if api_key_input:
        os.environ["ANTHROPIC_API_KEY"] = api_key_input

api_key_ready = bool(os.environ.get("ANTHROPIC_API_KEY"))

st.title("💡 Metaphor Creator")
st.caption("Turn any complex concept into vivid, audience-tailored metaphors.")

# ── Inputs ────────────────────────────────────────────────────────────────────
concept = st.text_input(
    "Complex concept",
    placeholder="e.g. quantum entanglement, compound interest, recursion…",
)

audience = st.text_input(
    "Target audience",
    placeholder="e.g. middle schoolers, gardeners, financial advisors…",
)

show_thinking = st.toggle("Show agent thinking", value=False)

if not api_key_ready:
    st.info("Enter your Anthropic API key in the sidebar to get started.", icon="🔑")

generate_btn = st.button(
    "Generate metaphors",
    type="primary",
    disabled=not (concept and audience and api_key_ready),
)

# ── Generation & display ──────────────────────────────────────────────────────
if generate_btn:
    with st.spinner("Thinking through metaphors…"):
        result = run_metaphor_agent(concept.strip(), audience.strip())

    metaphors = result["metaphors"]
    trace = result["trace"]
    iterations = result["iterations_run"]

    st.success(f"Generated in {iterations} iteration{'s' if iterations != 1 else ''}.")

    # ── Metaphor cards ────────────────────────────────────────────────────────
    st.subheader("Your metaphors")

    for i, m in enumerate(metaphors, 1):
        with st.expander(f"Metaphor {i}", expanded=True):
            st.markdown(f"**{m['metaphor']}**")

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Why it works**")
                st.write(m["why_it_works"])
            with col_b:
                st.markdown("**Where it breaks down**")
                st.write(m["where_it_breaks"])

            # Final scores (from last iteration)
            scores = m.get("scores", {})
            if scores:
                clarity = scores.get("clarity", "–")
                novelty = scores.get("novelty", "–")
                accuracy = scores.get("accuracy", "–")
                avg = round(
                    (scores.get("clarity", 0) + scores.get("novelty", 0) + scores.get("accuracy", 0)) / 3,
                    1,
                )
                st.markdown(
                    f"<small>Scores — Clarity: **{clarity}**/10 · "
                    f"Novelty: **{novelty}**/10 · "
                    f"Accuracy: **{accuracy}**/10 · "
                    f"Avg: **{avg}**/10</small>",
                    unsafe_allow_html=True,
                )

    # ── Agent thinking panel ──────────────────────────────────────────────────
    if show_thinking:
        st.divider()
        st.subheader("Agent thinking")

        for step in trace:
            n = step["iteration"]
            avg = step["iteration_avg"]
            label = f"Iteration {n}  —  avg score {avg}/10"
            if step["improved"]:
                label += "  *(improved from previous)*"

            with st.expander(label):
                for j, m in enumerate(step["metaphors"], 1):
                    s = m.get("scores", {})
                    st.markdown(
                        f"**Metaphor {j}** — "
                        f"Clarity {s.get('clarity','–')} · "
                        f"Novelty {s.get('novelty','–')} · "
                        f"Accuracy {s.get('accuracy','–')}"
                    )
                    if s.get("feedback"):
                        st.caption(f"Feedback: {s['feedback']}")
