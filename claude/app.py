"""Streamlit UI for the Metaphor Creator app."""

import streamlit as st
from metaphor_agent import run_metaphor_agent

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Metaphor Creator",
    page_icon="💡",
    layout="centered",
)

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

col_tone, col_cat = st.columns(2)

with col_tone:
    tone = st.selectbox(
        "Tone",
        ["Any", "Serious", "Professional", "Funny", "Sarcastic", "Playful", "Poetic", "Blunt"],
    )

with col_cat:
    category = st.selectbox(
        "Category",
        ["Any", "Sports", "Food", "Nature", "Business", "Money", "Technology", "Music", "Architecture"],
    )

show_thinking = st.toggle("Show agent thinking", value=False)

generate_btn = st.button(
    "Generate metaphors",
    type="primary",
    disabled=not (concept and audience),
)

# ── Generation & display ──────────────────────────────────────────────────────
if generate_btn:
    with st.spinner("Thinking through metaphors…"):
        result = run_metaphor_agent(concept.strip(), audience.strip(), tone=tone, category=category)

    metaphors = result["metaphors"]
    trace = result["trace"]
    iterations = result["iterations_run"]
    usage = result["usage"]

    col_status, col_usage = st.columns([2, 1])
    with col_status:
        st.success(f"Generated in {iterations} iteration{'s' if iterations != 1 else ''}.")
    with col_usage:
        st.caption(
            f"**{usage['input_tokens']:,}** in · **{usage['output_tokens']:,}** out · "
            f"**${usage['cost_usd']:.4f}**"
        )

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
