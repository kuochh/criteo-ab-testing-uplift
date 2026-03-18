import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from helpers import load, styled_metric, COLORS, PLOTLY_LAYOUT

st.header("Act 2: What CATE Actually Captures")
st.markdown(
    "Act 1 showed that CATE identifies high-value users. Now we look at what CATE actually "
    "measures. What is the score actually picking up? With 96.4% non-compliance, a high CATE "
    "could reflect several different things."
)

# ── 2.1 CATE distribution ─────────────────────────────────────────────
st.divider()
st.subheader("2.1 CATE Score Distribution")

cate_pred = load("cate_predictions_test_sample.csv")
learner_col = st.selectbox(
    "* Select Learner",
    ["x_cate", "s_cate", "t_cate", "r_cate"],
    format_func=lambda x: x.replace("_cate", "-Learner").capitalize(),
    index=0,
)

fig_hist = px.histogram(
    cate_pred,
    x=learner_col,
    nbins=100,
    color_discrete_sequence=[COLORS["teal"]],
)
fig_hist.update_layout(
    **PLOTLY_LAYOUT,  # pyright: ignore[reportArgumentType]
    title=f"Distribution of CATE Scores ({learner_col.replace('_cate', '-Learner').capitalize()})",
    xaxis_title="Predicted CATE",
    yaxis_title="Count",
    showlegend=False,
)
st.plotly_chart(fig_hist, width="stretch")

st.markdown(
    "Most users cluster near zero. This is expected: with 96.4% non-compliance, the average "
    "treatment effect is heavily diluted, so most predicted CATEs are small. The right tail "
    "contains users with meaningfully positive scores. These are users the model predicts will "
    "benefit most from campaign assignment."
)

# ── 2.2 Three users example ────────────────────────────────────────────
st.divider()
st.subheader("2.2 Same CATE, Different Mechanisms")
st.latex(r"\text{CATE}(X) = \underbrace{P(E \mid T{=}1, X)}_{\text{Reachability}} \times \underbrace{\text{LATE}(X)}_{\text{Responsiveness}}")

st.info(
    "**CATE** is the overall effect of campaign assignment on conversion. "
    "It decomposes into two components:  \n"
    "- **Reachability** P(E): how likely this user is to actually "
    "see the ad when assigned.  \n"
    "- **Responsiveness** LATE: how much seeing the ad increases this "
    "user's conversion probability."
)
st.markdown(
    "All three vary by user based on their features, which is what "
    "makes user-level bidding possible. For the rest of this analysis, we use CATE, P(E), and LATE."
)

with st.expander("Derivation"):
    st.latex(r"\text{CATE}(X) = E[Y \mid T{=}1, X] - E[Y \mid T{=}0, X]")
    st.markdown("Decompose $E[Y \\mid T{=}1, X]$ by exposure (law of total expectation):")
    st.latex(r"= E[Y \mid E{=}1, T{=}1, X] \cdot P(E{=}1 \mid T{=}1, X) + E[Y \mid E{=}0, T{=}1, X] \cdot P(E{=}0 \mid T{=}1, X)")
    st.markdown("Apply exclusion restriction (assignment affects $Y$ only through $E$):")
    st.latex(r"E[Y \mid E{=}0, T{=}1, X] = E[Y \mid E{=}0, X] \quad \text{and} \quad E[Y \mid T{=}0, X] = E[Y \mid E{=}0, X]")
    st.markdown("Substitute and simplify:")
    st.latex(r"= E[Y \mid E{=}1, X] \cdot P(E{=}1 \mid T{=}1, X) + E[Y \mid E{=}0, X] \cdot [1 - P(E{=}1 \mid T{=}1, X)] - E[Y \mid E{=}0, X]")
    st.latex(r"= E[Y \mid E{=}1, X] \cdot P(E{=}1 \mid T{=}1, X) - E[Y \mid E{=}0, X] \cdot P(E{=}1 \mid T{=}1, X)")
    st.latex(r"= [E[Y \mid E{=}1, X] - E[Y \mid E{=}0, X]] \cdot P(E{=}1 \mid T{=}1, X)")
    st.latex(r"= \text{LATE}(X) \cdot P(E{=}1 \mid T{=}1, X)")

st.markdown(
    "Three hypothetical users with identical CATE scores illustrate the problem:"
)

example_data = pd.DataFrame({
    "User": ["Alice", "Bob", "Carol"],
    "CATE": [0.005, 0.005, 0.005],
    "P(E)": ["High (80%)", "Low (5%)", "Medium (25%)"],
    "LATE (Response)": ["Low (0.6%)", "High (10%)", "Medium (2%)"],
    "Optimal Bid": ["Low — easy to reach", "High — hard to reach but responds well", "Medium"],
    "Mechanism": ["Reachability-driven", "Responsiveness-driven", "Balanced"],
})

st.dataframe(example_data, width="stretch", hide_index=True)


st.markdown(
    "All three users look identical to any CATE-based system. They would all receive the same bid "
    "in a real-time auction. But Alice is already easy to reach (80% reachability). She will likely "
    "see the ad even with a lower bid level, so overbidding wastes budget. Bob is nearly impossible "
    "to reach (5% reachability) but converts at 10% when he does see the ad. He is worth paying a "
    "premium for. Carol falls in between. Optimal bidding requires knowing *which mechanism* is "
    "driving the score, not just the score itself."
)

# ── 2.3 Estimating Reachability ────────────────────────────────────────
st.divider()
st.subheader("2.3 Estimating Reachability")

st.markdown(
    "We know CATE blends reachability and responsiveness. To separate them, we start with "
    "the easier component: reachability.\n\n"
    "P(E) is estimated by training an XGBoost classifier on the treatment group to predict "
    "which users actually see ads, using the 12 anonymized user features."
)

exp_model = load("exposure_model_details.csv").iloc[0]
styled_metric("P(E) Model AUC", f"{exp_model['test_auc']:.3f}", "XGBoost classifier predicting ad exposure")

# ── 2.4 CATE vs P(E) scatter ──────────────────────────────────────────
st.divider()
st.subheader("2.4 CATE Correlates with Reachability")

scatter_data = load("cate_vs_pe_sample.csv")
decomp = load("decomposition_correlations.csv").iloc[0]

fig_scatter = px.scatter(
    scatter_data,
    x="p_exposure",
    y="x_cate",
    opacity=0.3,
    color_discrete_sequence=[COLORS["teal"]],
)
fig_scatter.update_layout(
    **PLOTLY_LAYOUT,  # pyright: ignore[reportArgumentType]
    title=f"X-Learner CATE vs P(E) — Correlation: {decomp['cate_vs_pe']:.3f}",
    xaxis_title="P(Exposure | T=1, X)",
    yaxis_title="X-Learner CATE",
    showlegend=False,
)
st.plotly_chart(fig_scatter, width="stretch")

styled_metric("CATE vs P(E) Correlation", f"{decomp['cate_vs_pe']:.3f}")

st.markdown(
    f"The correlation between CATE and P(E) is {decomp['cate_vs_pe']:.2f}. A large portion of the CATE signal "
    "comes from reachability: users who have high CATE scores tend to be users who are easy to reach "
    "with ads. This confirms the blind spot from Act 1. CATE is a useful ranking tool, but it "
    "cannot tell us *why* a user scores high. Is it because they are reachable, responsive, or both?"
)

# ── Summary ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("Summary")

st.markdown("""
<div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 1.2rem 1.5rem; margin: 0.5rem 0;">
<strong>Key Takeaways:</strong><br><br>
1. <strong>CATE correlates with reachability.</strong> With 96.4% non-compliance, CATE correlates strongly with P(E). A high CATE score does not tell us whether a user is easy to reach, responsive, or both.<br>
2. <strong>Same CATE, different mechanisms.</strong> Users with identical CATE scores can be reachability-driven or responsiveness-driven, requiring opposite bids.<br>
3. <strong>A single score is not enough for bidding.</strong> To bid optimally in ad auctions, we need to separate reachability from responsiveness.
</div>
""", unsafe_allow_html=True)
