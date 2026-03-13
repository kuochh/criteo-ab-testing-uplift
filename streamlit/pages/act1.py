import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from helpers import load, styled_metric, COLORS, PLOTLY_LAYOUT, LEARNER_COLORS

st.header("Act 1: Uplift Modeling Works")
st.markdown(
    "Before estimating treatment effects, we verify the experiment is sound. "
    "Then we fit four meta-learners (S, T, X, R) to estimate CATE: how much does being "
    "assigned to treatment group increase a user's conversion probability, conditional on their features?"
)

with st.expander("Intro to Uplift Models"):
    st.markdown(
        "Uplift modeling is a machine learning technique that predicts the incremental impact "
        "of an action (in this case, an ad campaign) on an individual's behavior.\n\n"
        "A more detailed explanation of meta-learners and uplift models is in the "
        "[appendix](https://github.com/kuochh/criteo-ab-testing-uplift/blob/main/notebooks/appendix_learners.ipynb)."
    )

# ── 1.1 Randomization quality ──────────────────────────────────────────
st.divider()
st.subheader("1.1 Randomization Check")
st.markdown(
    "Standardized Mean Differences (SMD) compare feature distributions across groups. "
    "An SMD below |0.1| indicates the groups are balanced on that feature."
)

cov_bal = load("covariate_balance.csv")
comparison = st.selectbox(
    "* Select Comparison",
    cov_bal["comparison"].unique(),
    index=0,
)
cov_sub = cov_bal[
    (cov_bal["comparison"] == comparison)
    & (cov_bal["variable"].str.startswith("f"))
].copy()

fig_smd = go.Figure()
fig_smd.add_bar(
    x=cov_sub["variable"],
    y=cov_sub["smd"],
    marker_color=[COLORS["teal"] if abs(v) < 0.1 else COLORS["rose"] for v in cov_sub["smd"]],
)
fig_smd.add_hline(y=0.1, line_dash="dash", line_color=COLORS["rose"], annotation_text="SMD = 0.1")
fig_smd.add_hline(y=-0.1, line_dash="dash", line_color=COLORS["rose"])
max_abs_smd = cov_sub["smd"].abs().max()
fig_smd.update_layout(
    **PLOTLY_LAYOUT,  # pyright: ignore[reportArgumentType]
    title=f"Standardized Mean Differences by Feature (max |SMD| = {max_abs_smd:.3f})",
    xaxis_title="Feature",
    yaxis_title="SMD",
    showlegend=False,
)
st.plotly_chart(fig_smd, width="stretch")

n_large = (cov_sub["smd"].abs() > 0.1).sum()
if comparison == "Treatment vs Control":
    st.success("All SMDs < |0.1|. Randomization is valid: treatment and control groups are balanced.")
elif comparison == "Non-exposed vs Control":
    st.success(
        "All SMDs < |0.1|. Non-exposed treated users look similar to control users, "
        "which is expected since neither group saw ads."
    )
elif n_large > 0:
    st.warning(
        f"{n_large} features have SMD > |0.1|. This reflects non-random compliance, not randomization failure. "
        "Like a vaccine trial: assignment is random, but who actually gets the shot depends on individual behavior."
    )

st.markdown(
    "The table below summarizes the maximum |SMD| for each comparison. "
    "The contrast between Treatment vs Control (balanced) and the exposure-based "
    "comparisons (imbalanced) confirms that randomization is clean, but compliance is selective."
)

# Compute max |SMD| per comparison
max_smd_rows = []
interp_map = {
    "Treatment vs Control": "Balanced. Randomization is valid.",
    "Exposed vs Control": "Imbalanced. Compliance is selective, not random.",
    "Non-exposed vs Control": "Balanced. Neither group saw ads.",
    "Exposed vs Non-exposed": "Imbalanced. Same compliance effect.",
}
for comp in cov_bal["comparison"].unique():
    sub = cov_bal[(cov_bal["comparison"] == comp) & (cov_bal["variable"].str.startswith("f"))]
    max_smd_rows.append({
        "Comparison": comp,
        "Max |SMD|": f"{sub['smd'].abs().max():.4f}",
        "Interpretation": interp_map.get(comp, ""),
    })
st.table(pd.DataFrame(max_smd_rows))

# ── 1.2 Non-compliance ─────────────────────────────────────────────────
st.divider()
st.subheader("1.2 The Non-Compliance Problem")
st.markdown(
    "While treatment assignment is randomized, actual ad exposure is not. "
    "Only 3.6% of assigned users actually see ads while 96.4% are assigned but never exposed."
)

overview = load("dataset_overview.csv").iloc[0]
compliance = load("compliance_stats.csv")
itt = load("itt_effects.csv")

col1, col2, col3, col4 = st.columns(4)
with col1:
    styled_metric("Total Users", f"{overview['total_users']:,.0f}", "Criteo Uplift Dataset")
with col2:
    styled_metric("Treatment Rate", f"{overview['treatment_rate']:.1%}", "randomized assignment")
with col3:
    styled_metric("Exposure Rate", f"{overview['exposure_rate_treated']:.1%}", "of treated users")
with col4:
    styled_metric("Non-Compliance", f"{overview['noncompliance_rate']:.1%}", "assigned but not exposed")

st.markdown("")

col1, col2 = st.columns(2)
with col1:
    conv_itt = itt[itt["outcome"] == "conversion"].iloc[0]
    styled_metric(
        "Conversion ITT Effect",
        f"+{conv_itt['itt_effect_pp']:.2f}pp",
        f"Treated: {conv_itt['treated_rate']:.4%} vs Control: {conv_itt['control_rate']:.4%}",
    )
with col2:
    visit_itt = itt[itt["outcome"] == "visit"].iloc[0]
    styled_metric(
        "Visit ITT Effect",
        f"+{visit_itt['itt_effect_pp']:.2f}pp",
        f"Treated: {visit_itt['treated_rate']:.2%} vs Control: {visit_itt['control_rate']:.2%}",
    )
st.markdown("")
st.markdown(
    "The ITT effects are statistically significant but small. This is not because the ad does not work. "
    "It is because 96.4% of treated users never saw it. The signal is buried under non-compliance."
)

# ── 1.3 Meta-learner comparison ────────────────────────────────────────
st.divider()
st.subheader("1.3 Meta-Learner Comparison")
st.markdown(
    "Four meta-learners (S, T, X, R) each estimate CATE: the predicted difference in conversion "
    "probability between treatment and control for each user, conditional on their features. "
    "Each learner uses a different strategy to handle treatment/control imbalance and model "
    "specification, but they all estimate the same CATE.\n\n"
    "Qini curves show cumulative uplift when users are targeted in order of predicted CATE. "
    "A curve above the diagonal (random targeting) means the model identifies high-value users "
    "better than chance."
)

qini_scores = load("qini_scores.csv")
qini_curves = load("qini_curves_sample.csv")
itt_test = load("itt_test.csv").iloc[0]
n_test = itt_test["n_test"]

all_learner_options = ["S-Learner", "T-Learner", "X-Learner", "R-Learner"]
selected_learners = st.multiselect(
    "* Select Learners",
    all_learner_options,
    default=all_learner_options,
)

fig_qini = go.Figure()

# Random baseline
fig_qini.add_trace(go.Scatter(
    x=[0, 100],
    y=[0, itt_test["itt_test"] * n_test],
    mode="lines",
    name="Random",
    line=dict(color=COLORS["slate"], dash="dash"),
))

for learner in selected_learners:
    ldata = qini_curves[qini_curves["learner"] == learner]
    score = qini_scores[qini_scores["learner"] == learner].iloc[0]
    fig_qini.add_trace(go.Scatter(
        x=ldata["fraction_targeted"] / n_test * 100,
        y=ldata["cumulative_uplift"],
        mode="lines",
        name=f"{learner} (Qini={score['qini_score']:.4f})",
        line=dict(color=LEARNER_COLORS.get(learner, COLORS["teal"])),
    ))

fig_qini.update_layout(
    **PLOTLY_LAYOUT,  # pyright: ignore[reportArgumentType]
    title="Qini Curves: Cumulative Uplift by Fraction Targeted",
    xaxis_title="% of Users Targeted",
    yaxis_title="Cumulative Additional Conversions",
)
st.plotly_chart(fig_qini, width="stretch")

# Qini scores table
st.dataframe(
    qini_scores.style.format({"qini_score": "{:.4f}", "auuc_score": "{:.6f}"}),
    width="stretch",
    hide_index=True,
)

st.markdown(
    "All four learners outperform random targeting. Performance differences are small and "
    "concentrated in the lower portion of the distribution, where the signal is weakest. "
    "The top 5 to 10% of users look similar across learners.\n\n"
    "X-Learner edges out slightly overall (Qini: 0.1753) and is used as the CATE baseline "
    "for the rest of this analysis.\n\n"
    "One important caveat: these models estimate the effect of *assignment*, not the effect "
    "of *seeing the ad*. The non-compliance problem from 1.2 has not been solved. CATE tells "
    "us who benefits from being entered into ad campaign, which is useful, but it does not "
    "tell us who responds to the ad itself."
)

# ── 1.4 Targeting efficiency ───────────────────────────────────────────
st.divider()
st.subheader("1.4 Targeting Efficiency by Decile")
st.markdown(
    "The Qini curves showed overall ranking performance. Now we zoom into deciles to see "
    "where the uplift concentrates. Users are sorted by predicted CATE and divided into ten equal groups."
)

decile = load("decile_uplift.csv")
all_learners = list(decile["learner"].unique())
selected_learners_dec = st.multiselect(
    "* Select Learners",
    all_learners,
    default=all_learners,
    key="decile_learner_multiselect",
)

decile_order = list(reversed(decile["decile_label"].unique().tolist()))

fig_dec = go.Figure()
for learner in selected_learners_dec:
    ldata = decile[decile["learner"] == learner]
    fig_dec.add_bar(
        x=ldata["decile_label"],
        y=ldata["uplift"] * 100,
        marker_color=LEARNER_COLORS.get(learner, COLORS["teal"]),
        name=learner,
    )
fig_dec.add_hline(
    y=itt_test["itt_test"] * 100,
    line_dash="dash",
    line_color=COLORS["amber"],
    annotation_text=f"ITT = {itt_test['itt_test']*100:.3f}pp",
)
fig_dec.update_layout(
    **PLOTLY_LAYOUT,  # pyright: ignore[reportArgumentType]
    title="Marginal Uplift by CATE Decile",
    xaxis_title="CATE Decile",
    yaxis_title="Uplift (pp)",
    barmode="group",
    xaxis=dict(categoryorder="array", categoryarray=decile_order),
)
st.plotly_chart(fig_dec, width="stretch")

st.markdown(
    "The top decile shows uplift well above the ITT baseline, while all other deciles fall "
    "short of ITT and the bottom deciles hover near zero. Across learners, the pattern is "
    "consistent: most of the identifiable uplift concentrates in the top 10 to 20% of users."
)

# ── Summary ────────────────────────────────────────────────────────────
st.divider()
st.subheader("Summary")

st.markdown("""
<div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 1.2rem 1.5rem; margin: 0.5rem 0;">
<strong>Key Takeaways:</strong><br><br>
1. <strong>Randomization is valid.</strong> Treatment and control groups are balanced on all observed features.<br>
2. <strong>Non-compliance is severe.</strong> Only 3.6% of treated users actually see ads, diluting the ITT effect to near-zero.<br>
3. <strong>CATE models work.</strong> Meta-learners successfully identify who benefits from treatment assignment. Differences between learners are small.<br>
4. <strong>But CATE has a blind spot.</strong> Most analyses stop here. CATE ranks users, targeting works. But 96.4% non-compliance suggests there is more to the story.
</div>
""", unsafe_allow_html=True)
