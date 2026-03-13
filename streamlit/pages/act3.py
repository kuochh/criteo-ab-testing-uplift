import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from helpers import load, styled_metric, COLORS, PLOTLY_LAYOUT, STRATEGY_COLORS, STRATEGY_RENAME

st.header("Act 3: Decomposing the Signal")
st.markdown(
    "Random treatment assignment gives us an instrument. Users are randomly made eligible for "
    "ad auctions, but whether they actually see ads is not random. This separation lets us "
    "estimate two things independently: how likely is delivery (reachability), and how effective "
    "is the ad when delivered (responsiveness). Once we have **both** scores per user, we can map every user into a bidding strategy."
)

# ── 3.1 IV framework ──────────────────────────────────────────────────
st.divider()
st.subheader("3.1 Estimating Responsiveness")

st.markdown(
    "Act 2 showed that CATE blends reachability P(E) and responsiveness LATE into one score. "
    "Estimating P(E) is straightforward: an XGBoost classifier trained on the treatment group "
    "predicts who actually sees ads. The harder question is LATE: how much does a user's "
    "conversion probability increase when they actually see the ad?\n\n"
    "We cannot just compare exposed vs. non-exposed users. As section 1.1 showed, those groups "
    "differ systematically. Exposure is not random, so the comparison is confounded.\n\n"
    "This is where instrumental variables come in. Random treatment assignment is our instrument: "
    "it affects exposure but has no direct effect on conversion. This lets us isolate the causal "
    "effect of actually seeing the ad."
)

st.latex(r"\text{CATE}(X) = \underbrace{P(E \mid T{=}1, X)}_{\substack{\text{Reachability} \\ \text{(XGBoost classifier)}}} \times \underbrace{\text{LATE}(X)}_{\substack{\text{Responsiveness} \\ \text{(EconML DMLIV)}}}")

st.info(
    "**CATE** is the overall effect of campaign assignment on conversion. "
    "It decomposes into two components:  \n"
    "- **Reachability** P(E): how likely this user is to actually "
    "see the ad when assigned.  \n"
    "- **Responsiveness** LATE: how much seeing the ad increases this "
    "user's conversion probability."
)

st.markdown(
    "We estimate LATE using EconML's DMLIV (Double Machine Learning IV), which uses treatment "
    "assignment as an instrument for actual exposure. The exclusion restriction is credible: "
    "control users are never entered into auctions and never see ads, so assignment affects "
    "conversion only through exposure."
)

iv_diag = load("iv_diagnostics.csv").iloc[0]

# ── 3.2 Correlation evidence ──────────────────────────────────────────
st.divider()
st.subheader("3.2 Two Distinct Dimensions")

st.markdown(
    "If the decomposition is working, P(E) and LATE should capture genuinely different aspects "
    "of user value. The correlation structure confirms this."
)

decomp = load("decomposition_correlations.csv").iloc[0]
corr_data = load("pe_late_sample.csv")

pairs = [
    ("p_exposure", "x_cate", "P(E)", "CATE", decomp["cate_vs_pe"], COLORS["teal"]),
    ("late_x", "x_cate", "LATE(X)", "CATE", decomp["cate_vs_late"], COLORS["amber"]),
    ("p_exposure", "late_x", "P(E)", "LATE(X)", decomp["pe_vs_late"], COLORS["rose"]),
]

col1, col2, col3 = st.columns(3)
for col, (xcol, ycol, xlabel, ylabel, corr, color) in zip([col1, col2, col3], pairs):
    with col:
        fig_corr = px.scatter(
            corr_data.sample(n=2000, random_state=42),
            x=xcol, y=ycol, opacity=0.25,
            color_discrete_sequence=[color],
        )
        fig_corr.update_layout(
            **{**PLOTLY_LAYOUT, "margin": dict(l=40, r=10, t=40, b=40)},  # pyright: ignore[reportArgumentType]
            title=f"r = {corr:.3f}",
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            showlegend=False,
            height=350,
        )
        st.plotly_chart(fig_corr, width="stretch")

col1, col2, col3 = st.columns(3)
with col1:
    styled_metric("CATE vs P(E)", f"{decomp['cate_vs_pe']:.3f}", "correlation")
with col2:
    styled_metric("CATE vs LATE", f"{decomp['cate_vs_late']:.3f}", "correlation")
with col3:
    styled_metric("P(E) vs LATE", f"{decomp['pe_vs_late']:.3f}", "distinct dimensions")

st.markdown(
    f"P(E) and LATE are nearly uncorrelated ({decomp['pe_vs_late']:.2f}), confirming they capture "
    f"different dimensions. CATE correlates with both ({decomp['cate_vs_pe']:.2f} with P(E), "
    f"{decomp['cate_vs_late']:.2f} with LATE) without distinguishing them."
)

# ── 3.3 Performance comparison ─────────────────────────────────────────
st.divider()
st.subheader("3.3 What Do We Gain and Lose from Decomposition?")

st.markdown(
    "CATE gives us one score per user. We can rank by it and target the top. But what if we "
    "rank users by the decomposed components instead? Reachability alone, or the full "
    "Reachability x Response product? We lose some ranking accuracy. But we gain two separate "
    "scores per user, which tell us not just who to target but why they score high."
)

perf = load("performance_comparison.csv")
perf["strategy"] = perf["strategy"].map(STRATEGY_RENAME).fillna(perf["strategy"])
uplift_curves = load("uplift_curves_comparison_sample.csv")
uplift_curves["strategy"] = uplift_curves["strategy"].map(STRATEGY_RENAME).fillna(uplift_curves["strategy"])
itt_test = load("itt_test.csv").iloc[0]
n_test = itt_test["n_test"]

all_strategy_options = list(perf["strategy"])
selected_strategies = st.multiselect(
    "* Select strategies",
    all_strategy_options,
    default=all_strategy_options,
    key="strategy_multiselect",
)

# Qini curves
fig_uplift = go.Figure()

# Random baseline
fig_uplift.add_trace(go.Scatter(
    x=[0, 100],
    y=[0, itt_test["itt_test"] * n_test],
    mode="lines",
    name="Random",
    line=dict(color=COLORS["slate"], dash="dash"),
))

for _, row in perf.iterrows():
    strategy = row["strategy"]
    if strategy not in selected_strategies:
        continue
    sdata = uplift_curves[uplift_curves["strategy"] == strategy]
    fig_uplift.add_trace(go.Scatter(
        x=sdata["fraction_targeted"] / n_test * 100,
        y=sdata["cumulative_uplift"],
        mode="lines",
        name=f"{strategy} ({row['pct_of_cate_qini']:.0f}%)",
        line=dict(color=STRATEGY_COLORS.get(strategy, COLORS["teal"])),
    ))

fig_uplift.update_layout(
    **PLOTLY_LAYOUT,  # pyright: ignore[reportArgumentType]
    title="Uplift Curves: CATE vs Decomposed Strategies",
    xaxis_title="% of Users Targeted",
    yaxis_title="Cumulative Additional Conversions",
)
st.plotly_chart(fig_uplift, width="stretch")

st.markdown(
    "The uplift curves show aggregate ranking performance. The decile view below breaks down "
    "the same comparison by ranking decile."
)

# Strategy decile comparison
strategy_decile = load("strategy_decile_uplift_sample.csv")
strategy_decile["strategy"] = strategy_decile["strategy"].map(STRATEGY_RENAME).fillna(strategy_decile["strategy"])
decile_order = list(reversed(strategy_decile["decile_label"].unique().tolist()))
all_strategies = list(strategy_decile["strategy"].unique())
selected_strategies_dec = st.multiselect(
    "* Select strategies",
    all_strategies,
    default=all_strategies,
    key="strategy_decile_multiselect",
)

fig_sdec = go.Figure()
for strategy in selected_strategies_dec:
    sdata = strategy_decile[strategy_decile["strategy"] == strategy]
    fig_sdec.add_bar(
        x=sdata["decile_label"],
        y=sdata["marginal_uplift"],
        marker_color=STRATEGY_COLORS.get(strategy, COLORS["teal"]),
        name=strategy,
    )
# ITT baseline: total uplift / 10 = expected per decile under random targeting
first_strategy = strategy_decile[strategy_decile["strategy"] == all_strategies[0]]
itt_per_decile = first_strategy["marginal_uplift"].sum() / 10
fig_sdec.add_hline(
    y=itt_per_decile,
    line_dash="dash",
    line_color=COLORS["amber"],
    annotation_text=f"ITT = {itt_per_decile:.0f}",
)
fig_sdec.update_layout(
    **PLOTLY_LAYOUT,  # pyright: ignore[reportArgumentType]
    title="Marginal Uplift by Decile",
    xaxis_title="Decile (ranked by strategy score)",
    yaxis_title="Marginal Uplift (additional conversions)",
    barmode="group",
    xaxis=dict(categoryorder="array", categoryarray=decile_order),
)
st.plotly_chart(fig_sdec, width="stretch")

# Performance bars
fig_perf = go.Figure()
fig_perf.add_bar(
    x=perf["strategy"],
    y=perf["pct_of_cate_qini"],
    marker_color=[STRATEGY_COLORS.get(s, COLORS["teal"]) for s in perf["strategy"]],
    text=[f"{v:.1f}%" for v in perf["pct_of_cate_qini"]],
    textposition="outside",
)
fig_perf.update_layout(
    **PLOTLY_LAYOUT,  # pyright: ignore[reportArgumentType]
    title="Qini Score as % of X-Learner CATE",
    yaxis_title="% of CATE Qini",
    yaxis_range=[0, 110],
    showlegend=False,
)
st.plotly_chart(fig_perf, width="stretch")

st.markdown(
    f"Reachability alone achieves {perf.iloc[1]['pct_of_cate_qini']:.0f}% of CATE performance. "
    "This confirms what Act 2 showed: reachability is correlated with CATE. "
    f"The full decomposition (Reachability x Response) reaches {perf.iloc[2]['pct_of_cate_qini']:.0f}%."
)

st.info(
    "**What Do We Gain or Lose From Decomposition:** This 10% cost is worst-case: it only measures "
    "ranking loss. What we gain is the ability to treat users differently based on why they score "
    "high. Next section shows what that could look like in practice."
)

# ── 3.4 P(E) vs LATE quadrant scatter ──────────────────────────────────
st.divider()
st.subheader("3.4 The Reachability-Responsiveness Map")
st.markdown(
    "This is the result. Every user now has two scores instead of one: how reachable they are, "
    "and how responsive they are when reached. In previous section, we showed the full decomposition "
    "retains 90% of CATE's ranking performance. The 10% cost buys us this: a map where position "
    "suggests different bidding strategies, rather than a single ranked list that treats all "
    "high-scoring users the same." 
)

st.markdown(
    "The optimal P(E) and LATE thresholds should be determined by business goals: maximize conversions, minimize cost per acquisition, "
    "or hit a target ROI. The dataset does not include bid prices or conversion values, so we cannot compute those thresholds here."
)

st.markdown(
    "But the point is structural: instead of one CATE score that ranks users identically regardless "
    "of mechanism, we now have two dimensions that allow different actions for different types of users."
)

st.markdown("""
<table>
<thead>
<tr><th>Quadrant</th><th>Reachability</th><th>Responsiveness</th><th style="white-space:nowrap">Action</th><th>Why</th></tr>
</thead>
<tbody>
<tr><td><strong>Ideal</strong></td><td>High</td><td>High</td><td style="white-space:nowrap">Standard bid</td><td>Likely to see the ad and likely to convert. These users drive the bulk of campaign ROI.</td></tr>
<tr><td><strong>Bid high</strong></td><td>Low</td><td>High</td><td style="white-space:nowrap">Aggressive bid</td><td>The hidden value that CATE cannot find. These users rarely see ads at standard bid levels, but convert at high rates when reached. A CATE-only system would underinvest here.</td></tr>
<tr><td><strong>Bid low</strong></td><td>High</td><td>Low</td><td style="white-space:nowrap">Minimum bid</td><td>Easy impressions that don't convert. High reachability inflates their CATE scores, but actual response rate is low. Overbidding wastes budget.</td></tr>
<tr><td><strong>Skip</strong></td><td>Low</td><td>Low</td><td style="white-space:nowrap">Don't bid</td><td>Hard to reach and unlikely to convert even if reached. Lowest expected return of any segment.</td></tr>
</tbody>
</table>
""", unsafe_allow_html=True)

scatter_sample = load("pe_late_sample.csv")

st.markdown(r"**\* Select Optimal Threshold**")
col1, col2 = st.columns(2)
with col1:
    pe_threshold = st.slider(
        "P(E) threshold",
        min_value=0.0,
        max_value=0.9,
        value=0.15,
        step=0.01,
    )
with col2:
    late_threshold = st.slider(
        "LATE threshold",
        min_value=0.0,
        max_value=0.10,
        value=0.03,
        step=0.005,
    )

# Assign quadrants
scatter_sample = scatter_sample.copy()
conditions = [
    (scatter_sample["p_exposure"] >= pe_threshold) & (scatter_sample["late_x"] >= late_threshold),
    (scatter_sample["p_exposure"] < pe_threshold) & (scatter_sample["late_x"] >= late_threshold),
    (scatter_sample["p_exposure"] >= pe_threshold) & (scatter_sample["late_x"] < late_threshold),
    (scatter_sample["p_exposure"] < pe_threshold) & (scatter_sample["late_x"] < late_threshold),
]
labels = [
    "Ideal",
    "Bid High",
    "Bid Low",
    "Skip",
]
quadrant_colors = [COLORS["emerald"], COLORS["amber"], COLORS["cyan"], COLORS["slate"]]

scatter_sample["quadrant"] = "unassigned"
for cond, label in zip(conditions, labels):
    scatter_sample.loc[cond, "quadrant"] = label

fig_quad = px.scatter(
    scatter_sample,
    x="p_exposure",
    y="late_x",
    color="quadrant",
    color_discrete_map=dict(zip(labels, quadrant_colors)),
    opacity=0.4,
    category_orders={"quadrant": labels},
)
fig_quad.add_vline(x=pe_threshold, line_dash="dash", line_color="#94a3b8")
fig_quad.add_hline(y=late_threshold, line_dash="dash", line_color="#94a3b8")
fig_quad.update_layout(
    **PLOTLY_LAYOUT,  # pyright: ignore[reportArgumentType]
    title="User Segments: Reachability vs Responsiveness",
    xaxis_title="P(Exposure | T=1, X) — Reachability",
    yaxis_title="LATE(X) — Responsiveness",
    legend_title="Segment",
)
st.plotly_chart(fig_quad, width="stretch")

# Quadrant counts
quad_counts = scatter_sample["quadrant"].value_counts()
total_sample = len(scatter_sample)

cols = st.columns(4)
for i, (label, color) in enumerate(zip(labels, quadrant_colors)):
    count = quad_counts.get(label, 0)
    with cols[i]:
        short_label = label.split("(")[1].rstrip(")") if "(" in label else label
        styled_metric(short_label, f"{count/total_sample:.1%}", f"n={count:,}")

st.markdown("")

st.info(
    "**The Key Insight:** a CATE-only system treats 'bid high' and 'bid low' users the same if their "
    "CATE scores happen to match. The decomposition reveals they need opposite strategies. More "
    "importantly, the 'bid high' quadrant represents users that a CATE-only system systematically "
    "undervalues. These users are hard to reach, so they rarely get exposed, so their CATE is low. "
    "But they are the most responsive users in the campaign. Without decomposition, budget that "
    "should go toward winning these impressions gets spent on easy-to-reach users who are unlikely "
    "to convert."
)



# ── 3.5 Summary ────────────────────────────────────────────────────────
st.divider()
st.subheader("Summary")

# exp_model = load("exposure_model_details.csv").iloc[0]

# col1, col2, col3 = st.columns(3)
# with col1:
#     styled_metric("P(E) Model AUC", f"{exp_model['test_auc']:.3f}", "XGBoost classifier predicting ad exposure")
# with col2:
#     styled_metric("CATE vs P(E) Correlation", f"{iv_diag['cate_exposure_corr']:.3f}", "reachability dominates")
# with col3:
#     styled_metric("P(E) vs LATE Correlation", f"{iv_diag['exposure_late_corr']:.3f}", "distinct dimensions")

st.markdown("")

st.markdown("""
<div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 1.2rem 1.5rem; margin: 0.5rem 0;">
<strong>Key Takeaways:</strong><br><br>
1. <strong>Standard uplift modeling works.</strong> CATE meta-learners correctly identify who benefits from ad assignment.<br>
2. <strong>But CATE conflates two mechanisms.</strong> With high non-compliance, reachability and responsiveness are blended into one score.<br>
3. <strong>IV decomposition separates them.</strong> Using treatment assignment as an instrument, we recover distinct reachability and responsiveness scores for each user.<br>
4. <strong>The 10% Qini cost is a feature, not a bug.</strong> It trades ranking performance for operational control: the ability to bid high on responsive users, bid low on reachable ones, and skip the rest.
</div>
""", unsafe_allow_html=True)
