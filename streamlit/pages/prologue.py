import streamlit as st

st.header("Prologue")
st.markdown(
    "This analysis uses the [Criteo Uplift Dataset](https://huggingface.co/datasets/criteo/criteo-uplift), "
    "a real-world randomized controlled trial "
    "covering 13.9 million users in an online advertising campaign. The dataset contains "
    "anonymized features, treatment and outcome variables."
)

st.divider()

st.subheader("The Experiment")
st.markdown("""An online ad experiment randomly assigned 13.9M users into two groups:

- **Treatment (85%):** Users became *eligible* for ad auctions. If the company won the auction, the user saw an ad.
- **Control (15%):** Users were never entered into auctions and never saw ads.""")

st.info(
    "**The Key Distinction:** Treatment means *eligibility*, not *exposure*. "
    "Being assigned to the treatment group does not guarantee seeing an ad. "
    "Treated users might never see an ad because of auction losses, ad blockers, "
    "inventory limits, or technical failures."
)

st.subheader("Treatment vs. Exposure")
st.markdown("""- **Treatment** = assigned to be eligible for ad auctions (randomized)
- **Exposure** = actually seeing the ad (not randomized)
- **Outcome** = binary conversion (1 = converted, 0 = did not)

Only **3.6%** of treated users actually saw an ad. The other 96.4% were assigned to treatment group but never exposed. This gap between what was assigned and what actually happened is called *non-compliance*.

Non-compliance changes what standard models can estimate. Uplift models estimate the Conditional Average Treatment Effect (CATE): how much does being *treated* increase a user's conversion probability? But in this experiment, "treated" means assigned to the campaign, not shown an ad. CATE estimates the effect of campaign assignment, not the effect of actually seeing the ad.

And since 96.4% of treated users had the exact same experience as control users (neither group saw an ad), the assignment effect is expected to be small. Not because the ad doesn't work, but because it averages the ad's impact across millions of users who were never exposed to it.""")

with st.expander("Background: Non-Compliance in Experiments"):
    st.markdown(
        "This is a common problem in real-world experiments. In a vaccine trial, researchers "
        "randomly assign people to receive a vaccine or a placebo. But some people assigned to "
        "the vaccine group never show up for their appointment. Comparing outcomes between the "
        "two groups as assigned gives us the effect of being *offered* the vaccine, not the "
        "effect of *getting vaccinated*. When most of the treatment group skips the appointment, "
        "the measured effect will be small even if the vaccine works well. The same logic applies here."
    )

st.subheader("The Problem")
st.markdown("""Suppose we build uplift models and they work: we can rank users by predicted CATE score and concentrate budget on those who benefit most. That solves the targeting problem.

But with a 96.4% non-compliance rate, what is the CATE score actually picking up? Most treated users never saw the ad, so a high CATE cannot simply mean "this user responds well to ads." It could mean the user is easy to reach (likely to actually see the ad if targeted), or it could mean the user is genuinely responsive (likely to convert if they see it), or some combination. CATE blends these two signals into a single number without distinguishing them.

If we can separate reachability from responsiveness, we gain more than just a better understanding of the score. We get actionable signals: which users are worth paying a premium to reach, which ones will convert without extra effort, and which ones are not worth the ad spend at all.""")

st.subheader("What This Analysis Covers")
st.markdown(
    "1. **Act 1** builds standard uplift models and confirms they work for ranking users by predicted treatment effect.\n"
    "2. **Act 2** decomposes CATE into reachability and responsiveness, estimates reachability, and shows it correlates with CATE.\n"
    "3. **Act 3** estimates responsiveness using instrumental variables, completing the decomposition into two actionable scores per user."
)
st.caption(
    "Scatter plots use a random subsample of 5,000 points for performance. "
    "All statistics and model results are computed on the full dataset."
)
