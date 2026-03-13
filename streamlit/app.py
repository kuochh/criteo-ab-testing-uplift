import streamlit as st

st.set_page_config(
    page_title="A/B Testing: Uplift Modeling & Causal Decomposition",
    page_icon=":bar_chart:",
    layout="wide",
)

pg = st.navigation([
    st.Page("pages/prologue.py", title="Prologue", default=True),
    st.Page("pages/act1.py", title="Act 1: Uplift Modeling Works"),
    st.Page("pages/act2.py", title="Act 2: What CATE Actually Captures"),
    st.Page("pages/act3.py", title="Act 3: Decomposing the Signal"),
])

with st.sidebar:
    st.title("A/B Testing with Uplift Modeling & Causal Decomposition")
    st.caption("Criteo Uplift Dataset · 13.9M Users")
    st.divider()
    st.markdown("**Chia-Hung Kuo**")
    st.markdown(
        "[GitHub](https://github.com/kuochh/criteo-ab-testing-uplift) · "
        "[Portfolio](https://kuochh.github.io) · "
        "[LinkedIn](https://linkedin.com/in/chia-hung-kuo)"
    )

pg.run()
