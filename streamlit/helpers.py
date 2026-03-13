import streamlit as st
import pandas as pd
import os

# ---------------------------------------------------------------------------
# Config & helpers
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

COLORS = {
    "teal": "#0d9488",
    "blue": "#3b82f6",
    "amber": "#f59e0b",
    "rose": "#f43f5e",
    "slate": "#64748b",
    "emerald": "#10b981",
    "violet": "#8b5cf6",
    "cyan": "#06b6d4",
}

LEARNER_COLORS = {
    "S-Learner": COLORS["blue"],
    "T-Learner": COLORS["amber"],
    "X-Learner": COLORS["teal"],
    "R-Learner": COLORS["violet"],
    "Random": COLORS["slate"],
}

STRATEGY_COLORS = {
    "CATE": COLORS["teal"],
    "Reachability": COLORS["amber"],
    "Reachability x Response": COLORS["rose"],
    "Random": COLORS["slate"],
}

STRATEGY_RENAME = {
    "X-learner CATE": "CATE",
    "Exposure P(E|T=1,X)": "Reachability",
    "Expected Uplift P(E)×LATE(X)": "Reachability x Response",
    "Exposure-based": "Reachability",
    "Expected uplift": "Reachability x Response",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e2e8f0"),
    margin=dict(l=60, r=20, t=40, b=60),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)


@st.cache_data
def load(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, name))


def styled_metric(label: str, value: str, delta: str | None = None):
    """Render a metric inside a styled container."""
    delta_html = f'<p style="color:#0d9488;font-size:0.85rem;margin:0">{delta}</p>' if delta else ""
    st.markdown(
        f'<div style="background:#1e293b;border-radius:8px;padding:16px 20px;text-align:center">'
        f'<p style="color:#94a3b8;font-size:0.8rem;margin:0 0 4px 0;text-transform:uppercase;letter-spacing:0.05em">{label}</p>'
        f'<p style="color:#e2e8f0;font-size:1.8rem;font-weight:700;margin:0">{value}</p>'
        f'{delta_html}'
        f'</div>',
        unsafe_allow_html=True,
    )
