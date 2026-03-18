"""Downsample large uplift curve CSVs to ~1000 points per learner/strategy."""
import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
N_POINTS = 1000  # points per learner/strategy


def downsample_curve(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Downsample each group to N_POINTS evenly-spaced percentile points."""
    sampled = []
    for name, group in df.groupby(group_col, sort=False):
        group = group.sort_values("fraction_targeted").reset_index(drop=True)
        n = len(group)
        if n <= N_POINTS:
            sampled.append(group)
            continue
        # Evenly spaced indices, always including first and last
        idx = np.unique(np.linspace(0, n - 1, N_POINTS, dtype=int))
        sampled.append(group.iloc[idx])
    return pd.concat(sampled, ignore_index=True)


# 1. Qini curves (4 learners + Random)
qini_path = os.path.join(DATA_DIR, "qini_curves.csv")
if os.path.exists(qini_path):
    print("Loading qini_curves.csv...")
    qini = pd.read_csv(qini_path)
    print(f"  Original: {len(qini):,} rows")
    qini_sampled = downsample_curve(qini, "learner")
    out_path = os.path.join(DATA_DIR, "qini_curves_sample.csv")
    qini_sampled.to_csv(out_path, index=False)
    print(f"  Sampled:  {len(qini_sampled):,} rows -> {out_path}")
else:
    print("Skipping qini_curves.csv (not found, sample already exists)")

# 2. Uplift curves comparison (3 strategies + Random)
uplift_path = os.path.join(DATA_DIR, "uplift_curves_comparison.csv")
if os.path.exists(uplift_path):
    print("Loading uplift_curves_comparison.csv...")
    uplift = pd.read_csv(uplift_path)
    print(f"  Original: {len(uplift):,} rows")
    uplift_sampled = downsample_curve(uplift, "strategy")
    out_path = os.path.join(DATA_DIR, "uplift_curves_comparison_sample.csv")
    uplift_sampled.to_csv(out_path, index=False)
    print(f"  Sampled:  {len(uplift_sampled):,} rows -> {out_path}")
else:
    print("Skipping uplift_curves_comparison.csv (not found, sample already exists)")

# 3. CATE predictions test (section 2.1 histogram)
cate_pred_path = os.path.join(DATA_DIR, "cate_predictions_test.csv")
if os.path.exists(cate_pred_path):
    print("Loading cate_predictions_test.csv...")
    cate_pred = pd.read_csv(cate_pred_path)
    print(f"  Original: {len(cate_pred):,} rows")
    cate_pred_sampled = cate_pred.sample(n=min(10000, len(cate_pred)), random_state=42)
    out_path = os.path.join(DATA_DIR, "cate_predictions_test_sample.csv")
    cate_pred_sampled.to_csv(out_path, index=False)
    print(f"  Sampled:  {len(cate_pred_sampled):,} rows -> {out_path}")
else:
    print("Skipping cate_predictions_test.csv (not found)")

# 4. CATE vs P(E) scatter (section 2.3)
print("Loading cate_vs_pe.csv...")
cate_pe = pd.read_csv(os.path.join(DATA_DIR, "cate_vs_pe.csv"))
print(f"  Original: {len(cate_pe):,} rows")
cate_pe_sampled = cate_pe.sample(n=min(10000, len(cate_pe)), random_state=42)
out_path = os.path.join(DATA_DIR, "cate_vs_pe_sample.csv")
cate_pe_sampled.to_csv(out_path, index=False)
print(f"  Sampled:  {len(cate_pe_sampled):,} rows -> {out_path}")

# 4. P(E) vs LATE scatter (section 3.3/3.4) - filter out negative LATE
print("Loading pe_late_estimates.csv...")
pe_late = pd.read_csv(os.path.join(DATA_DIR, "pe_late_estimates.csv"))
print(f"  Original: {len(pe_late):,} rows")
pe_late_pos = pe_late[pe_late["late_x"] >= 0]
print(f"  After filtering negative LATE: {len(pe_late_pos):,} rows")
pe_late_sampled = pe_late_pos.sample(n=min(10000, len(pe_late_pos)), random_state=42)
out_path = os.path.join(DATA_DIR, "pe_late_sample.csv")
pe_late_sampled.to_csv(out_path, index=False)
print(f"  Sampled:  {len(pe_late_sampled):,} rows -> {out_path}")

# 5. Strategy decile uplift (section 3.2) - derived from uplift curves
print("Computing strategy decile uplift from uplift_curves_comparison_sample.csv...")
uplift_s = pd.read_csv(os.path.join(DATA_DIR, "uplift_curves_comparison_sample.csv"))

# Display name mapping for strategies
display_names = {
    "X-learner CATE": "X-learner CATE",
    "Exposure-based": "Exposure P(E|T=1,X)",
    "Expected uplift": "Expected Uplift P(E)×LATE(X)",
}

# Labels matching 1.4 convention: Bottom 10% (left) to Top 10% (right)
DECILE_LABELS = [
    "Bottom 10%", "2nd Decile", "3rd Decile", "4th Decile", "5th Decile",
    "6th Decile", "7th Decile", "8th Decile", "9th Decile", "Top 10%",
]

decile_rows = []
for strategy_name, group in uplift_s.groupby("strategy", sort=False):
    if strategy_name == "Random":
        continue
    group = group.sort_values("fraction_targeted").reset_index(drop=True)
    max_frac = group["fraction_targeted"].max()
    if max_frac == 0:
        continue
    # Interpolate cumulative uplift at each decile boundary (10%, 20%, ..., 100%)
    pct = group["fraction_targeted"] / max_frac  # normalize to 0-1
    cum = group["cumulative_uplift"].values
    boundaries = np.linspace(0.1, 1.0, 10)
    cum_at_boundary = np.interp(boundaries, pct, cum)
    # Marginal uplift per decile (curve sorts best-first, so index 0 = top decile)
    marginal = np.diff(np.concatenate([[0], cum_at_boundary]))
    # Reverse so index 0 = Bottom 10%, index 9 = Top 10% (matching 1.4 layout)
    marginal = marginal[::-1]
    display = display_names.get(strategy_name, strategy_name)
    for i, m in enumerate(marginal):
        decile_rows.append({
            "strategy": display,
            "decile": i + 1,
            "decile_label": DECILE_LABELS[i],
            "marginal_uplift": float(m),
        })

decile_df = pd.DataFrame(decile_rows)
out_path = os.path.join(DATA_DIR, "strategy_decile_uplift_sample.csv")
decile_df.to_csv(out_path, index=False)
print(f"  Saved: {len(decile_df)} rows -> {out_path}")

print("\nDone.")
