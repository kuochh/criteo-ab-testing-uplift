# Uplift Modeling & Causal Decomposition: Criteo Ad Campaign

[![Live App](https://img.shields.io/badge/Live_App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://criteo-ab-testing.streamlit.app)

## Summary

This project applies uplift modeling and instrumental variable decomposition to Criteo's randomized ad campaign (13.9M users, 85/15 treatment/control). Only 3.6% of treated users actually see ads. With 96.4% non-compliance, standard CATE scores cannot distinguish reachable users from responsive ones.

Using instrumental variables, CATE decomposes into deliverability P(E) and responsiveness LATE(X). The result is two scores per user instead of one. A user with high reachability but low responsiveness needs a minimum bid. A user with low reachability but high responsiveness is worth paying a premium for. CATE treats them identically. The decomposition trades 10% ranking performance for the ability to act on why each user scores high, not just that they do.

See the [Reachability-Responsiveness Map](https://criteo-ab-testing.streamlit.app/act3#3-4-the-reachability-responsiveness-map) in the live app for an interactive version with adjustable thresholds.

## Key Numbers At a Glance

**Experiment:**

| | |
|---|---|
| Users | 13.9M (85% treatment, 15% control) |
| Non-compliance | 96.4% of treated users never see ads |
| Conversion (exposed treated) | 5.38% |
| Conversion (non-exposed treated) | 0.12% |
| Conversion (control) | 0.19% |

**Results:**

| | |
|---|---|
| X-Learner Qini | 0.1753 |
| Top decile conversion | 0.96% vs 0.19% control (5x) |
| Mean LATE | 2.86pp, range -7pp to 48pp (24x the ITT) |
| P(E) alone | Qini 0.1668 (95% of CATE) |
| P(E) x LATE(X) | Qini 0.1580 (90% of CATE) |


## Project Structure

| Path | Description |
|------|-------------|
| `streamlit/pages/prologue.py` | Context: experiment design & non-compliance problem |
| `streamlit/pages/act1.py` | Standard uplift modeling — meta-learner comparison, decile analysis |
| `streamlit/pages/act2.py` | CATE's blind spot — correlation with exposure probability |
| `streamlit/pages/act3.py` | IV decomposition, LATE estimation & bidding strategy quadrant |
| `notebooks/01_eda.ipynb` | Exploratory data analysis & randomization validation |
| `notebooks/02_uplift_cate.ipynb` | CATE estimation with EconML meta-learners |
| `notebooks/03_exposure_late.ipynb` | IV decomposition into P(E) x LATE(X) |
| `notebooks/appendix_learners.ipynb` | Hand-coded S/T/X/R-learner implementations |

## Tech Stack

Polars · XGBoost · EconML · sklift · Streamlit · Plotly

---

**[Live App](https://criteo-ab-testing.streamlit.app)** · [Criteo Uplift Dataset](https://ailab.criteo.com/ressources/) · MIT License
