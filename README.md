# A/B Testing with Uplift Modeling and Causal Decomposition: Criteo Ad Campaign Analysis

## Executive Summary

Standard uplift models achieve 0.1753 Qini identifying high-value users (top 10% convert at 5× control), but with severe delivery constraints (96.4% never see ads), CATE's single score cannot separate "who can be reached" with "who responds when reached."

Using instrumental variables, I decompose CATE to reveal why each user scores high: reachability P(E) versus ad responsiveness LATE(X). This enables differentiated bidding targeting users with higher ad responsiveness.

**Result:** 10% performance tradeoff yields actionable optimization — ads boost exposed users to 3.05% conversion vs 0.19% control (16× increase), proving delivery is the constraint worth optimizing.

## Overview

This project analyzes a randomized controlled trial of an advertising campaign using the Criteo dataset (13.9M users). Users were randomly assigned to treatment (85%, eligible for ad auctions) or control (15%, never shown ads). The experiment exhibits severe non-compliance: only 3.6% of treatment-assigned users actually see ads due to auction losses, ad blockers, and technical constraints.

**Experimental Structure:**
```
Random Assignment → Ad Auction → Exposure → Conversion
    (85/15 split)   (Non-compliance)  (3.6%)   (Outcome)
```

## Key Numbers at a Glance

| Metric | Value | Insight |
|--------|-------|---------|
| **Control Group Conversion** | 0.194% | Baseline conversion rate without ad exposure |
| **Non-compliance Rate** | 96.4% | Only 3.6% of assigned users see ads in RCT |
| **X-Learner Qini Coefficient** | 0.1753 | Standard CATE uplift performance (baseline) |
| **Top Decile Uplift** | 0.77pp | Top 10% convert at 0.96% vs 0.19% control—5× increase |
| **Deliverability Dominance** | 95% of CATE | P(Exposure) alone achieves 95% of CATE performance |
| **Full Decomposition Tradeoff** | -10% Qini | Cost of separating P(E) from LATE(X) for strategic insights |
| **LATE vs ITT** | 24× larger | Ads highly effective when delivered (2.86pp vs 0.12pp) |

## What Makes This Different

Most uplift modeling projects implement standard meta-learners and report metrics. This experimentation project:

1. **Identifies actionable limitation** - CATE's single score hides mechanism differences critical for bidding optimization
2. **Quantifies tradeoffs** - 10% ranking performance loss is worst-case baseline; decomposition enables bidding optimization that can exceed this cost through strategic budget allocation
3. **Applies causal inference rigor** - Instrumental variables separate deliverability from responsiveness via LATE
4. **Provides business framework** - Quantified guidance for budget allocation based on mechanism decomposition
5. **Demonstrates critical thinking** - Reports actual findings: delivery dominates, decomposition enables optimization

## Business Impact

**Quantified Performance-Interpretability Tradeoff in Uplift Modeling:**

| Approach | Qini | Performance | Business Value |
|----------|------|-------------|----------------|
| **CATE (X-Learner)** | 0.1753 | 100% (baseline) | Maximize conversions (black box) |
| **P(E)×LATE(X) Decomposition** | 0.1580 | 90% | Strategic bidding optimization |

**Programmatic Advertising Bidding Strategy:**

CATE identifies high-value users but provides no guidance on bid levels. Decomposition reveals *why* each user scores high:

**Two users with identical CATE score (0.004):**
- **User A**: P(E)=0.80, LATE=0.005 → Easy to reach, modest response → **Bid the SAME or LOW** 
- **User B**: P(E)=0.20, LATE=0.020 → Hard to reach, strong response → **Bid HIGH** (worth the cost)

CATE treats these identically. Decomposition enables budget tiering impossible with a single score.

**Key Insight:** With 96.4% non-compliance, delivery is the bottleneck. Decomposition proves deliverability contributes 95% of CATE performance—focus budget on winning auctions for high-LATE users, not just high-CATE users.

## Technical Implementation

### The Causal Decomposition Framework

**Standard CATE Uplift Modeling (Black Box):**
```
Treatment Assignment ────────────→ Conversion
     CATE(X) = E[Y|T=1,X] - E[Y|T=0,X]
```

**Decomposed Approach via Instrumental Variables:**
```
Treatment Assignment ──→ Exposure ──→ Conversion
     P(E|T=1,X)          LATE(X)
   "Deliverability"   "Responsiveness"

   CATE(X) ≈ P(Exposure|T=1,X) × LATE(X)
```

### Core Methodology

- **Non-compliance in experimentation:** Random assignment ≠ actual ad exposure (3.6% compliance in A/B test)
- **Instrumental variables strategy:** Use randomized treatment assignment as instrument for exposure
- **LATE estimation:** Local average treatment effect for compliers (users who see ads when assigned)
- **Causal decomposition:** Separate P(E|T=1,X) from LATE(X) to reveal uplift mechanisms
- **Validation:** Compare uplift curves, AUUC, and correlation analysis across meta-learners

## Project Structure

**Analysis Notebooks:**
- **01_eda.ipynb** - Exploratory data analysis establishing non-compliance problem and A/B test randomization validation
- **02_uplift_cate.ipynb** - Standard CATE estimation using EconML meta-learners + discovery of CATE-exposure correlation
- **03_exposure_late.ipynb** - **Main contribution:** Instrumental variables decomposition into P(Exposure) × LATE(X)
- **appendix_learners.ipynb** - Hand-coded S/T/X/R-learners demonstrating causal inference algorithm understanding

**Key Visualizations:**
- `cate_exposure.png` - Correlation between CATE predictions and exposure probability in uplift modeling
- `cate_late_uplift.png` - Causal decomposition performance comparison
- `uplift_decile.png` - Decile analysis showing deliverability dominance in heterogeneous treatment effects
- `uplift_learners.png` - Meta-learner comparison (X-learner optimal for imbalanced treatment in A/B testing)

## Technical Skills Demonstrated

- **Causal inference:** CATE estimation, instrumental variables, LATE, exclusion restrictions, heterogeneous treatment effects
- **A/B testing & experimentation:** Randomization validation, non-compliance analysis, intent-to-treat vs average treatment effect
- **Large-scale machine learning:** 13.9M observations processed with Polars for memory efficiency
- **Production ML libraries:** EconML for uplift modeling, XGBoost, scikit-learn, sklift
- **Algorithm implementation:** Meta-learners (S-learner, T-learner, X-learner, R-learner) coded from scratch
- **Statistical rigor:** IV assumptions validation, correlation analysis, AUUC and Qini coefficient metrics

## Tech Stack

```python
# Data Processing for Large-Scale A/B Testing
polars >= 0.19.0      # Memory-efficient handling of 13.9M observations
numpy >= 1.24.0

# Machine Learning & Uplift Modeling
xgboost >= 2.0.0      # Base learner for meta-learner implementations
scikit-learn >= 1.3.0

# Causal Inference & Experimentation
econml >= 0.15.0      # Microsoft's library for CATE estimation (X/S/T-learners)

# Uplift Modeling Evaluation
sklift >= 0.4.0       # AUUC, Qini coefficient, uplift curve metrics

# Statistical Visualization
matplotlib >= 3.7.0
seaborn >= 0.12.0
```

## Results Summary

**The Randomized Experiment Design:**
- Proper randomization in A/B test: 85% treatment (eligible for ads), 15% control (never shown ads)
- Baseline conversion: 0.194% (control group)
- Severe non-compliance: Only 3.6% of treatment-assigned users actually exposed to ads
- Intent-to-treat effect: 0.12pp conversion lift (0.309% vs 0.194%, diluted by non-compliance)

**CATE Uplift Modeling:**
- X-Learner meta-learner: 0.1753 Qini (best for imbalanced treatment assignment)
- Discovery: Strong correlation (r > 0.5) between CATE predictions and P(E|T=1,X)
- Limitation: Single CATE score cannot distinguish reachability from responsiveness

**Causal Decomposition Results:**
- **Local average treatment effect (LATE):** 
   - 2.86pp uniform effect (24× larger than ITT)
   - 2.71pp mean heterogeneous, ranging from -7pp to 48pp
- **CATE X-Learner baseline:** 0.1753 Qini coefficient
- **Deliverability component:** 0.1668 Qini (95% of CATE)—proves P(E) dominates
- **Combined P(E)×LATE(X):** 0.1580 Qini (90% of CATE)—enables strategic optimization

**Key Insight:** Delivery constraints dominate this campaign. Decomposition proves deliverability contributes 95% of CATE performance, enabling strategic optimization: allocate budget toward winning auctions for high-LATE users who respond strongly when reached, rather than treating all high-CATE users identically.

## Why This Matters for Data Science Roles

This uplift modeling project demonstrates:
- **Critical thinking:** Identified actionable limitation (CATE's single score) rather than implementing black-box models
- **Methodological rigor:** Applied instrumental variables to solve real business optimization problem
- **Honest evaluation:** Quantified performance-interpretability tradeoffs rather than cherry-picking metrics
- **Business translation:** Connected decomposition to concrete bidding strategies with quantified impact
- **Communication:** Clear explanation of complex causal inference concepts with actionable takeaways

*Ideal for roles in experimentation, causal ML, growth analytics, or data science positions requiring both technical depth and business acumen.*

---

**Dataset:** [Criteo Uplift Modeling Dataset v2.1](https://ailab.criteo.com/ressources/)  
**License:** MIT  
**Contact:** kuochh@github.io