# Loan Default Analysis & Early Warning Model

A leakage-aware machine learning pipeline for predicting loan default risk from Lending Club loan data and translating model scores into risk-aware investment decisions.

The project is designed around a practical credit-risk question:

> **Given information available when a loan is issued, can we rank loans by default risk and use those scores to improve portfolio selection?**

Rather than stopping at model accuracy, the notebook covers the full workflow: data auditing, leakage detection, temporal validation, exploratory analysis, feature engineering, target encoding, model comparison, probability calibration, explainability, segment-level diagnostics, and decision-policy evaluation.

## Key Results

The dataset contains **589,549 loans and 152 columns**, covering 36-month Lending Club loans issued from **2012 through 2015**. The overall default rate is approximately **14.0%**.

| Model | Split | ROC AUC | PR AUC | Brier Score |
|---|---|---:|---:|---:|
| Baseline | Test | 0.5000 | 0.1490 | 0.1273 |
| Logistic Regression | Test | 0.6892 | 0.2645 | 0.1204 |
| **LightGBM** | **Test** | **0.6971** | **0.2745** | **0.1200** |

Additional findings:

- LightGBM improves test PR AUC from **0.2645 to 0.2745** over logistic regression.
- Probability calibration with isotonic regression reduces the test Brier score from **0.1200 to 0.1193**.
- The model was trained on **2012–2013**, tuned on **2014**, and evaluated once on **2015** to avoid future-information leakage.
- A fixed-budget ranking strategy based on model scores achieved a simulated return of about **8.72%**, compared with **5.75%** when funding the entire test book.

> These results are historical backtests on Lending Club data and should not be interpreted as evidence of live trading or lending performance.

## Why This Project Matters

Credit-risk datasets contain an unusually easy trap: many columns describe events that happen **after** a loan is approved. Using those variables can produce unrealistically high scores while making the model unusable in practice.

This project explicitly treats leakage prevention and temporal validation as first-class modeling problems.

For example, the notebook removes payment-history variables recorded after origination. Keeping a variable such as `recoveries` can produce a near-perfect AUC, but that performance is meaningless because the information is only known after the loan outcome has started to unfold.

## Project Workflow

```text
Raw Lending Club Data
        │
        ▼
Data Audit
        │
        ├── Target definition
        ├── Missingness analysis
        └── Column-quality checks
        │
        ▼
Leakage Audit
        │
        └── Remove post-origination information
        │
        ▼
Time-Based Split
        │
        ├── Train: 2012–2013
        ├── Validation: 2014
        └── Test: 2015
        │
        ▼
EDA + Drift Analysis
        │
        ▼
Feature Engineering
        │
        ├── Numeric transformations
        ├── Ratios
        └── Fold-safe target encoding
        │
        ▼
Model Ladder
        │
        ├── Base-rate baseline
        ├── Logistic Regression
        └── LightGBM
        │
        ▼
Evaluation
        │
        ├── PR AUC
        ├── ROC AUC
        └── Brier Score
        │
        ▼
Probability Calibration
        │
        └── Isotonic Regression
        │
        ▼
Decision Layer
        │
        ├── Cost-sensitive threshold
        └── Fixed-budget portfolio ranking
        │
        ▼
Explainability + Segment Diagnostics
        │
        ├── Feature importance
        ├── SHAP
        └── Segment-level calibration / ranking
```

## Dataset

The analysis uses the **Lending Club Loan Data** dataset available on Kaggle:

**Kaggle:** https://www.kaggle.com/datasets/wordsforthewise/lending-club

The analysis uses Lending Club loans with the following scope:

- **Loan term:** 36 months
- **Issue period:** January 2012 – December 2015
- **Rows:** 589,549
- **Features:** 152 columns before filtering
- **Target:** `default`

The target is defined as:

- `1` → `Charged Off`, `Default`, or `Late (31-120 days)`
- `0` → `Fully Paid`

The notebook also documents the class distribution and the change in default rate over time.

### Target distribution

Approximately **14.04%** of loans are labeled as defaults in the complete analysis set.

## Data Quality & Leakage Control

A major part of the project is determining whether a feature would genuinely have been available at loan approval time.

The notebook identifies highly incomplete columns and removes post-origination/payment-history information that would reveal the future outcome.

It also uses a **calendar-based train/validation/test split** rather than a random split. This reflects the real deployment setting more closely: a lender trains on historical loans and scores loans from a future period.

### Why time splitting?

The default rate increases from roughly **12.3% in 2013 to 14.9% in 2015**, so treating all observations as exchangeable would hide temporal drift.

Using the split:

```text
Train      → 2012–2013
Validation → 2014
Test       → 2015
```

reduces the risk of accidentally using information from the future.

## Exploratory Analysis

The notebook investigates several practical questions:

- How stable is the default rate over time?
- How strongly does Lending Club grade correlate with realized default?
- Which numerical variables separate defaulters from non-defaulters?
- Which loan purposes are associated with higher observed risk?
- How much do important numerical features overlap across the two classes?

One useful sanity check is Lending Club's own grade system:

| Grade | Default Rate | Avg. Interest Rate |
|---|---:|---:|
| A | 5.45% | 7.21% |
| B | 11.23% | 10.86% |
| C | 18.04% | 13.94% |
| D | 23.57% | 17.26% |
| E | 28.92% | 19.98% |
| F | 33.18% | 23.90% |
| G | 40.17% | 26.27% |

This establishes an important baseline: a useful model has to provide information beyond simply rediscovering the platform's existing risk grades.

## Feature Engineering

The notebook transforms raw Lending Club fields into model-ready variables, including:

- Numeric conversion for fields stored as strings
- Loan-to-income style ratios
- Credit-history related variables
- Derived financial relationships between loan size, income, debt, and repayment burden
- Categorical variables for model consumption

A particularly important feature is the **state-level default-rate encoding** of `addr_state`.

Because target encoding can leak the target, it is implemented in a **fold-safe** manner rather than calculating the state default rate using each row's own outcome.

## Models

### 1. Base-Rate Baseline

The baseline predicts the training default rate for every observation.

It provides the minimum useful benchmark and confirms that the ML models are learning ranking information rather than simply reproducing the class prior.

### 2. Logistic Regression

A linear baseline using:

- Median imputation for numerical variables
- Standard scaling
- Most-frequent imputation for categorical variables
- One-hot encoding for categorical features

This provides an interpretable benchmark before introducing a nonlinear model.

### 3. LightGBM

The final model is a gradient-boosted tree classifier trained with a binary objective and average precision as the validation metric.

The training procedure uses early stopping and regularization to limit overfitting.

Best validation iteration: **533 boosting rounds**.

## Evaluation Metrics

Accuracy is deliberately **not** used as the primary metric.

Because the positive class is relatively uncommon and financially costly, the project focuses on:

### PR AUC

Average precision measures how well the model concentrates true defaults near the top of its ranking.

### ROC AUC

Measures overall ranking quality and provides a familiar comparison metric.

### Brier Score

Measures the quality of predicted probabilities rather than only their ordering.

For this problem, the ranking metrics and calibration metric answer different questions:

```text
PR AUC / ROC AUC → Can the model rank risk?
Brier Score       → Do the probabilities mean what they claim?
```

## Probability Calibration

A ranking model is not automatically a well-calibrated probability model.

The project therefore fits an **isotonic regression calibrator** on the validation period and evaluates the resulting probabilities on the held-out test period.

Results:

```text
Raw LightGBM Brier Score        0.1200
Calibrated LightGBM Brier Score 0.1193
```

Calibration improves the probability mapping without changing the underlying ranking.

## Decision Policy

The project goes one step beyond predicting default probability and asks:

> **What should an investor actually do with the score?**

Instead of blindly using a probability threshold of `0.5`, the notebook models the asymmetric cost of the two possible mistakes:

- Declining a loan that would have paid means giving up potential interest income.
- Approving a loan that defaults exposes the lender to principal loss.

The resulting threshold is selected on the validation set and applied once to the test period.

The analysis finds a chosen threshold of approximately **0.332** with a simulated approval rate of approximately **98.65%**.

This leads to an important business insight: simply declining loans above a single default-risk threshold has limited upside because the platform's existing pricing already captures much of the expected loss.

## Portfolio Selection Experiment

The more useful decision problem is treated as **ranking under a fixed deployment budget**.

For a deployment share of **30%** of the available loans, the notebook compares:

- Funding the entire book
- Ranking by the model's predicted risk
- Ranking by Lending Club sub-grade
- Ranking by yield / interest rate

Simulated returns from the notebook:

| Strategy | Return per Dollar |
|---|---:|
| Fund Everything | 5.75% |
| **Model Ranking** | **8.72%** |
| Sub-Grade Ranking | 4.47% |
| Yield Chasing | 3.64% |

The model-ranking strategy also produces a test-set default rate of approximately **12.94%** among the selected loans.

These are backtest results under the assumptions implemented in the notebook, not guaranteed future returns.

## Explainability

The project uses both global and local model explanations.

### Feature Importance

LightGBM gain importance highlights the variables most heavily used in tree splits.

### SHAP

SHAP values are used to explain the direction and magnitude of individual feature contributions to a prediction.

This helps answer questions such as:

> Why did the model assign this particular loan a higher default probability?

## Segment Diagnostics

A strong average AUC can hide systematic problems in particular groups.

The notebook therefore evaluates performance and calibration across segments to identify cases where the model may:

- Rank poorly within a segment
- Systematically over-predict risk
- Systematically under-predict risk

This is an important step before considering any credit-risk model for real operational use.

## Project Structure

A recommended repository layout is:

```text
loan-default-analysis/
│
├── data/
│   └── interim/
│       └── loans_36m_2012_2015.parquet
│
├── models/
│   └── default_model.joblib
│
├── reports/
│   ├── metrics.json
│   └── default_by_grade.png
│
├── notebooks/
│   └── loan_default_analysis.ipynb
│
├── README.md
└── requirements.txt
```

The notebook creates the `models/` and `reports/` directories automatically when executed.

## Installation

Python 3.10+ is recommended.

Install the required packages:

```bash
pip install pandas numpy scikit-learn lightgbm shap matplotlib seaborn joblib pyarrow
```

Or use a `requirements.txt` file:

```text
numpy
pandas
scikit-learn
lightgbm
shap
matplotlib
seaborn
joblib
pyarrow
```

## Running the Project

1. Place the prepared Lending Club dataset at:

```text
data/interim/loans_36m_2012_2015.parquet
```

2. Open the notebook:

```bash
jupyter notebook notebooks/loan_default_analysis.ipynb
```

3. Run the notebook from top to bottom.

The notebook generates:

- Evaluation metrics
- Diagnostic plots
- Model explanations
- A serialized model artifact
- `reports/metrics.json`

## Reproducibility

The notebook uses a fixed random seed:

```python
RANDOM_STATE = 42
```

The train/validation/test split is chronological and the test period is held out until final evaluation.

## Limitations

This project is a historical modeling study rather than a production credit-decision system.

Important limitations include:

1. **Approved-loan selection bias / reject inference**  
   The model only observes applications that Lending Club approved. It does not learn how rejected borrowers would have performed.

2. **Temporal drift**  
   The notebook already detects distributional and calibration drift across time. A frozen model should therefore not be assumed to remain reliable indefinitely.

3. **Economic assumptions**  
   The decision layer uses simplified assumptions for interest income and loss given default. These assumptions can materially change the optimal threshold and portfolio ranking.

4. **Fairness and governance**  
   Segment-level checks are included, but a real lending system would require a much more extensive fairness, compliance, and model-risk review.

5. **Historical dataset**  
   Results on 2012–2015 Lending Club loans should not be treated as evidence that the same performance will hold on current lending data.

## Future Improvements

Potential extensions include:

- Reject inference using rejected-loan applications
- Rolling-window retraining and out-of-time monitoring
- More robust drift monitoring
- Hyperparameter optimization
- Cost-sensitive / expected-value learning
- Fairness analysis across additional borrower segments
- Production inference API
- Automated model monitoring and recalibration

## Takeaway

The central lesson of the project is that credit-risk modeling is not just a classification problem.

A credible solution needs to answer four questions:

```text
1. Is the data available at prediction time?
2. Does the model generalize to future periods?
3. Are the predicted probabilities calibrated?
4. Do the scores improve an actual economic decision?
```

This project addresses all four, making the notebook a compact end-to-end example of **applied machine learning for credit risk and decision analytics**.

---

### Disclaimer

This repository is for educational and research purposes. The historical results presented here are based on the assumptions and dataset used in the notebook and are not financial advice or a production-ready lending model.
