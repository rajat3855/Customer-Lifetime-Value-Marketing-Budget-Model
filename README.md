# Customer Lifetime Value: how much can marketing spend to acquire each segment?

**Business question.** Marketing needs the *maximum* it can pay to acquire a customer. Spend more than a customer's
value and you lose money; spend less than you could and you leave growth on the table.

**Data.** [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii): 1,067,371 invoice lines from a
UK online gift-ware retailer, Dec 2009 - Dec 2011 (many buyers are small shops). CC BY 4.0.

**Stack.** Python - pandas, `lifetimes` (BG/NBD + Gamma-Gamma), scikit-learn (gradient-boosting challenger), matplotlib;
Excel (openpyxl, live formulas) for the budget.

## Headline results

| | |
|---|---|
| Max CAC, new customer with a **large** first order (UK) | **£250** (vs **£60** for a small first order) |
| Max CAC, average new customer | £139 |
| Holdout purchases, predicted vs actual (180 days) | 7,570 vs 7,699 (**−1.7%**) |
| Holdout revenue, predicted vs actual | −12.4% (holdout contains the Sep-Nov peak; model assumes a steady rate) |
| Top 20% by predicted value: share of real holdout revenue | **74.0%** |
| BG/NBD vs gradient boosting (Spearman) | **0.627** vs 0.593 - the statistical model wins on this data |
| Tier A (top 10% of customers): share of next-12-month value | **54%** |

*Assumptions: 30% gross margin, 3x target LTV:CAC - both editable in the workbook. The dataset has no cost or marketing data.*

## How the project is built - simple first, then only what earns its place

| Stage | What | Why |
|---|---|---|
| 1 | Clean (row-count waterfall) | Every removed row has a business reason |
| 2 | EDA | Found strong Sep-Nov seasonality and heavy concentration (top 20% = 77% of revenue) |
| 3 | Textbook CLV formula | Simplest answer - one number for everyone, untested. Kept only as a sanity check |
| 4 | Cohort CLV | Real first-year spend of new customers by **first-order size x region** - segments visible at acquisition, so they're actionable. **First Max CAC answer, no model needed** |
| 5 | RFM segments | Simple, explainable ranking of existing customers - but no £ |
| 6 | Time-based holdout + 3 baselines | Fit on the past, test on the last 180 days. Cap learned on calibration data only (no leakage) |
| 7 | BG/NBD + Gamma-Gamma | Handles unobserved churn and outputs £. Assumption checks + a Gamma-Gamma penalizer pitfall caught |
| 8 | ML challenger (HistGradientBoosting, Poisson loss) | Same-season training, fixed look-back features. Did **not** beat BG/NBD - reported honestly |
| 9 | Final 12-month CLV, value tiers, P(alive) | Refit on all data, score every customer |
| 10 | Max CAC + Excel workbook | Live formulas: change margin or target ratio, every number updates |

## What's in the folder

```
notebooks/CLV_Project.ipynb      main deliverable - every stage with explanations and outputs
src/clv_pipeline.py              same code as one script:  python src/clv_pipeline.py
src/excel_report.py              Excel workbook builder
plots/                           8 charts (01_... to 08_...)
outputs/CLV_Budget_Summary.xlsx  budget workbook: Assumptions, Acquisition_CAC, Existing_Tiers, Sensitivity, Validation
outputs/*.csv, key_numbers.json  per-customer CLV, segment tables, model comparison, all headline numbers
docs/learn_clv_project.html      interactive learning guide (Hinglish) + interview prep - open in a browser
data/online_retail_II.csv.gz     the dataset (compressed copy of the UCI Excel file)
```

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate            # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/clv_pipeline.py        # ~1 minute; or open notebooks/CLV_Project.ipynb
```

Versions are pinned because `lifetimes` is unmaintained and can break with newer pandas/numpy.

## Recommendation

1. **Pay for basket size, not clicks.** New customers whose first order is large are worth ~4x small-basket ones over
   12 months, so campaigns that bring trade buyers with big first orders justify a much higher CAC.
2. **Protect Tier A.** 10% of current customers carry 54% of next year's predicted value; treat 90 days of silence as
   the trigger for a retention offer.
3. **Get the real margin first.** Every Max CAC scales linearly with gross margin - see the *Sensitivity* sheet.

## Limitations

Margin and CAC are assumed; BG/NBD assumes a steady purchase rate (under-predicts peak-season windows); wholesale and
retail buyers are mixed; international segments are small; returns are dropped rather than netted; BG/NBD gives
one-time buyers P(alive) = 1 by construction.

**Next steps:** add marketing-channel data, test a seasonal adjustment, try Pareto/NBD or PyMC-Marketing (Bayesian,
with uncertainty ranges) as challengers, and model wholesale and retail customers separately.
