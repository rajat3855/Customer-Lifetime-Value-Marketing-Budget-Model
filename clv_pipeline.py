"""
clv_pipeline.py - the whole notebook as one script.
Run from the project folder:  python src/clv_pipeline.py
Generated from notebooks/CLV_Project.ipynb (same code, same order).
"""


import sys, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
IN_NOTEBOOK = "ipykernel" in sys.modules
if not IN_NOTEBOOK:
    matplotlib.use("Agg")          
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data, calibration_and_holdout_data
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pandas")

ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
DATA, PLOTS, OUT = ROOT / "data", ROOT / "plots", ROOT / "outputs"
PLOTS.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))

MARGIN = 0.30          # gross margin - ASSUMED, the data has no cost figures
TARGET_RATIO = 3.0     # target LTV:CAC (spend at most 1/3 of a customer's gross profit)
HOLDOUT_DAYS = 180     # length of the "future" we hide for testing
HORIZON_MONTHS = 12    # CLV horizon
MONTHLY_DISCOUNT = 0.01  # 1% a month: £1 next year is worth a bit less than £1 today
CAP_QUANTILE = 0.995   # winsorise extreme purchase-days (one-off giant wholesale orders)
SEED = 42

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})

def save(fig, name):
    fig.tight_layout()
    fig.savefig(PLOTS / name, dpi=150)
    if IN_NOTEBOOK:
        plt.show()
    else:
        plt.close(fig)

pd.set_option("display.width", 160); pd.set_option("display.max_columns", 20)
if not IN_NOTEBOOK:
    sys.stdout.reconfigure(encoding="utf-8")   
print("Project folder:", ROOT)

csv_path, xlsx_path = DATA / "online_retail_II.csv.gz", DATA / "online_retail_II.xlsx"
if csv_path.exists():
    raw = pd.read_csv(csv_path, dtype={"Invoice": str, "StockCode": str}, parse_dates=["InvoiceDate"])
else:
    print("First run: reading the Excel file (1-3 minutes)...")
    raw = pd.concat(pd.read_excel(xlsx_path, sheet_name=None, dtype={"Invoice": str, "StockCode": str}).values(),
                    ignore_index=True)
    raw.to_csv(csv_path, index=False)
print(raw.shape)
print(raw.head())
NON_PRODUCT = r"^(?:POST|DOT|M|C2|D|S|B|BANK CHARGES|ADJUST\d?|AMAZONFEE|PADS|CRUK|TEST\d+|SP1002)$"

steps = [("Raw rows", len(raw))]
df = raw.drop_duplicates();                                    steps.append(("Drop exact duplicates (sheet overlap)", len(df)))
df = df[~df["Invoice"].str.startswith(("C", "A"))];            steps.append(("Drop cancellations / adjustments", len(df)))
df = df.dropna(subset=["Customer ID"]);                        steps.append(("Drop guest checkouts (no Customer ID)", len(df)))
df = df[(df["Quantity"] > 0) & (df["Price"] > 0)];             steps.append(("Keep Quantity > 0 and Price > 0", len(df)))
df = df[~df["StockCode"].str.upper().str.match(NON_PRODUCT)];  steps.append(("Drop postage / fees / manual codes", len(df)))

df = df.assign(revenue=df["Quantity"] * df["Price"],
               date=df["InvoiceDate"].dt.normalize(),
               customer=df["Customer ID"].astype(int))
waterfall = pd.DataFrame(steps, columns=["step", "rows_left"])
waterfall["removed"] = -waterfall["rows_left"].diff().fillna(0).astype(int)
print(waterfall.to_string(index=False))
print(f"\nCustomers: {df['customer'].nunique():,} | invoices: {df['Invoice'].nunique():,} | "
      f"revenue: £{df['revenue'].sum():,.0f} | {df['date'].min().date()} to {df['date'].max().date()}")
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.barh(waterfall["step"][::-1], waterfall["rows_left"][::-1] / 1e3, color=BLUE, height=0.55)
for y, v in enumerate(waterfall["rows_left"][::-1]):
    ax.text(v / 1e3 + 10, y, f"{v:,}", va="center", fontsize=9, color="#52514e")
ax.set_xlabel("Rows left (thousands)"); ax.set_title("Cleaning waterfall: 1.07M lines -> 0.78M usable")
ax.set_xlim(0, 1250); ax.grid(axis="y", visible=False)
save(fig, "01_cleaning_waterfall.png")

tx = df.groupby(["customer", "date"], as_index=False)["revenue"].sum()   
country = df.groupby("customer")["Country"].agg(lambda s: s.mode().iat[0])
is_uk = (country == "United Kingdom")
OBS_END = tx["date"].max()
print(f"{len(tx):,} purchase days from {tx['customer'].nunique():,} customers; last date {OBS_END.date()}")
print(f"UK customers: {is_uk.mean():.1%}")
print(tx.head())

monthly = tx.assign(month=tx["date"].dt.to_period("M")).groupby("month")["revenue"].sum()
fig, ax = plt.subplots(figsize=(9, 3.6))
colors = [ORANGE if m.month in (9, 10, 11) else BLUE for m in monthly.index]
ax.bar(range(len(monthly)), monthly.values / 1e3, color=colors, width=0.75)
ax.set_xticks(range(0, len(monthly), 3)); ax.set_xticklabels([str(m) for m in monthly.index[::3]])
ax.set_ylabel("Revenue (£ thousands)")
ax.set_title("Monthly revenue: a big Sep-Nov peak every year (orange)")
save(fig, "02_monthly_revenue.png")
print("Note: Dec 2011 is only 9 days of data.")
cust_rev = tx.groupby("customer")["revenue"].sum().sort_values(ascending=False)
share = cust_rev.cumsum() / cust_rev.sum() * 100
pct = np.arange(1, len(cust_rev) + 1) / len(cust_rev) * 100
top10 = share.iloc[int(0.10 * len(share)) - 1]
top20 = share.iloc[int(0.20 * len(share)) - 1]

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(pct, share.values, color=BLUE, lw=2)
ax.plot([0, 100], [0, 100], color=GREY, lw=1, ls="--")
ax.scatter([20], [top20], color=BLUE, s=40, zorder=3)
ax.annotate(f"Top 20% of customers = {top20:.0f}% of revenue", (20, top20), xytext=(32, 42),
            fontsize=9, color="#0b0b0b", arrowprops=dict(arrowstyle="-", color=GREY))
ax.set_xlabel("% of customers (biggest first)"); ax.set_ylabel("% of total revenue")
ax.set_title("Revenue is concentrated in a few customers")
save(fig, "03_revenue_concentration.png")
print(f"Top 10% of customers: {top10:.1f}% of revenue | top 20%: {top20:.1f}%")
print(cust_rev.describe(percentiles=[.5, .9, .99]).round(0).to_string())

n_cust = tx["customer"].nunique()
aov = tx["revenue"].mean()                                    # average purchase-day value
years = (OBS_END - tx["date"].min()).days / 365
freq_per_year = len(tx) / n_cust / years                      # purchases per customer per year
repeat_rate = (tx.groupby("customer").size() > 1).mean()      # share who ever came back
lifespan = 1 / (1 - repeat_rate)                              # classic "1 / churn" shortcut
clv_v1 = aov * freq_per_year * lifespan * MARGIN
print(f"AOV £{aov:,.0f} x {freq_per_year:.2f} purchases/yr x lifespan {lifespan:.1f} yrs x margin {MARGIN:.0%}")
print(f"=> textbook CLV (gross profit) = £{clv_v1:,.0f} per customer  =>  'max CAC' = £{clv_v1 / TARGET_RATIO:,.0f}")

first_date = tx.groupby("customer")["date"].min()
first_order = tx.sort_values("date").groupby("customer")["revenue"].first()
cohort_ids = first_date[(first_date >= "2010-03-01") & (first_date <= OBS_END - pd.Timedelta(days=365))].index

ct = tx[tx["customer"].isin(cohort_ids)].join(first_date.rename("first_date"), on="customer")
ct = ct[ct["date"] < ct["first_date"] + pd.Timedelta(days=365)]          # first 12 months only
cohort = pd.DataFrame({"ltv_12m": ct.groupby("customer")["revenue"].sum(),
                       "purchases_12m": ct.groupby("customer").size()})
cohort["first_order"] = first_order.reindex(cohort.index)
cohort["order_band"] = pd.qcut(cohort["first_order"], 3, labels=["Small", "Medium", "Large"])
cohort["region"] = np.where(is_uk.reindex(cohort.index), "UK", "International")
band_edges = cohort.groupby("order_band", observed=True)["first_order"].agg(["min", "max"]).round(0)

print(f"Cohort: {len(cohort):,} new customers | came back within 12 months: {(cohort['purchases_12m'] > 1).mean():.1%}")
print(f"Mean 12-month revenue £{cohort['ltv_12m'].mean():,.0f} | median £{cohort['ltv_12m'].median():,.0f}")
print("\nFirst-order bands (£):"); print(band_edges.to_string())

acq = (cohort.groupby(["order_band", "region"], observed=True)
             .agg(customers=("ltv_12m", "size"), avg_ltv_12m=("ltv_12m", "mean"),
                  median_ltv_12m=("ltv_12m", "median"), repeat_rate=("purchases_12m", lambda s: (s > 1).mean())))
acq["max_cac"] = acq["avg_ltv_12m"] * MARGIN / TARGET_RATIO
print(acq.round(2))
lab = [f"{b}\n{r}" for b, r in acq.index]
fig, ax = plt.subplots(figsize=(8, 3.8))
x = np.arange(len(acq))
ax.bar(x, acq["max_cac"], color=[BLUE if r == "UK" else AQUA for _, r in acq.index], width=0.6)
for xi, (v, n) in enumerate(zip(acq["max_cac"], acq["customers"])):
    ax.text(xi, v + 5, f"£{v:,.0f}\n(n={n})", ha="center", fontsize=8.5, color="#0b0b0b")
ax.set_xticks(x); ax.set_xticklabels(lab)
ax.set_ylabel(f"Max CAC (£) at {MARGIN:.0%} margin, {TARGET_RATIO:.0f}x")
ax.set_ylim(0, acq["max_cac"].max() * 1.25)
ax.set_title("Max CAC by first-order size: big first baskets justify ~4x the spend")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE, label="UK"), Patch(color=AQUA, label="International")], loc="upper left")
save(fig, "04_max_cac_by_acquisition_segment.png")

CAL_END = OBS_END - pd.Timedelta(days=HOLDOUT_DAYS)
past = tx[tx["date"] <= CAL_END]
future = tx[tx["date"] > CAL_END]

rfm = past.groupby("customer").agg(last=("date", "max"), F=("date", "size"), M=("revenue", "sum"))
rfm["R_days"] = (CAL_END - rfm["last"]).dt.days
rfm["R"] = pd.qcut(rfm["R_days"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)  # recent = 5
rfm["F_s"] = pd.qcut(rfm["F"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm["M_s"] = pd.qcut(rfm["M"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm["RFM_score"] = rfm["R"] + rfm["F_s"] + rfm["M_s"]

def rfm_segment(r, f):
    if r >= 4 and f >= 4: return "1 Champions"
    if r >= 3 and f >= 3: return "2 Loyal"
    if r >= 4:            return "3 New / promising"
    if r == 3:            return "4 Needs attention"
    if f >= 3:            return "5 At risk"
    return "6 Hibernating"

rfm["segment"] = [rfm_segment(r, f) for r, f in zip(rfm["R"], rfm["F_s"])]
fut_rev = future.groupby("customer")["revenue"].sum()
rfm["future_rev"] = fut_rev.reindex(rfm.index).fillna(0)

rfm_tbl = rfm.groupby("segment").agg(customers=("F", "size"), avg_days_since=("R_days", "mean"),
                                     avg_past_spend=("M", "mean"),
                                     came_back=("future_rev", lambda s: (s > 0).mean()),
                                     avg_next_180d_rev=("future_rev", "mean"))
print(rfm_tbl.round(1))
fig, ax = plt.subplots(figsize=(8, 3.6))
y = np.arange(len(rfm_tbl))[::-1]
ax.barh(y, rfm_tbl["avg_next_180d_rev"], color=BLUE, height=0.55)
for yi, v, cb in zip(y, rfm_tbl["avg_next_180d_rev"], rfm_tbl["came_back"]):
    ax.text(v + 30, yi, f"£{v:,.0f}  ({cb:.0%} bought again)", va="center", fontsize=9)
ax.set_yticks(y); ax.set_yticklabels([s[2:] for s in rfm_tbl.index])
ax.set_xlabel("Average revenue in the NEXT 180 days (£)"); ax.set_xlim(0, rfm_tbl["avg_next_180d_rev"].max() * 1.45)
ax.set_title("RFM segments really do predict the future"); ax.grid(axis="y", visible=False)
save(fig, "05_rfm_segments_next_180d.png")

cap = past["revenue"].quantile(CAP_QUANTILE)
tx_c = tx.assign(revenue=tx["revenue"].clip(upper=cap))
print(f"Purchase-day cap: £{cap:,.0f}  ({(past['revenue'] > cap).sum()} calibration purchase-days capped)")

cal = calibration_and_holdout_data(tx_c, "customer", "date", calibration_period_end=CAL_END,
                                   observation_period_end=OBS_END, freq="D", monetary_value_col="revenue")
actual = tx_c[tx_c["date"] > CAL_END].groupby("customer")["revenue"].sum().reindex(cal.index).fillna(0)
print(f"Customers in test: {len(cal):,} | actual holdout revenue (capped): £{actual.sum():,.0f}")
print(cal.head())
def evaluate(name, pred, actual=actual, money=True):
    pred = pd.Series(np.asarray(pred, dtype=float), index=actual.index)
    top = pred.rank(ascending=False, method="first") <= 0.2 * len(pred)
    return {"model": name,
            "total_error": pred.sum() / actual.sum() - 1 if money else np.nan,
            "MAE_gbp": (pred - actual).abs().mean() if money else np.nan,
            "spearman": spearmanr(pred, actual)[0] if pred.nunique() > 1 else np.nan,
            "top20_capture": actual[top].sum() / actual.sum()}

results = []
past_c = tx_c[tx_c["date"] <= CAL_END]
hist_total = past_c.groupby("customer")["revenue"].sum().reindex(cal.index)
pred_naive = hist_total / (cal["T_cal"] + 1) * HOLDOUT_DAYS           # baseline 1: history repeats
results.append(evaluate("Baseline: history repeats", pred_naive))
results.append(evaluate("Baseline: RFM score", rfm["RFM_score"].reindex(cal.index), money=False))
results.append(evaluate("Baseline: same £ for everyone", np.full(len(cal), actual.mean())))
print(pd.DataFrame(results).set_index("model").round(3))

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cal["frequency_cal"], cal["recency_cal"], cal["T_cal"])
print("BG/NBD parameters:", {k: round(v, 3) for k, v in bgf.params_.items()})

pred_purchases = bgf.conditional_expected_number_of_purchases_up_to_time(
    HOLDOUT_DAYS, cal["frequency_cal"], cal["recency_cal"], cal["T_cal"])
print(f"Holdout purchase days - predicted {pred_purchases.sum():,.0f} vs actual {cal['frequency_holdout'].sum():,.0f} "
      f"({pred_purchases.sum() / cal['frequency_holdout'].sum() - 1:+.1%})")

bucket = cal["frequency_cal"].clip(upper=7).astype(int)
calib = pd.DataFrame({"actual": cal["frequency_holdout"], "predicted": pred_purchases, "bucket": bucket}).groupby("bucket").mean()
fig, ax = plt.subplots(figsize=(7, 3.8))
xb = np.arange(len(calib))
ax.bar(xb - 0.2, calib["actual"], 0.38, color=BLUE, label="Actual")
ax.bar(xb + 0.2, calib["predicted"], 0.38, color=ORANGE, label="Predicted (BG/NBD)")
ax.set_xticks(xb); ax.set_xticklabels([("7+" if i == 7 else str(i)) for i in calib.index])
ax.set_xlabel("Repeat purchases during calibration"); ax.set_ylabel("Avg purchase days in holdout")
ax.set_title("BG/NBD holdout check: predictions track reality"); ax.legend()
save(fig, "06_bgnbd_holdout_calibration.png")
repeaters = cal[cal["frequency_cal"] > 0]
corr = np.corrcoef(repeaters["frequency_cal"], repeaters["monetary_value_cal"])[0, 1]
print(f"Gamma-Gamma assumption check - corr(frequency, basket value) = {corr:.2f}  (near 0 is good; > 0.3 would be a worry)")

def implied_mean(g):
    p, q, v = g.params_["p"], g.params_["q"], g.params_["v"]
    return p * v / (q - 1) if q > 1 else np.nan

emp = repeaters["monetary_value_cal"].mean()
for pen in (0.001, 0.0):
    g = GammaGammaFitter(penalizer_coef=pen).fit(repeaters["frequency_cal"], repeaters["monetary_value_cal"])
    m = implied_mean(g)
    shown = f"£{m:,.0f}" if np.isfinite(m) else "UNDEFINED (q < 1 means an infinite average basket!)"
    print(f"  penalizer={pen}: params {g.params_.round(2).to_dict()} -> implied avg basket {shown} vs real £{emp:,.0f}")

ggf = GammaGammaFitter(penalizer_coef=0.0).fit(repeaters["frequency_cal"], repeaters["monetary_value_cal"])
assert abs(implied_mean(ggf) / emp - 1) < 0.2, "Gamma-Gamma implied mean is implausible - do not trust it"

pred_basket = ggf.conditional_expected_average_profit(cal["frequency_cal"], cal["monetary_value_cal"])
pred_bg = pred_purchases * pred_basket
results.append(evaluate("BG/NBD + Gamma-Gamma", pred_bg))
print(pd.DataFrame(results).set_index("model").round(3))

def ml_features(tx_, cutoff, lookback=180):
    h = tx_[(tx_["date"] <= cutoff) & (tx_["date"] > cutoff - pd.Timedelta(days=lookback))]
    g = h.groupby("customer")
    h90 = h[h["date"] > cutoff - pd.Timedelta(days=90)].groupby("customer")
    f = pd.DataFrame({"recency_days": (cutoff - g["date"].max()).dt.days,
                      "first_seen_days": (cutoff - g["date"].min()).dt.days,
                      "n_purchase_days": g.size(), "revenue": g["revenue"].sum(),
                      "avg_basket": g["revenue"].mean(), "max_basket": g["revenue"].max(),
                      "n_last_90d": h90.size(), "revenue_last_90d": h90["revenue"].sum()}).fillna(0)
    f["is_uk"] = is_uk.reindex(f.index).astype(int)
    return f

def next_revenue(tx_, cutoff, days=HOLDOUT_DAYS):
    w = tx_[(tx_["date"] > cutoff) & (tx_["date"] <= cutoff + pd.Timedelta(days=days))]
    return w.groupby("customer")["revenue"].sum()

train_cut = CAL_END - pd.Timedelta(days=365)
X_train = ml_features(tx_c, train_cut)
y_train = next_revenue(tx_c, train_cut).reindex(X_train.index).fillna(0)
X_test = ml_features(tx_c, CAL_END)
print(f"Train: {len(X_train):,} customers (cut-off {train_cut.date()}) | Test: {len(X_test):,} active customers (cut-off {CAL_END.date()})")

gbm = HistGradientBoostingRegressor(loss="poisson", max_iter=300, learning_rate=0.05,
                                    max_leaf_nodes=15, min_samples_leaf=40, random_state=SEED)
gbm.fit(X_train, y_train)

pred_ml = pred_bg.copy()                                    # start from BG/NBD for everyone
pred_ml.loc[X_test.index] = np.clip(gbm.predict(X_test), 0, None)   # overwrite active customers with ML
results.append(evaluate("ML: gradient boosting (+BG/NBD for inactive)", pred_ml))
comparison = pd.DataFrame(results).set_index("model")
print(comparison.round(3))
cmp_plot = comparison.dropna(subset=["spearman"]).iloc[:, [2, 3]]
cmp_plot = cmp_plot.drop(index="Baseline: same £ for everyone", errors="ignore")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
names = [n.replace("Baseline: ", "").replace(" (+BG/NBD for inactive)", "") for n in cmp_plot.index]
for ax, col, title in zip(axes, ["spearman", "top20_capture"], ["Ranking quality (Spearman)", "Top-20% revenue capture"]):
    c = [ORANGE if "BG/NBD +" in n else BLUE for n in cmp_plot.index]
    ax.barh(names[::-1], cmp_plot[col][::-1], color=c[::-1], height=0.55)
    for yi, v in enumerate(cmp_plot[col][::-1]):
        ax.text(v + 0.01, yi, f"{v:.3f}" if col == "spearman" else f"{v:.1%}", va="center", fontsize=9)
    ax.set_title(title); ax.set_xlim(0, 1); ax.grid(axis="y", visible=False)
fig.suptitle("Holdout comparison: BG/NBD + Gamma-Gamma (orange) ranks customers best", x=0.01, ha="left", fontweight="bold")
save(fig, "07_model_comparison.png")

cap_all = tx["revenue"].quantile(CAP_QUANTILE)
tx_all = tx.assign(revenue=tx["revenue"].clip(upper=cap_all))
summ = summary_data_from_transaction_data(tx_all, "customer", "date", monetary_value_col="revenue",
                                          observation_period_end=OBS_END, freq="D")
bgf_f = BetaGeoFitter(penalizer_coef=0.001).fit(summ["frequency"], summ["recency"], summ["T"])
rep_f = summ[summ["frequency"] > 0]
ggf_f = GammaGammaFitter(penalizer_coef=0.0).fit(rep_f["frequency"], rep_f["monetary_value"])
assert abs(implied_mean(ggf_f) / rep_f["monetary_value"].mean() - 1) < 0.2

summ["clv_12m"] = ggf_f.customer_lifetime_value(bgf_f, summ["frequency"], summ["recency"], summ["T"],
                                                summ["monetary_value"], time=HORIZON_MONTHS,
                                                discount_rate=MONTHLY_DISCOUNT, freq="D")
summ["p_alive"] = bgf_f.conditional_probability_alive(summ["frequency"], summ["recency"], summ["T"])
summ["exp_purchases_12m"] = bgf_f.conditional_expected_number_of_purchases_up_to_time(365, summ["frequency"], summ["recency"], summ["T"])
summ["days_since_last"] = summ["T"] - summ["recency"]
summ["region"] = np.where(is_uk.reindex(summ.index), "UK", "International")

TIERS = [("A - Top 10%", 0.90, 1.00), ("B - Next 20%", 0.70, 0.90), ("C - Middle 40%", 0.30, 0.70), ("D - Bottom 30%", 0.00, 0.30)]
rank_pct = summ["clv_12m"].rank(pct=True, method="first")
summ["tier"] = ""
for name, lo, hi in TIERS:
    summ.loc[(rank_pct > lo) & (rank_pct <= hi), "tier"] = name

tiers = summ.groupby("tier").agg(customers=("clv_12m", "size"), avg_clv_12m=("clv_12m", "mean"),
                                 total_clv_12m=("clv_12m", "sum"), avg_p_alive=("p_alive", "mean"),
                                 avg_purchases_12m=("exp_purchases_12m", "mean"), avg_days_since_last=("days_since_last", "mean"))
tiers["share_of_value"] = tiers["total_clv_12m"] / tiers["total_clv_12m"].sum()
print("Final BG/NBD:", {k: round(v, 3) for k, v in bgf_f.params_.items()}, "| Gamma-Gamma:", ggf_f.params_.round(2).to_dict())
print(f"Total predicted 12-month revenue from existing customers: £{summ['clv_12m'].sum():,.0f}")
print(tiers.round(2))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
ax = axes[0]
ax.bar([t[0].split(" - ")[0] for t in TIERS], tiers["share_of_value"] * 100, color=BLUE, width=0.6)
for xi, (v, n) in enumerate(zip(tiers["share_of_value"] * 100, tiers["customers"])):
    ax.text(xi, v + 1.5, f"{v:.0f}%\n{n:,} cust.", ha="center", fontsize=8.5)
ax.set_ylim(0, 75); ax.set_ylabel("% of predicted 12-month value"); ax.set_title("Tier A: 10% of customers, ~half of value")
ax.grid(axis="x", visible=False)
ax = axes[1]
repeat_only = summ[summ["frequency"] > 0]          # one-time buyers always get p_alive = 1 in BG/NBD (see note)
ax.scatter(repeat_only["days_since_last"], repeat_only["p_alive"], s=8, alpha=0.35, color=BLUE, edgecolors="none")
ax.set_xlabel("Days since last purchase"); ax.set_ylabel("P(still a customer)")
ax.set_title("Silence = churn risk (repeat buyers)")
save(fig, "08_clv_tiers_and_p_alive.png")

from excel_report import write_budget_workbook

acq_rows = [(f"{band} first order - {region}", r.customers, r.avg_ltv_12m) for (band, region), r in acq.iterrows()]
acq_rows.append(("All new customers (cohort average)", len(cohort), cohort["ltv_12m"].mean()))
tier_rows = [(f"Tier {name}", tiers.loc[name, "customers"], tiers.loc[name, "avg_clv_12m"]) for name, _, _ in TIERS]
tier_rows.append(("All existing customers", len(summ), summ["clv_12m"].mean()))

bg = comparison.loc["BG/NBD + Gamma-Gamma"]
validation = {
    "Calibration period ends": str(CAL_END.date()),
    "Holdout length (days)": HOLDOUT_DAYS,
    "Customers tested": len(cal),
    "Purchases: prediction error (%)": float(pred_purchases.sum() / cal["frequency_holdout"].sum() - 1),
    "Revenue: prediction error (%)": float(bg["total_error"]),
    "MAE per customer (£), BG/NBD": float(bg["MAE_gbp"]),
    "MAE per customer (£), 'history repeats' baseline": float(comparison.loc["Baseline: history repeats", "MAE_gbp"]),
    "Spearman rank correlation, BG/NBD": float(bg["spearman"]),
    "Spearman rank correlation, ML challenger": float(comparison.iloc[-1]["spearman"]),
    "Top 20% predicted: share of actual revenue": float(bg["top20_capture"]),
    "Gamma-Gamma check: corr(frequency, basket)": float(corr),
}
window = f"{tx['date'].min().date()} to {OBS_END.date()}"
write_budget_workbook(OUT / "CLV_Budget_Summary.xlsx", acq_rows, tier_rows, MARGIN, TARGET_RATIO, validation, window)

budget = pd.DataFrame(acq_rows, columns=["segment", "customers", "avg_12m_revenue"])
budget["gross_profit_12m"] = budget["avg_12m_revenue"] * MARGIN
budget["max_cac"] = budget["gross_profit_12m"] / TARGET_RATIO
print("Saved outputs/CLV_Budget_Summary.xlsx")
print(budget.round(0))
# Save everything a reviewer might want to open
summ.reset_index().rename(columns={"customer": "customer_id"}).round(4).to_csv(OUT / "customer_clv_12m.csv", index=False)
acq.round(2).to_csv(OUT / "acquisition_segments.csv")
tiers.round(2).to_csv(OUT / "existing_customer_tiers.csv")
comparison.round(4).to_csv(OUT / "model_comparison.csv")
waterfall.to_csv(OUT / "cleaning_waterfall.csv", index=False)
key = {
    "cleaning": steps, "cap_gbp": float(cap), "assumptions": {"margin": MARGIN, "target_ratio": TARGET_RATIO,
    "holdout_days": HOLDOUT_DAYS, "monthly_discount": MONTHLY_DISCOUNT},
    "textbook_clv_gp": float(clv_v1), "cohort_size": int(len(cohort)),
    "cohort_mean_ltv_12m": float(cohort["ltv_12m"].mean()), "cohort_median_ltv_12m": float(cohort["ltv_12m"].median()),
    "cohort_repeat_rate": float((cohort["purchases_12m"] > 1).mean()),
    "top20_share_of_revenue": float(top20 / 100), "validation": validation,
    "comparison": comparison.round(4).reset_index().to_dict(orient="records"),
    "acquisition": budget.round(2).to_dict(orient="records"),
    "tiers": tiers.round(2).reset_index().to_dict(orient="records"),
    "bgnbd_final": {k: float(v) for k, v in bgf_f.params_.items()},
    "gg_final": {k: float(v) for k, v in ggf_f.params_.items()},
}
(OUT / "key_numbers.json").write_text(json.dumps(key, indent=2, default=str))
print("Saved:", sorted(p.name for p in OUT.iterdir()))

