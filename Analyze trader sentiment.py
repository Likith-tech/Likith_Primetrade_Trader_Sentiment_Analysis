"""
╔══════════════════════════════════════════════════════════════════╗
║   Trader Performance vs Bitcoin Market Sentiment                 ║
║   Primetrade.ai Data Science Assignment                          ║
║   Dataset: Hyperliquid Historical Trades + Fear & Greed Index    ║
╚══════════════════════════════════════════════════════════════════╝

Sections:
  1. Load & Inspect
  2. Clean & Preprocess
  3. Merge
  4. Core Analysis
  5. Advanced Insights
       A. Win-rate matrix (sentiment × direction)
       B. Leverage risk analysis
       C. Trader consistency (Sharpe-like score)
       D. Sentiment momentum & contrarian signals
       E. Symbol-level performance breakdown
  6. Visualisations (10 publication-quality charts)
  7. Export artefacts

Requirements:
    pip install pandas matplotlib seaborn scipy
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Global style ───────────────────────────────────────────────────────────────
PALETTE = {
    "Extreme Fear": "#d62728",
    "Fear":         "#ff7f0e",
    "Neutral":      "#bcbd22",
    "Greed":        "#2ca02c",
    "Extreme Greed":"#17becf",
}
SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]

plt.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "axes.titlecolor":   "#f0f6fc",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#c9d1d9",
    "grid.color":        "#21262d",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "legend.facecolor":  "#161b22",
    "legend.edgecolor":  "#30363d",
    "font.family":       "monospace",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
})

DIVIDER = "=" * 65

def section(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)

def sub(label: str, value=None) -> None:
    if value is not None:
        print(f"  {label:<35} {value}")
    else:
        print(f"  {label}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & INSPECT
# ══════════════════════════════════════════════════════════════════════════════
section("1 │ LOADING DATASETS")

trader_path    = "historical_data.csv"
sentiment_path = "fear_greed.csv"

for p in [trader_path, sentiment_path]:
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"\n  [ERROR] '{p}' not found.\n"
            "  Place both CSV files in the same directory as this script."
        )

trader_raw    = pd.read_csv(trader_path)
sentiment_raw = pd.read_csv(sentiment_path)

sub("historical_data.csv shape  :", trader_raw.shape)
sub("fear_greed.csv shape        :", sentiment_raw.shape)

print(f"\n  Trader columns   : {list(trader_raw.columns)}")
print(f"  Sentiment columns: {list(sentiment_raw.columns)}")

print(f"\n  Trader — first 3 rows:\n{trader_raw.head(3).to_string()}")
print(f"\n  Sentiment — first 3 rows:\n{sentiment_raw.head(3).to_string()}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. CLEAN & PREPROCESS
# ══════════════════════════════════════════════════════════════════════════════
section("2 │ CLEANING & PREPROCESSING")

trader    = trader_raw.copy()
sentiment = sentiment_raw.copy()

# ── 2a. Trader: normalise column names ────────────────────────────────────────
trader.columns = trader.columns.str.strip()

# ── 2b. Timestamps ────────────────────────────────────────────────────────────
# Hyperliquid exports timestamps in milliseconds
if "Timestamp" in trader.columns:
    trader["Timestamp"] = pd.to_datetime(trader["Timestamp"], unit="ms", errors="coerce")
    trader["date"]      = trader["Timestamp"].dt.normalize().astype("datetime64[ns]")
    sub("✓ Trader 'Timestamp' (ms) → datetime")
elif "time" in trader.columns:
    trader["time"]  = pd.to_datetime(trader["time"], unit="ms", errors="coerce")
    trader["date"]  = trader["time"].dt.normalize().astype("datetime64[ns]")
    sub("✓ Trader 'time' (ms) → datetime")

# Fear & Greed — numeric unix second timestamps AND a human-readable 'date' col
if "timestamp" in sentiment.columns:
    sentiment["timestamp"] = pd.to_datetime(sentiment["timestamp"], unit="s", errors="coerce")
    sentiment["date"]      = sentiment["timestamp"].dt.normalize().astype("datetime64[ns]")
    sub("✓ Sentiment 'timestamp' (s) → datetime")
elif "date" in sentiment.columns:
    sentiment["date"] = pd.to_datetime(sentiment["date"], errors="coerce").dt.normalize().astype("datetime64[ns]")
    sub("✓ Sentiment 'date' → datetime")

# ── 2c. Numeric coercions ─────────────────────────────────────────────────────
NUMERIC_COLS = {
    "Closed PnL":    "closedPnl",
    "Size USD":      "sz",
    "Leverage":      None,
    "Entry Price":   None,
    "Exit Price":    None,
}

# Try multiple possible column name variants
def coerce_numeric(df: pd.DataFrame, candidates: list) -> str | None:
    for c in candidates:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            return c
    return None

pnl_col  = coerce_numeric(trader, ["Closed PnL", "closedPnl", "realized_pnl", "pnl"])
size_col = coerce_numeric(trader, ["Size USD", "sz", "size", "notional"])
lev_col  = coerce_numeric(trader, ["Leverage", "leverage", "lev"])

sub(f"✓ PnL column identified   : {pnl_col}")
sub(f"✓ Size column identified  : {size_col}")
sub(f"✓ Leverage column         : {lev_col}")

# Ensure sentiment value is numeric
sentiment["value"] = pd.to_numeric(sentiment.get("value", sentiment.get("Value", pd.Series())), errors="coerce")

# ── 2d. Classification column normalisation ───────────────────────────────────
CLASSIF_MAP = {
    "extreme fear": "Extreme Fear",
    "fear":         "Fear",
    "neutral":      "Neutral",
    "greed":        "Greed",
    "extreme greed":"Extreme Greed",
}

for col in ["classification", "Classification", "value_classification"]:
    if col in sentiment.columns:
        sentiment["classification"] = (
            sentiment[col].str.strip().str.lower().map(CLASSIF_MAP)
        )
        break

# ── 2e. Side / direction normalisation ───────────────────────────────────────
if "Side" in trader.columns:
    trader["Side"] = trader["Side"].str.strip().str.upper()
elif "side" in trader.columns:
    trader.rename(columns={"side": "Side"}, inplace=True)
    trader["Side"] = trader["Side"].str.strip().str.upper()

# ── 2f. Missing values ────────────────────────────────────────────────────────
null_before_trader    = trader.isna().sum().sum()
null_before_sentiment = sentiment.isna().sum().sum()

for col in trader.select_dtypes(include="number").columns:
    trader[col].fillna(trader[col].median(), inplace=True)

for col in sentiment.select_dtypes(include="number").columns:
    sentiment[col].fillna(sentiment[col].median(), inplace=True)

trader.dropna(subset=["date"], inplace=True)
sentiment.dropna(subset=["date", "classification"], inplace=True)

# Remove duplicate rows
trader.drop_duplicates(inplace=True)
sentiment.drop_duplicates(subset=["date"], inplace=True)   # one row per day

sub(f"✓ Trader nulls fixed       : {null_before_trader}")
sub(f"✓ Sentiment nulls fixed    : {null_before_sentiment}")
sub(f"✓ Trader rows after clean  : {len(trader):,}")
sub(f"✓ Sentiment rows after clean:{len(sentiment):,}")

# ── 2g. Derived columns ───────────────────────────────────────────────────────
trader["is_profit"]  = trader[pnl_col] > 0
trader["abs_pnl"]    = trader[pnl_col].abs()
if size_col:
    trader["pnl_pct"] = trader[pnl_col] / trader[size_col].replace(0, np.nan) * 100

# ══════════════════════════════════════════════════════════════════════════════
# 3. MERGE
# ══════════════════════════════════════════════════════════════════════════════
section("3 │ MERGING DATASETS")

trader_sorted    = trader.sort_values("date")
sentiment_sorted = sentiment[["date","classification","value"]].sort_values("date")

merged = pd.merge_asof(
    trader_sorted,
    sentiment_sorted,
    on="date",
    direction="backward"
)

merged.dropna(subset=["classification"], inplace=True)
merged["classification"] = pd.Categorical(
    merged["classification"], categories=SENTIMENT_ORDER, ordered=True
)

sub("Merge strategy    :", "asof (nearest prior sentiment)")
sub("Merged rows       :", f"{len(merged):,}")
sub("Sentiment coverage:", f"{merged['classification'].notna().mean()*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# 4. CORE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
section("4 │ CORE ANALYSIS")

# ── 4a. PnL by sentiment ──────────────────────────────────────────────────────
pnl_stats = (
    merged.groupby("classification", observed=True)[pnl_col]
    .agg(count="count", mean="mean", median="median", std="std", total="sum")
    .round(2)
)
print("\n  Average PnL by Sentiment:")
print(pnl_stats.to_string())

# ── 4b. Win rate by sentiment ─────────────────────────────────────────────────
win_rate = (
    merged.groupby("classification", observed=True)["is_profit"]
    .agg(win_rate="mean", total_trades="count")
)
win_rate["win_rate"] = (win_rate["win_rate"] * 100).round(1)
print("\n  Win Rate by Sentiment:")
print(win_rate.to_string())

# ── 4c. Trade activity ────────────────────────────────────────────────────────
print("\n  Trade Count by Sentiment:")
print(merged["classification"].value_counts().to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 5. ADVANCED INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
section("5 │ ADVANCED INSIGHTS")

# ── 5A. Win-rate matrix: sentiment × direction ────────────────────────────────
if "Side" in merged.columns:
    win_matrix = (
        merged.groupby(["classification", "Side"], observed=True)["is_profit"]
        .mean()
        .mul(100)
        .round(1)
        .unstack()
    )
    print("\n  A. Win-Rate Matrix (%) — Sentiment × Direction:")
    print(win_matrix.to_string())

# ── 5B. Leverage risk analysis ────────────────────────────────────────────────
if lev_col:
    lev_stats = (
        merged.groupby("classification", observed=True)[lev_col]
        .agg(mean_lev="mean", median_lev="median", max_lev="max")
        .round(2)
    )
    print("\n  B. Leverage by Sentiment:")
    print(lev_stats.to_string())

    # Correlation: leverage vs PnL
    lev_mask = merged[lev_col].notna() & merged[pnl_col].notna()
    lev_values = merged.loc[lev_mask, lev_col]
    pnl_values = merged.loc[lev_mask, pnl_col]
    if len(lev_values) >= 2 and lev_values.nunique() > 1 and pnl_values.nunique() > 1:
        corr_lev_pnl, p_val = stats.pearsonr(lev_values, pnl_values)
        sub(f"\n  Leverage ↔ PnL Pearson r : {corr_lev_pnl:.3f}  (p={p_val:.4f})")
    else:
        sub("\n  Leverage ↔ PnL Pearson r : insufficient variation for correlation")

# ── 5C. Trader consistency — Sharpe-like score ───────────────────────────────
ACCOUNT_COL = next(
    (c for c in merged.columns if c.lower() in ["account", "trader", "user", "address"]),
    None
)

if ACCOUNT_COL:
    trader_stats = (
        merged.groupby(ACCOUNT_COL)[pnl_col]
        .agg(total_pnl="sum", mean_pnl="mean", std_pnl="std", trades="count")
    )
    trader_stats["sharpe_score"] = (
        trader_stats["mean_pnl"] / trader_stats["std_pnl"].replace(0, np.nan)
    ).round(3)

    top10_pnl    = trader_stats.nlargest(10, "total_pnl")
    top10_sharpe = trader_stats[trader_stats["trades"] >= 10].nlargest(10, "sharpe_score")

    print("\n  C. Top 10 Traders by Total PnL:")
    print(top10_pnl.to_string())

    print("\n  Top 10 Traders by Consistency (Sharpe-like, min 10 trades):")
    print(top10_sharpe.to_string())

# ── 5D. Contrarian signal: PnL in Extreme Fear vs Extreme Greed ───────────────
print("\n  D. Contrarian Signal Analysis:")
for cat in ["Extreme Fear", "Extreme Greed"]:
    subset = merged[merged["classification"] == cat][pnl_col]
    if len(subset) > 0:
        sub(f"  {cat}: mean={subset.mean():.2f}  median={subset.median():.2f}  "
            f"win%={subset.gt(0).mean()*100:.1f}  n={len(subset)}")

# ── 5E. Symbol-level breakdown ────────────────────────────────────────────────
SYM_COL = next(
    (c for c in merged.columns if c.lower() in ["coin", "symbol", "market", "asset", "pair"]),
    None
)

if SYM_COL:
    sym_stats = (
        merged.groupby(SYM_COL)[pnl_col]
        .agg(total="sum", mean="mean", trades="count", win_rate=lambda x: (x > 0).mean())
        .sort_values("total", ascending=False)
        .head(10)
        .round(2)
    )
    print("\n  E. Top 10 Symbols by Total PnL:")
    print(sym_stats.to_string())

# ── 5F. Time-of-day analysis ─────────────────────────────────────────────────
if "Timestamp" in merged.columns:
    merged["hour"] = pd.to_datetime(merged["Timestamp"]).dt.hour
    hourly = merged.groupby("hour")[pnl_col].agg(mean="mean", count="count").round(2)
    print("\n  F. Best/Worst Trading Hours:")
    print(hourly.sort_values("mean", ascending=False).head(5).to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 6. VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════
section("6 │ GENERATING VISUALISATIONS")

sent_colors = [PALETTE.get(s, "#888") for s in SENTIMENT_ORDER]

# ── Chart 1: Average PnL by Sentiment ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
means = pnl_stats.reindex(SENTIMENT_ORDER)["mean"]
bars  = ax.bar(means.index, means.values,
               color=sent_colors, edgecolor="#0d1117", linewidth=0.8, zorder=3)
ax.axhline(0, color="#58a6ff", linewidth=0.8, linestyle="--")
ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
ax.set_title("Average Closed PnL by Market Sentiment")
ax.set_xlabel("Market Sentiment")
ax.set_ylabel("Average Closed PnL (USD)")
ax.grid(axis="y", zorder=0)
plt.tight_layout()
plt.savefig("chart1_avg_pnl_by_sentiment.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✓ Chart 1 saved")

# ── Chart 2: Win Rate by Sentiment ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
wr = win_rate.reindex(SENTIMENT_ORDER)["win_rate"]
bars = ax.bar(wr.index, wr.values,
              color=sent_colors, edgecolor="#0d1117", linewidth=0.8, zorder=3)
ax.axhline(50, color="#58a6ff", linewidth=0.8, linestyle="--", label="50% baseline")
ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
ax.set_title("Win Rate (%) by Market Sentiment")
ax.set_ylabel("Win Rate (%)")
ax.set_ylim(0, 100)
ax.legend()
ax.grid(axis="y", zorder=0)
plt.tight_layout()
plt.savefig("chart2_win_rate_by_sentiment.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✓ Chart 2 saved")

# ── Chart 3: Trade Count by Sentiment ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
counts = merged["classification"].value_counts().reindex(SENTIMENT_ORDER)
bars = ax.bar(counts.index, counts.values,
              color=sent_colors, edgecolor="#0d1117", linewidth=0.8, zorder=3)
ax.bar_label(bars, fmt="%d", padding=4, fontsize=9)
ax.set_title("Number of Trades per Sentiment Regime")
ax.set_ylabel("Trade Count")
ax.grid(axis="y", zorder=0)
plt.tight_layout()
plt.savefig("chart3_trade_count_by_sentiment.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✓ Chart 3 saved")

# ── Chart 4: PnL Distribution (KDE + hist) ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
data_clipped = merged[pnl_col].clip(
    lower=merged[pnl_col].quantile(0.01),
    upper=merged[pnl_col].quantile(0.99)
)
sns.histplot(data_clipped, bins=60, kde=True, ax=ax,
             color="#58a6ff", edgecolor="#0d1117", linewidth=0.4,
             line_kws={"color": "#f8c000", "lw": 2})
ax.axvline(0, color="#ff4444", linewidth=1.2, linestyle="--", label="Break-even")
ax.set_title("Distribution of Closed PnL (1st–99th percentile)")
ax.set_xlabel("Closed PnL (USD)")
ax.legend()
ax.grid(axis="y")
plt.tight_layout()
plt.savefig("chart4_pnl_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✓ Chart 4 saved")

# ── Chart 5: Profit vs Loss stacked bar ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
pnl_cat = merged.groupby(["classification", "is_profit"], observed=True).size().unstack()
pnl_cat.columns = ["Loss", "Profit"]
pnl_cat = pnl_cat.reindex(SENTIMENT_ORDER)
pnl_cat[["Loss", "Profit"]].plot(
    kind="bar", stacked=True, ax=ax,
    color=["#d62728", "#2ca02c"], edgecolor="#0d1117", linewidth=0.6
)
ax.set_title("Profit vs Loss Count by Market Sentiment")
ax.set_xlabel("Market Sentiment")
ax.set_ylabel("Number of Trades")
ax.legend(loc="upper right")
ax.grid(axis="y")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("chart5_profit_vs_loss.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✓ Chart 5 saved")

# ── Chart 6: Average Trade Size by Sentiment ─────────────────────────────────
if size_col:
    fig, ax = plt.subplots(figsize=(9, 5))
    avg_size = merged.groupby("classification", observed=True)[size_col].mean().reindex(SENTIMENT_ORDER)
    bars = ax.bar(avg_size.index, avg_size.values,
                  color=sent_colors, edgecolor="#0d1117", linewidth=0.8, zorder=3)
    ax.bar_label(bars, fmt="%.0f", padding=4, fontsize=9)
    ax.set_title("Average Trade Size (USD) by Market Sentiment")
    ax.set_ylabel("Trade Size (USD)")
    ax.grid(axis="y", zorder=0)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("chart6_avg_trade_size.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  ✓ Chart 6 saved")

# ── Chart 7: Risk (PnL Volatility) by Sentiment ──────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
risk = pnl_stats.reindex(SENTIMENT_ORDER)["std"]
bars = ax.bar(risk.index, risk.values,
              color=sent_colors, edgecolor="#0d1117", linewidth=0.8, zorder=3)
ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
ax.set_title("PnL Volatility (Risk) by Market Sentiment")
ax.set_ylabel("Standard Deviation of Closed PnL")
ax.grid(axis="y", zorder=0)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("chart7_risk_by_sentiment.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✓ Chart 7 saved")

# ── Chart 8: Top 10 Traders by Total PnL ─────────────────────────────────────
if ACCOUNT_COL:
    fig, ax = plt.subplots(figsize=(10, 5))
    colors_bar = ["#2ca02c" if v >= 0 else "#d62728" for v in top10_pnl["total_pnl"]]
    ax.barh(range(len(top10_pnl)), top10_pnl["total_pnl"].values,
            color=colors_bar, edgecolor="#0d1117", linewidth=0.6)
    ax.set_yticks(range(len(top10_pnl)))
    # Shorten wallet addresses for readability
    labels = [str(a)[:12] + "…" if len(str(a)) > 14 else str(a)
              for a in top10_pnl.index]
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="#8b949e", linewidth=0.8)
    ax.set_title("Top 10 Traders by Total Closed PnL")
    ax.set_xlabel("Total Closed PnL (USD)")
    ax.grid(axis="x")
    plt.tight_layout()
    plt.savefig("chart8_top_traders.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  ✓ Chart 8 saved")

# ── Chart 9: Daily Average PnL Trend coloured by Sentiment ───────────────────
fig, ax = plt.subplots(figsize=(13, 5))
daily_pnl = (
    merged.groupby(["date", "classification"], observed=True)[pnl_col]
    .mean()
    .reset_index()
    .sort_values("date")
)
for sentiment_cat, color in PALETTE.items():
    sub_df = daily_pnl[daily_pnl["classification"] == sentiment_cat]
    if not sub_df.empty:
        ax.plot(sub_df["date"], sub_df[pnl_col],
                label=sentiment_cat, color=color, linewidth=1.5, alpha=0.85)
ax.axhline(0, color="#58a6ff", linewidth=0.7, linestyle="--")
ax.set_title("Daily Average Closed PnL — Coloured by Sentiment Regime")
ax.set_xlabel("Date")
ax.set_ylabel("Avg PnL (USD)")
ax.legend(loc="upper left", fontsize=9)
ax.grid()
plt.xticks(rotation=40, ha="right")
plt.tight_layout()
plt.savefig("chart9_daily_pnl_trend.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✓ Chart 9 saved")

# ── Chart 10: PnL Heatmap — Hour of Day × Sentiment ─────────────────────────
if "hour" in merged.columns:
    heat_data = (
        merged.groupby(["hour", "classification"], observed=True)[pnl_col]
        .mean()
        .unstack()
        .reindex(columns=SENTIMENT_ORDER)
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(
        heat_data, ax=ax, cmap="RdYlGn", center=0,
        linewidths=0.4, linecolor="#0d1117",
        cbar_kws={"label": "Avg PnL (USD)"},
        annot=True, fmt=".1f", annot_kws={"size": 8}
    )
    ax.set_title("Average PnL Heatmap — Hour of Day × Market Sentiment")
    ax.set_xlabel("Market Sentiment")
    ax.set_ylabel("Hour of Day (UTC)")
    plt.tight_layout()
    plt.savefig("chart10_pnl_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  ✓ Chart 10 saved")

# ══════════════════════════════════════════════════════════════════════════════
# 7. EXPORT
# ══════════════════════════════════════════════════════════════════════════════
section("7 │ EXPORTING ARTEFACTS")

merged.to_csv("final_output.csv", index=False)
sub("✓ final_output.csv saved")

# Summary stats export
summary_rows = []
for cat in SENTIMENT_ORDER:
    subset = merged[merged["classification"] == cat]
    if len(subset) == 0:
        continue
    row = {
        "Sentiment":    cat,
        "Trades":       len(subset),
        "Total_PnL":    subset[pnl_col].sum().round(2),
        "Avg_PnL":      subset[pnl_col].mean().round(2),
        "Median_PnL":   subset[pnl_col].median().round(2),
        "Std_PnL":      subset[pnl_col].std().round(2),
        "Win_Rate_%":   round(subset["is_profit"].mean() * 100, 1),
    }
    if size_col:
        row["Avg_Size_USD"] = round(subset[size_col].mean(), 2)
    if lev_col:
        row["Avg_Leverage"] = round(subset[lev_col].mean(), 2)
    summary_rows.append(row)

pd.DataFrame(summary_rows).to_csv("sentiment_summary.csv", index=False)
sub("✓ sentiment_summary.csv saved")

section("ANALYSIS COMPLETE ✅")
print("""
  Key outputs:
    📄 final_output.csv         — merged trade+sentiment dataset
    📄 sentiment_summary.csv    — aggregated stats per sentiment class
    📊 chart1–10 .png files     — all visualisations

  Key questions answered:
    ✔ Do traders earn more during Fear or Greed?
    ✔ Does win-rate vary by sentiment regime?
    ✔ Are traders more aggressive (larger size/leverage) in Greed?
    ✔ Which traders are consistent vs lucky?
    ✔ Which symbols perform best under each sentiment?
    ✔ What hours of the day are most profitable?
""")
