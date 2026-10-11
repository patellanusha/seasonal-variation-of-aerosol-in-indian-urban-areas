"""
Exploratory Data Analysis for Seasonal AOD-Temperature Project
-----------------------------------------------------------------
Reads output/Combined_Dataset_Imputed.csv and generates all required
EDA visualizations, saved as high-res PNGs in output/eda_plots/.

Run this AFTER data_preprocessing.py has produced Combined_Dataset_Imputed.csv.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# -----------------------------
# Setup
# -----------------------------
INPUT_FILE = Path("output/Combined_Dataset_Imputed.csv")
PLOT_DIR = Path("output/eda_plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

SEASON_ORDER = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]


def load_data(exclude_fallback=True):
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"{INPUT_FILE} not found. Run data_preprocessing.py first."
        )
    df = pd.read_csv(INPUT_FILE, parse_dates=["Date"])

    # Rows imputed via 'city_mean_fallback' (e.g. Mangaluru Monsoon, which
    # had ZERO real AOD readings for that season in any year) are not
    # actual season measurements -- they're a city-wide average standing
    # in for a season with no data at all. Including them in seasonal
    # comparisons or correlation analysis would misrepresent them as real
    # Monsoon readings. Excluded from EDA/ML by default; set
    # exclude_fallback=False to include them anyway (e.g. for auditing).
    if exclude_fallback and "AOD_Imputation_Method" in df.columns:
        n_before = len(df)
        fallback_mask = df["AOD_Imputation_Method"] == "city_mean_fallback"
        n_excluded = fallback_mask.sum()
        if n_excluded > 0:
            excluded_summary = (
                df.loc[fallback_mask, ["City", "Season"]]
                .value_counts()
                .to_dict()
            )
            print(f"[INFO] Excluding {n_excluded} rows imputed via "
                  f"city_mean_fallback (no real seasonal data available):")
            for (city, season), count in excluded_summary.items():
                print(f"  {city} / {season}: {count} rows excluded")
            df = df[~fallback_mask].copy()
            print(f"[INFO] Dataset: {n_before} rows -> {len(df)} rows after exclusion\n")

    # Use every city actually present in the data (alphabetical), not a
    # hardcoded list -- so newly added cities are picked up automatically
    # instead of silently dropped.
    cities_present = sorted(df["City"].unique().tolist())
    df["City"] = pd.Categorical(df["City"], categories=cities_present, ordered=True)
    df["Season"] = pd.Categorical(df["Season"], categories=SEASON_ORDER, ordered=True)

    return df, cities_present


def get_palette(n_cities):
    # "tab20" gives 20 visually distinct colors -- scales to a much larger
    # city set than "Set2" (8 colors) without repeating/confusing colors.
    return sns.color_palette("tab20", n_colors=max(n_cities, 3))


def save(fig, name):
    path = PLOT_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved: {path}")


# -----------------------------
# 1. Monthly AOD trend (faceted small multiples -- scales to many cities)
# -----------------------------
def plot_monthly_aod_trend(df, cities):
    n_cols = 3 if len(cities) > 4 else min(len(cities), 3)
    g = sns.relplot(
        data=df, x="Date", y="AOD", col="City", col_wrap=n_cols,
        kind="line", color="steelblue", height=3.2, aspect=1.6,
        facet_kws={"sharey": True},
    )
    g.set_titles("{col_name}")
    g.fig.suptitle("Monthly AOD Trend by City — 2012–2020", y=1.02)
    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=30)
    g.savefig(PLOT_DIR / "01_monthly_aod_trend.png", bbox_inches="tight", dpi=300)
    plt.close(g.fig)
    print(f"  saved: {PLOT_DIR / '01_monthly_aod_trend.png'}")


# -----------------------------
# 2. Monthly temperature trend (faceted small multiples)
# -----------------------------
def plot_monthly_temp_trend(df, cities):
    n_cols = 3 if len(cities) > 4 else min(len(cities), 3)
    g = sns.relplot(
        data=df, x="Date", y="Temperature_C", col="City", col_wrap=n_cols,
        kind="line", color="firebrick", height=3.2, aspect=1.6,
        facet_kws={"sharey": True},
    )
    g.set_titles("{col_name}")
    g.set_ylabels("Temperature (°C)")
    g.fig.suptitle("Monthly Temperature Trend by City — 2012–2020", y=1.02)
    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=30)
    g.savefig(PLOT_DIR / "02_monthly_temperature_trend.png", bbox_inches="tight", dpi=300)
    plt.close(g.fig)
    print(f"  saved: {PLOT_DIR / '02_monthly_temperature_trend.png'}")


# -----------------------------
# 3. Seasonal average AOD (bar, grouped by city)
# -----------------------------
def plot_seasonal_avg_aod(df, palette):
    fig, ax = plt.subplots(figsize=(max(12, len(df["City"].cat.categories) * 0.8), 6))
    sns.barplot(data=df, x="Season", y="AOD", hue="City",
                order=SEASON_ORDER, palette=palette, ax=ax, errorbar="sd")
    ax.set_title("Seasonal Average AOD by City")
    ax.set_xlabel("Season")
    ax.set_ylabel("Mean AOD")
    ax.legend(title="City", bbox_to_anchor=(1.02, 1), loc="upper left", ncol=1, fontsize=9)
    save(fig, "03_seasonal_avg_aod")


# -----------------------------
# 4. Seasonal average temperature (bar, grouped by city)
# -----------------------------
def plot_seasonal_avg_temp(df, palette):
    fig, ax = plt.subplots(figsize=(max(12, len(df["City"].cat.categories) * 0.8), 6))
    sns.barplot(data=df, x="Season", y="Temperature_C", hue="City",
                order=SEASON_ORDER, palette=palette, ax=ax, errorbar="sd")
    ax.set_title("Seasonal Average Temperature by City")
    ax.set_xlabel("Season")
    ax.set_ylabel("Mean Temperature (°C)")
    ax.legend(title="City", bbox_to_anchor=(1.02, 1), loc="upper left", ncol=1, fontsize=9)
    save(fig, "04_seasonal_avg_temperature")


# -----------------------------
# 5. City-wise comparison (AOD and Temp side by side)
# -----------------------------
def plot_citywise_comparison(df, palette):
    n_cities = len(df["City"].cat.categories)
    fig, axes = plt.subplots(1, 2, figsize=(max(16, n_cities * 1.2), 6))
    sns.boxplot(data=df, x="City", y="AOD", palette=palette, hue="City", legend=False, ax=axes[0])
    axes[0].set_title("City-wise AOD Distribution (2012–2020)")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=45)

    sns.boxplot(data=df, x="City", y="Temperature_C", palette=palette, hue="City", legend=False, ax=axes[1])
    axes[1].set_title("City-wise Temperature Distribution (2012–2020)")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Temperature (°C)")
    axes[1].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    save(fig, "05_citywise_comparison")


# -----------------------------
# 6. Correlation matrix (numeric features)
# -----------------------------
def plot_correlation_matrix(df):
    numeric_cols = ["AOD", "Temperature_C", "Year", "Month"]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix (All Cities Combined)")
    save(fig, "06_correlation_matrix")

    # Per-city correlation between AOD and Temperature specifically
    print("\nAOD vs Temperature correlation by city (all seasons pooled):")
    for city in df["City"].cat.categories:
        sub = df[df["City"] == city]
        r = sub["AOD"].corr(sub["Temperature_C"])
        print(f"  {city}: r = {r:.3f}")


# -----------------------------
# 6b. Season-specific correlation (City x Season)
# -----------------------------
# A single pooled correlation across all 12 months can be misleading:
# Winter (low AOD, low temp), Summer (low AOD, high temp), Monsoon (low
# AOD from rain washout, moderate temp), and Post-Monsoon (often highest
# AOD) each pull a pooled correlation in different directions. Computing
# AOD-Temperature correlation SEPARATELY within each season removes that
# confound and is the more scientifically meaningful number for a study
# specifically about seasonal variation.
def plot_seasonal_correlation(df):
    records = []
    for city in df["City"].cat.categories:
        for season in SEASON_ORDER:
            sub = df[(df["City"] == city) & (df["Season"] == season)]
            if len(sub) >= 3:  # need at least a few points for a meaningful r
                r = sub["AOD"].corr(sub["Temperature_C"])
            else:
                r = np.nan
            records.append({"City": city, "Season": season, "r": r})

    corr_table = pd.DataFrame(records).pivot(index="City", columns="Season", values="r")
    corr_table = corr_table[SEASON_ORDER]  # enforce column order

    fig, ax = plt.subplots(figsize=(9, max(6, len(corr_table) * 0.5)))
    sns.heatmap(corr_table, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title("AOD vs Temperature Correlation by City and Season")
    save(fig, "11_seasonal_correlation_heatmap")

    print("\nAOD vs Temperature correlation by City x Season:")
    print(corr_table.round(3).to_string())

    corr_table.to_csv(PLOT_DIR / "seasonal_correlation_table.csv")
    print(f"  saved: {PLOT_DIR / 'seasonal_correlation_table.csv'}")


# -----------------------------
# 7. Scatter plot: AOD vs Temperature (faceted by city, colored by season)
# -----------------------------
def plot_aod_vs_temp_scatter(df, cities):
    n_cols = 3 if len(cities) > 4 else min(len(cities), 3)
    g = sns.relplot(
        data=df, x="AOD", y="Temperature_C", hue="Season", hue_order=SEASON_ORDER,
        col="City", col_wrap=n_cols, kind="scatter", alpha=0.7,
        palette="viridis", height=3.2, aspect=1.3,
    )
    g.set_titles("{col_name}")
    g.set_ylabels("Temperature (°C)")
    g.fig.suptitle("AOD vs Temperature by City (colored by Season)", y=1.02)
    g.savefig(PLOT_DIR / "07_aod_vs_temperature_scatter.png", bbox_inches="tight", dpi=300)
    plt.close(g.fig)
    print(f"  saved: {PLOT_DIR / '07_aod_vs_temperature_scatter.png'}")


# -----------------------------
# 8. Boxplots for seasonal variation (AOD and Temp)
# -----------------------------
def plot_seasonal_boxplots(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.boxplot(data=df, x="Season", y="AOD", order=SEASON_ORDER,
                hue="Season", legend=False, palette="Blues", ax=axes[0])
    axes[0].set_title("Seasonal Variation in AOD (All Cities)")

    sns.boxplot(data=df, x="Season", y="Temperature_C", order=SEASON_ORDER,
                hue="Season", legend=False, palette="Reds", ax=axes[1])
    axes[1].set_title("Seasonal Variation in Temperature (All Cities)")
    axes[1].set_ylabel("Temperature (°C)")

    fig.tight_layout()
    save(fig, "08_seasonal_boxplots")


# -----------------------------
# 9. Heatmap: City x Month average AOD / Temperature
# -----------------------------
def plot_heatmaps(df):
    aod_pivot = df.pivot_table(index="City", columns="Month", values="AOD", aggfunc="mean", observed=True)
    temp_pivot = df.pivot_table(index="City", columns="Month", values="Temperature_C", aggfunc="mean", observed=True)

    n_cities = len(aod_pivot)
    row_height = max(0.5, 4.5 / max(n_cities, 5))
    fig, axes = plt.subplots(2, 1, figsize=(12, row_height * n_cities * 2 + 2))
    sns.heatmap(aod_pivot, annot=True, fmt=".2f", cmap="YlOrRd", ax=axes[0])
    axes[0].set_title("Average AOD by City and Month")
    axes[0].set_xlabel("Month")

    sns.heatmap(temp_pivot, annot=True, fmt=".1f", cmap="coolwarm", ax=axes[1])
    axes[1].set_title("Average Temperature (°C) by City and Month")
    axes[1].set_xlabel("Month")

    fig.tight_layout()
    save(fig, "09_city_month_heatmaps")


# -----------------------------
# 10. Year-wise seasonal trend (time series faceted by season)
# -----------------------------
def plot_yearly_seasonal_trend(df, palette):
    yearly_season = df.groupby(["Year", "Season", "City"], observed=True)["AOD"].mean().reset_index()

    g = sns.relplot(
        data=yearly_season, x="Year", y="AOD", hue="City",
        col="Season", col_order=SEASON_ORDER, col_wrap=2,
        kind="line", marker="o", palette=palette, height=4, aspect=1.3,
    )
    g.fig.suptitle("Year-wise AOD Trend by Season", y=1.03)
    sns.move_legend(g, "upper left", bbox_to_anchor=(1.02, 0.9), ncol=1, fontsize=9)
    g.savefig(PLOT_DIR / "10_yearly_seasonal_aod_trend.png", bbox_inches="tight", dpi=300)
    plt.close(g.fig)
    print(f"  saved: {PLOT_DIR / '10_yearly_seasonal_aod_trend.png'}")


# -----------------------------
# Main
# -----------------------------
def main():
    df, cities = load_data()
    print(f"Loaded {len(df)} records across {len(cities)} cities: {cities}\n")
    palette = get_palette(len(cities))

    print("Generating plots...")
    plot_monthly_aod_trend(df, cities)
    plot_monthly_temp_trend(df, cities)
    plot_seasonal_avg_aod(df, palette)
    plot_seasonal_avg_temp(df, palette)
    plot_citywise_comparison(df, palette)
    plot_correlation_matrix(df)
    plot_seasonal_correlation(df)
    plot_aod_vs_temp_scatter(df, cities)
    plot_seasonal_boxplots(df)
    plot_heatmaps(df)
    plot_yearly_seasonal_trend(df, palette)

    print(f"\nAll plots saved to: {PLOT_DIR.resolve()}")

    # Quick descriptive stats table, useful for the report
    summary = df.groupby("City", observed=True)[["AOD", "Temperature_C"]].agg(["mean", "std", "min", "max"])
    summary.to_csv(PLOT_DIR / "summary_statistics.csv")
    print(f"Descriptive statistics saved to: {PLOT_DIR / 'summary_statistics.csv'}")


if __name__ == "__main__":
    main()
