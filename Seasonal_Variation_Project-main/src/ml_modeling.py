"""
Machine Learning Modeling for Seasonal AOD-Temperature Project
-----------------------------------------------------------------
Reads output/Combined_Dataset_Imputed.csv, trains several regression
models to predict Temperature_C from AOD + City + Season + Year + Month,
and compares them using MAE, RMSE, and R^2.

Run this AFTER eda_analysis.py (or at least data_preprocessing.py).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from pathlib import Path
from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[warn] xgboost not installed -- skipping XGBoost model.")
    print("       Install with: pip install xgboost\n")

# -----------------------------
# Setup
# -----------------------------
INPUT_FILE = Path("output/Combined_Dataset_Imputed.csv")
OUTPUT_DIR = Path("output/ml_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

RANDOM_STATE = 42
TEST_SIZE = 0.2

NUMERIC_FEATURES = ["AOD", "Year", "Month"]
CATEGORICAL_FEATURES = ["City", "Season"]
TARGET = "Temperature_C"


# -----------------------------
# Load data
# -----------------------------
def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"{INPUT_FILE} not found. Run data_preprocessing.py first.")

    df = pd.read_csv(INPUT_FILE, parse_dates=["Date"])

    # Same exclusion as EDA: rows where AOD is a city-wide fallback average
    # (e.g. Mangaluru Monsoon, which has zero real readings for that
    # season) are not real measurements and would let the model "learn"
    # a partly circular relationship. Excluded from training/evaluation.
    if "AOD_Imputation_Method" in df.columns:
        n_before = len(df)
        fallback_mask = df["AOD_Imputation_Method"] == "city_mean_fallback"
        n_excluded = fallback_mask.sum()
        if n_excluded > 0:
            print(f"[INFO] Excluding {n_excluded} city_mean_fallback rows "
                  f"(no real seasonal AOD data) from ML training/evaluation.")
            df = df[~fallback_mask].copy()
            print(f"[INFO] Dataset: {n_before} rows -> {len(df)} rows after exclusion\n")

    return df


# -----------------------------
# Build preprocessing + model pipelines
# -----------------------------
def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_models():
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=8),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, max_depth=12
        ),
        "SVR": SVR(kernel="rbf", C=10, epsilon=0.2),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            random_state=RANDOM_STATE, n_jobs=-1
        )
    return models


def build_final_model():
    """
    The single model used for all follow-up experiments (ablation, AOD
    anomaly, lagged AOD) after the initial 5-model comparison. Random
    Forest was selected for its balance of accuracy and interpretability
    (feature_importances_ is what surfaced the City-confounding issue).
    """
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, max_depth=12
        )
    }


# -----------------------------
# Train, evaluate, compare
# -----------------------------
def evaluate_model(name, pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"  {name:<20} MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.4f}")
    return {"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2}, y_pred


def plot_actual_vs_predicted(y_test, y_pred, model_name):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, alpha=0.4, s=25, color="steelblue")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Temperature (°C)")
    ax.set_ylabel("Predicted Temperature (°C)")
    ax.set_title(f"Actual vs Predicted — {model_name}")
    ax.legend()
    fig.savefig(OUTPUT_DIR / f"actual_vs_predicted_{model_name.replace(' ', '_')}.png",
                bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_model_comparison(results_df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ["MAE", "RMSE", "R2"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for ax, metric, color in zip(axes, metrics, colors):
        ordered = results_df.sort_values(metric, ascending=(metric != "R2"))
        ax.barh(ordered["Model"], ordered[metric], color=color)
        ax.set_title(metric)
        ax.set_xlabel(metric)

    fig.suptitle("Model Comparison", y=1.03)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_comparison.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_feature_importance(pipeline, model_name):
    """Only meaningful for tree-based models with .feature_importances_"""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return

    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = (
        NUMERIC_FEATURES
        + list(preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES))
    )
    importances = model.feature_importances_

    imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    imp_df = imp_df.sort_values("Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.barplot(data=imp_df.head(15), x="Importance", y="Feature", color="steelblue", ax=ax)
    ax.set_title(f"Top 15 Feature Importances — {model_name}")
    fig.savefig(OUTPUT_DIR / f"feature_importance_{model_name.replace(' ', '_')}.png",
                bbox_inches="tight", dpi=300)
    plt.close(fig)

    # Print grouped importance: how much predictive power comes from AOD
    # itself vs. City identity vs. Season vs. Year/Month (climatology).
    # This is the real check on whether the model learned an AOD effect
    # or is mostly just memorizing each city's normal temperature.
    def group_of(feat):
        if feat == "AOD":
            return "AOD"
        if feat.startswith("City_"):
            return "City (climatology proxy)"
        if feat.startswith("Season_"):
            return "Season"
        return "Year/Month"

    imp_df["Group"] = imp_df["Feature"].apply(group_of)
    grouped = imp_df.groupby("Group")["Importance"].sum().sort_values(ascending=False)
    print(f"\n  [{model_name}] Importance by group (should sum to ~1.0):")
    for group, val in grouped.items():
        print(f"    {group:<28} {val:.3f}")


# -----------------------------
# Lagged AOD: does month t's AOD predict month t+1's temperature?
# -----------------------------
# Aerosol radiative effects on surface temperature aren't necessarily
# instantaneous -- particles can persist in the atmosphere and influence
# heating/cooling over subsequent weeks. This tests a DIFFERENT mechanism
# than same-month AOD: does an unusually hazy month predict the FOLLOWING
# month's temperature (anomaly), rather than that same month's.
def add_lagged_aod(df):
    df = df.sort_values(["City", "Date"]).copy()
    df["AOD_Lag1"] = df.groupby("City")["AOD"].shift(1)
    df["AOD_Anomaly_Lag1"] = df.groupby("City")["AOD_Anomaly"].shift(1)
    # Dropping the first month of each city's series (no lag available)
    df = df.dropna(subset=["AOD_Lag1", "AOD_Anomaly_Lag1"]).copy()
    return df


# -----------------------------
# Significance-tested correlation helper
# -----------------------------
# With 13 cities tested at once, some correlations will look "significant"
# by chance alone even if there's no real effect (the multiple-comparisons
# problem). A Bonferroni-corrected threshold (alpha=0.05 / 13 cities)
# is used to flag which per-city results actually survive that correction,
# alongside the uncorrected p-value for reference.
N_CITIES_TESTED = 13
ALPHA = 0.05
BONFERRONI_ALPHA = ALPHA / N_CITIES_TESTED


def corr_with_significance(x, y):
    """Returns (r, p) using scipy's pearsonr, dropping NaN pairs."""
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return np.nan, np.nan
    r, p = pearsonr(x[mask], y[mask])
    return r, p


def significance_flag(p):
    if pd.isna(p):
        return ""
    if p < BONFERRONI_ALPHA:
        return "**"  # survives Bonferroni correction across 13 cities
    if p < ALPHA:
        return "*"   # significant uncorrected, but not after correction
    return ""


def print_lagged_correlation(df):
    print("\n=================================")
    print("LAGGED AOD vs TEMPERATURE (does month t's AOD predict month t+1's temp?)")
    print("=================================")

    r_raw, p_raw = corr_with_significance(df["AOD_Lag1"], df["Temperature_C"])
    r_anom, p_anom = corr_with_significance(df["AOD_Anomaly_Lag1"], df["Temp_Anomaly"])
    print(f"\nOverall: raw AOD(t-1) vs Temp(t): r = {r_raw:.3f}, p = {p_raw:.4f}")
    print(f"Overall: AOD_Anomaly(t-1) vs Temp_Anomaly(t): r = {r_anom:.3f}, p = {p_anom:.4f}")

    print(f"\nBy city (AOD_Anomaly lag-1 vs Temp_Anomaly), n~{len(df) // N_CITIES_TESTED} per city:")
    print(f"  ** = significant after Bonferroni correction (p < {BONFERRONI_ALPHA:.4f})")
    print(f"  *  = significant uncorrected only (p < {ALPHA}), not robust to multiple comparisons")
    for city in sorted(df["City"].unique()):
        sub = df[df["City"] == city]
        r, p = corr_with_significance(sub["AOD_Anomaly_Lag1"], sub["Temp_Anomaly"])
        flag = significance_flag(p)
        print(f"  {city:<12} r = {r:6.3f}   p = {p:.4f}  {flag}")


# -----------------------------
# AOD Anomaly: isolate AOD's direct effect from season/city collinearity
# -----------------------------
# Raw AOD is highly collinear with Season and City (e.g. winter inversions
# systematically trap aerosols), so once the model already knows City+
# Season+Month, raw AOD adds little new information -- even if AOD and
# temperature are correlated within a season. The AOD ANOMALY --
# how much higher/lower AOD was than that City+Season's own average --
# strips out the season/city baseline and asks a sharper question:
# "was this particular month unusually hazy for its season, and if so,
# was it also unusually warm/cool?" This is a cleaner test of AOD's own
# direct association with temperature.
def add_anomaly_columns(df):
    df = df.copy()
    aod_group_mean = df.groupby(["City", "Season"])["AOD"].transform("mean")
    temp_group_mean = df.groupby(["City", "Season"])["Temperature_C"].transform("mean")
    df["AOD_Anomaly"] = df["AOD"] - aod_group_mean
    df["Temp_Anomaly"] = df["Temperature_C"] - temp_group_mean
    return df


def print_anomaly_correlation(df):
    print("\n=================================")
    print("AOD ANOMALY vs TEMPERATURE ANOMALY")
    print("(deviation from each City+Season's own average -- isolates")
    print(" AOD's direct association, removing season/city baseline)")
    print("=================================")

    overall_r, overall_p = corr_with_significance(df["AOD_Anomaly"], df["Temp_Anomaly"])
    print(f"\nOverall (all cities pooled): r = {overall_r:.3f}, p = {overall_p:.4f}")

    print(f"\nBy city (n~{len(df) // N_CITIES_TESTED} per city):")
    print(f"  ** = significant after Bonferroni correction (p < {BONFERRONI_ALPHA:.4f})")
    print(f"  *  = significant uncorrected only (p < {ALPHA}), not robust to multiple comparisons")
    for city in sorted(df["City"].unique()):
        sub = df[df["City"] == city]
        r, p = corr_with_significance(sub["AOD_Anomaly"], sub["Temp_Anomaly"])
        flag = significance_flag(p)
        print(f"  {city:<12} r = {r:6.3f}   p = {p:.4f}  {flag}")


def run_anomaly_model(df):
    print("\n=================================")
    print("MODELS USING AOD ANOMALY (instead of raw AOD)")
    print("=================================")

    features = ["AOD_Anomaly", "Year", "Month", "City", "Season"]
    X = df[features]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["AOD_Anomaly", "Year", "Month"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["City", "Season"]),
        ]
    )
    models = build_final_model()  # Random Forest only, per project decision

    results = []
    best_pipeline = None
    best_r2 = -np.inf
    best_name = None

    for name, model in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        result, _ = evaluate_model(name, pipeline, X_train, X_test, y_train, y_test)
        results.append(result)
        if result["R2"] > best_r2:
            best_r2 = result["R2"]
            best_pipeline = pipeline
            best_name = name

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
    print("\nAOD-Anomaly model comparison (sorted by R2):")
    print(results_df.to_string(index=False))
    results_df.to_csv(OUTPUT_DIR / "model_comparison_results_aod_anomaly.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'model_comparison_results_aod_anomaly.csv'}")

    # Check whether AOD_Anomaly's importance is higher than raw AOD was
    if hasattr(best_pipeline.named_steps["model"], "feature_importances_"):
        model = best_pipeline.named_steps["model"]
        preproc = best_pipeline.named_steps["preprocessor"]
        feat_names = (["AOD_Anomaly", "Year", "Month"]
                      + list(preproc.named_transformers_["cat"].get_feature_names_out(["City", "Season"])))
        imp = pd.Series(model.feature_importances_, index=feat_names)

        def grp(f):
            if f == "AOD_Anomaly":
                return "AOD_Anomaly"
            if f.startswith("City_"):
                return "City (climatology proxy)"
            if f.startswith("Season_"):
                return "Season"
            return "Year/Month"

        grouped = imp.groupby(imp.index.map(grp)).sum().sort_values(ascending=False)
        print(f"\n[{best_name}] Importance by group, using AOD_Anomaly instead of raw AOD:")
        for group, val in grouped.items():
            print(f"  {group:<28} {val:.3f}")
        print("\nCompare 'AOD_Anomaly' importance here to 'AOD' importance in the")
        print("earlier with-City run -- a higher value means the anomaly framing")
        print("is capturing a real direct effect, not just collinearity with Season.")


# -----------------------------
# Ablation: same models WITHOUT City, to isolate the real AOD effect
# -----------------------------
def run_ablation_without_city(df):
    print("\n=================================")
    print("ABLATION: models trained WITHOUT City")
    print("(isolates how much predictive power is AOD/Season/time vs.")
    print(" the model just memorizing each city's baseline climate)")
    print("=================================")

    features_no_city = ["AOD", "Year", "Month", "Season"]
    X = df[features_no_city]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["AOD", "Year", "Month"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Season"]),
        ]
    )
    models = build_final_model()  # Random Forest only, per project decision

    results = []
    for name, model in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        result, _ = evaluate_model(name, pipeline, X_train, X_test, y_train, y_test)
        results.append(result)

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
    print("\nWithout-City comparison (sorted by R2):")
    print(results_df.to_string(index=False))
    results_df.to_csv(OUTPUT_DIR / "model_comparison_results_no_city.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'model_comparison_results_no_city.csv'}")
    print("\nCompare this table to the WITH-City results above. A big R2 drop")
    print("confirms City was doing most of the work (climatology), not AOD.")



# -----------------------------
# Save trained models for the Streamlit app to load
# -----------------------------
# Saves each trained pipeline (preprocessing + model bundled together, so
# the Streamlit app doesn't need to reimplement the StandardScaler/
# OneHotEncoder logic) plus a metadata JSON describing the expected input
# schema (feature names, valid City/Season values, Year/Month ranges) so
# the "Predict Temperature" page can build its input form dynamically
# instead of hardcoding city/season lists that could drift out of sync.
def save_models(pipelines, df, best_model_name):
    print("\n=================================")
    print("SAVING MODELS")
    print("=================================")

    filenames = {}
    for name, pipeline in pipelines.items():
        safe_name = name.replace(" ", "_").lower()
        filepath = MODELS_DIR / f"{safe_name}.joblib"
        joblib.dump(pipeline, filepath)
        filenames[name] = filepath.name
        print(f"  Saved: {filepath}")

    metadata = {
        "target": TARGET,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_order": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "cities": sorted(df["City"].unique().tolist()),
        "seasons": ["Winter", "Summer", "Monsoon", "Post-Monsoon"],
        "year_min": int(df["Year"].min()),
        "year_max": int(df["Year"].max()),
        "month_min": 1,
        "month_max": 12,
        "recommended_model": "Random Forest",
        "best_model_by_r2": best_model_name,
        "model_files": filenames,
    }
    metadata_path = MODELS_DIR / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {metadata_path}")
    print("\nIn your Streamlit page, load with:")
    print("  import joblib, json")
    print(f"  model = joblib.load('models/random_forest.joblib')")
    print(f"  meta = json.load(open('models/model_metadata.json'))")
    print("  # meta['cities'], meta['seasons'] give you the dropdown options")


def main():
    df = load_data()

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train set: {len(X_train)} rows | Test set: {len(X_test)} rows\n")

    preprocessor = build_preprocessor()
    models = build_models()

    print("Training and evaluating models...")
    results = []
    predictions = {}
    pipelines = {}

    for name, model in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        result, y_pred = evaluate_model(name, pipeline, X_train, X_test, y_train, y_test)
        results.append(result)
        predictions[name] = y_pred
        pipelines[name] = pipeline

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)

    print("\n=================================")
    print("MODEL COMPARISON (sorted by R2)")
    print("=================================")
    print(results_df.to_string(index=False))

    results_df.to_csv(OUTPUT_DIR / "model_comparison_results.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'model_comparison_results.csv'}")

    # Plots
    plot_model_comparison(results_df)
    print(f"Saved: {OUTPUT_DIR / 'model_comparison.png'}")

    best_model_name = results_df.iloc[0]["Model"]
    print(f"\nBest model by R2: {best_model_name}")
    plot_actual_vs_predicted(y_test, predictions[best_model_name], best_model_name)
    print(f"Saved: actual_vs_predicted_{best_model_name.replace(' ', '_')}.png")

    # Feature importance for tree-based models (skip silently for others)
    for name in ["Decision Tree", "Random Forest", "XGBoost"]:
        if name in pipelines:
            plot_feature_importance(pipelines[name], name)
            print(f"Saved: feature_importance_{name.replace(' ', '_')}.png")

    print(f"\nAll ML outputs saved to: {OUTPUT_DIR.resolve()}")

    save_models(pipelines, df, best_model_name)

    run_ablation_without_city(df)

    df = add_anomaly_columns(df)
    print_anomaly_correlation(df)
    run_anomaly_model(df)

    df_lagged = add_lagged_aod(df)
    print_lagged_correlation(df_lagged)


if __name__ == "__main__":
    main()
