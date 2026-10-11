"""
Preprocessing pipeline for NASA Giovanni AOD & Temperature CSVs
-----------------------------------------------------------------
- Auto-detects header row (case-insensitive search for "time")
- Auto-detects the correct value column (first numeric, non-date column)
- Handles mixed date formats robustly (DD-MM-YYYY vs YYYY-MM-DD)
- Isolates per-city failures so one bad file doesn't kill the whole run
- Reports merge diagnostics (dropped rows) and a final QA summary
"""

import pandas as pd
import numpy as np
import os
import re
from pathlib import Path

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Cities to skip even if their folder exists (e.g. known incomplete data).
# Add a short reason so it's clear later why something was excluded.
EXCLUDE_CITIES = {
    "Mumbai": "raw Giovanni download only covers Jan 2012-Jan 2020 "
              "(11 months missing, Feb-Dec 2020) -- re-add once re-downloaded",
}


def discover_cities(data_dir: Path) -> list[str]:
    """
    Auto-discover cities from subfolders of data_dir. A folder counts as a
    valid city only if it contains both AOD.csv and Temperature.csv --
    this means adding a new city is just "add a folder", no code change.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    cities = []
    for entry in sorted(data_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in EXCLUDE_CITIES:
            print(f"  [skip] {entry.name}: excluded ({EXCLUDE_CITIES[entry.name]})")
            continue
        has_aod = (entry / "AOD.csv").exists()
        has_temp = (entry / "Temperature.csv").exists()
        if has_aod and has_temp:
            cities.append(entry.name)
        else:
            missing = []
            if not has_aod:
                missing.append("AOD.csv")
            if not has_temp:
                missing.append("Temperature.csv")
            print(f"  [skip] {entry.name}: missing {', '.join(missing)}")

    return cities


# -----------------------------
# Read a single Giovanni CSV
# -----------------------------
def read_giovanni_csv(file_path: Path, value_col_name: str) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    # Case-insensitive search for the header row
    header_row = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("time"):
            header_row = i
            break

    if header_row is None:
        raise ValueError(f"'time' header not found in {file_path}")

    df = pd.read_csv(file_path, skiprows=header_row)
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    # --- Robust date parsing ---
    # IMPORTANT: do not guess the format by comparing NaT counts between
    # dayfirst=True/False. For dates like "2012-02-01" (1st of the month),
    # BOTH interpretations parse successfully with zero NaT (month/day both
    # <=12), so a NaT-count tie-break silently swaps month and day and
    # corrupts the data without raising any error. Instead, detect the
    # format explicitly from the string pattern of the first valid value.
    date_strings = df["Date"].dropna().astype(str)
    sample = date_strings.iloc[0] if len(date_strings) > 0 else ""

    if re.match(r"^\d{4}-\d{1,2}-\d{1,2}", sample):
        # ISO: YYYY-MM-DD (year first -> unambiguous, no dayfirst swap)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=False)
    elif re.match(r"^\d{1,2}-\d{1,2}-\d{4}", sample):
        # DD-MM-YYYY (Giovanni's non-ISO export format)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    elif re.match(r"^\d{1,2}/\d{1,2}/\d{4}", sample):
        # DD/MM/YYYY variant, just in case
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    else:
        # Unknown pattern: let pandas infer, but warn so it gets checked
        print(f"  [warn] {file_path.name}: unrecognized date pattern '{sample}', "
              f"falling back to pandas auto-inference — verify output dates manually")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    n_bad = df["Date"].isna().sum()
    if n_bad > 0:
        print(f"  [warn] {file_path.name}: dropped {n_bad} rows with unparseable dates")
    df = df.dropna(subset=["Date"]).copy()

    # --- Pick the value column: first numeric column that isn't Date ---
    value_col = None
    for col in df.columns:
        if col == "Date":
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() > 0:
            value_col = col
            df[col] = numeric
            break

    if value_col is None:
        raise ValueError(f"No numeric data column found in {file_path}")

    df = df[["Date", value_col]].rename(columns={value_col: value_col_name})

    # --- Handle Giovanni fill/no-data sentinel values ---
    # Giovanni uses large negative numbers (commonly -9999, -999, -999.99...)
    # to mark "no valid retrieval this month" (e.g. cloud cover). These are
    # NOT real measurements and must be treated as missing, not as literal
    # data points -- otherwise a single -9999 will destroy any mean/average
    # calculation and badly distort the ML model.
    FILL_THRESHOLD = -900  # anything more negative than this is a sentinel
    is_fill = df[value_col_name] <= FILL_THRESHOLD
    n_fill = is_fill.sum()
    if n_fill > 0:
        print(f"  [warn] {file_path.name}: {n_fill} fill-value rows "
              f"(e.g. -9999, no valid retrieval) converted to NaN")
        df.loc[is_fill, value_col_name] = np.nan

    return df


def get_season(month: int) -> str:
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8]:
        return "Monsoon"
    else:
        return "Post-Monsoon"


# -----------------------------
# Process one city
# -----------------------------
def process_city(city: str) -> pd.DataFrame | None:
    print(f"Processing {city}...")

    aod_path = DATA_DIR / city / "AOD.csv"
    temp_path = DATA_DIR / city / "Temperature.csv"

    aod = read_giovanni_csv(aod_path, "AOD")
    temp = read_giovanni_csv(temp_path, "Temperature_K")

    temp["Temperature_C"] = temp["Temperature_K"] - 273.15

    merged = pd.merge(aod, temp, on="Date", how="inner")

    # Diagnostics: how much data was lost in the merge
    n_lost_aod = len(aod) - len(merged)
    n_lost_temp = len(temp) - len(merged)
    if n_lost_aod or n_lost_temp:
        print(f"  [info] {city}: merge dropped {n_lost_aod} AOD rows, "
              f"{n_lost_temp} Temp rows (dates didn't align)")

    if merged.empty:
        print(f"  [error] {city}: no overlapping dates between AOD and Temperature — skipped")
        return None

    merged["City"] = city
    merged["Year"] = merged["Date"].dt.year
    merged["Month"] = merged["Date"].dt.month
    merged["Season"] = merged["Month"].apply(get_season)

    merged = merged.sort_values("Date").reset_index(drop=True)
    merged.to_csv(OUTPUT_DIR / f"{city}_Cleaned.csv", index=False)

    print(f"  {city} completed ({len(merged)} rows, "
          f"{merged['Date'].min().date()} to {merged['Date'].max().date()})")
    return merged


# -----------------------------
# Main
# -----------------------------
def main():
    cities = discover_cities(DATA_DIR)
    print(f"\nDiscovered {len(cities)} cities to process: {cities}\n")

    all_data = []
    failed_cities = []

    for city in cities:
        try:
            df = process_city(city)
            if df is not None:
                all_data.append(df)
            else:
                failed_cities.append(city)
        except Exception as e:
            print(f"  [FAILED] {city}: {e}")
            failed_cities.append(city)

    if not all_data:
        print("\nNo cities processed successfully. Check your data folder paths.")
        return

    combined = pd.concat(all_data, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "Combined_Dataset.csv", index=False)

    print("\n=================================")
    print("PROJECT PREPROCESSING COMPLETE")
    print("=================================")
    print(f"\nTotal records : {len(combined)}")
    print(f"Cities processed : {sorted(combined['City'].unique().tolist())}")
    if failed_cities:
        print(f"Cities FAILED   : {failed_cities}")
    print(f"Year range : {combined['Year'].min()} - {combined['Year'].max()}")

    # --- QA summary: expected 108 months (2012-2020) per city ---
    print("\nRecords per city (expect 108 for full 2012-2020 monthly coverage):")
    print(combined.groupby("City").size())

    # --- QA: fill-value gaps that survived as NaN inside the date range ---
    nan_counts = combined.groupby("City")[["AOD", "Temperature_C"]].apply(
        lambda g: g.isna().sum()
    )
    if nan_counts.to_numpy().sum() > 0:
        print("\n[QA] NaN values within present rows (from -9999 fill values):")
        print(nan_counts[(nan_counts["AOD"] > 0) | (nan_counts["Temperature_C"] > 0)])
        print("These months exist in the timeline but have no valid reading.")
        print("Decide before EDA/ML: drop these rows, or interpolate them.")

        # Break the NaNs down by season, to check if they cluster around
        # monsoon (expected -- MODIS can't see through heavy cloud cover)
        # or are scattered randomly (a different kind of data issue).
        nan_rows = combined[combined["AOD"].isna()]
        season_breakdown = nan_rows.groupby(["City", "Season"]).size().unstack(fill_value=0)
        print("\n[QA] Missing AOD by City x Season (checking for monsoon clustering):")
        print(season_breakdown)
    else:
        print("\n[QA] No fill-value NaNs remaining in the dataset.")

    missing_report = []
    for city in combined["City"].unique():
        city_months = set(
            combined.loc[combined["City"] == city, "Date"].dt.to_period("M")
        )
        expected_months = set(
            pd.period_range("2012-01", "2020-12", freq="M")
        )
        missing = sorted(expected_months - city_months)
        if missing:
            missing_report.append((city, len(missing), missing))

    if missing_report:
        print("\n[QA] Missing months detected:")
        for city, n, months in missing_report:
            print(f"  {city}: {n} missing -> {months[:5]}{'...' if n > 5 else ''}")
    else:
        print("\n[QA] No missing months — full monthly coverage for all cities.")

    print("\nFirst 5 rows of combined dataset:")
    print(combined.head())

    # -----------------------------------------------------------------
    # Seasonal-mean imputation for NaN AOD values
    # -----------------------------------------------------------------
    # Rationale: a missing AOD reading (e.g. Bengaluru, Monsoon 2015) is
    # filled using the mean AOD for that SAME city and SAME season across
    # the OTHER years (2012-2020) -- not by interpolating from adjacent
    # calendar months. Simple time-interpolation would blend a monsoon gap
    # with pre/post-monsoon values and flatten out exactly the seasonal
    # signal this project is trying to measure. Seasonal-mean imputation
    # preserves that signal instead.
    #
    # EDGE CASE: if a city has NO valid AOD readings at all for a given
    # season (e.g. every single Monsoon month failed retrieval -- this
    # happens for coastal/high-cloud-cover cities), there is nothing to
    # average, so the seasonal mean itself is NaN and the fill does
    # nothing. This is detected explicitly below rather than silently
    # left broken. Affected rows fall back to that city's overall mean
    # AOD (all seasons) and are flagged in AOD_Imputation_Method so you
    # can identify/exclude them from season-specific claims later.
    #
    # The original combined dataset (with NaNs) is kept as-is for
    # transparency. A second file, with imputation applied, is saved
    # separately for use in EDA/ML.
    imputed = combined.copy()
    imputed["AOD_Imputation_Method"] = "none"
    originally_missing = imputed["AOD"].isna()

    # Pass 1: City + Season mean
    season_means = imputed.groupby(["City", "Season"])["AOD"].transform("mean")
    filled_by_season = originally_missing & season_means.notna()
    imputed.loc[filled_by_season, "AOD"] = season_means[filled_by_season]
    imputed.loc[filled_by_season, "AOD_Imputation_Method"] = "season_mean"

    # Pass 2: fallback for city-season groups with ZERO valid data
    still_missing = imputed["AOD"].isna()
    if still_missing.any():
        affected_groups = (
            imputed.loc[still_missing, ["City", "Season"]]
            .drop_duplicates()
            .to_records(index=False)
            .tolist()
        )
        print(f"\n[WARN] {still_missing.sum()} rows have ZERO valid AOD "
              f"readings for their City+Season -- seasonal mean is undefined:")
        for city, season in affected_groups:
            print(f"  {city} / {season}: no valid AOD in any year for this season")
        print("Falling back to each city's OVERALL mean AOD (all seasons) for these rows.")
        print("These rows are flagged 'city_mean_fallback' -- treat season-specific")
        print("conclusions for these city/season combinations with caution, or")
        print("consider excluding them from seasonal analysis entirely.")

        city_means = imputed.groupby("City")["AOD"].transform("mean")
        fallback_filled = still_missing & city_means.notna()
        imputed.loc[fallback_filled, "AOD"] = city_means[fallback_filled]
        imputed.loc[fallback_filled, "AOD_Imputation_Method"] = "city_mean_fallback"

    # Final verification: confirm nothing is left unfilled
    remaining_nan = imputed["AOD"].isna().sum()
    n_imputed = originally_missing.sum()

    if remaining_nan > 0:
        print(f"\n[ERROR] {remaining_nan} AOD values still NaN after all "
              f"imputation attempts -- these cities/rows have no usable AOD "
              f"data at all and must be handled manually.")

    if n_imputed > 0:
        imputed.to_csv(OUTPUT_DIR / "Combined_Dataset_Imputed.csv", index=False)
        method_counts = imputed.loc[originally_missing, "AOD_Imputation_Method"].value_counts()
        print(f"\n[INFO] {n_imputed} missing AOD values processed:")
        for method, count in method_counts.items():
            print(f"  {count} via {method}")
        print("Saved: output/Combined_Dataset_Imputed.csv (use this for EDA/ML)")
        print("  -> includes 'AOD_Imputation_Method' column: 'none' / "
              "'season_mean' / 'city_mean_fallback'")
        print("Original with NaNs preserved in: output/Combined_Dataset.csv")
    else:
        print("\n[INFO] No imputation needed -- dataset is already complete.")


if __name__ == "__main__":
    main()
