
from pathlib import Path

import numpy as np
import pandas as pd

from src.shipment_prep import FEATURES, clean_shipments, build_features


# Get the project folder and data/output folders.
PROJECT_FOLDER = Path(__file__).resolve().parent

DATA_FOLDER = PROJECT_FOLDER / "data"
OUTPUT_FOLDER = PROJECT_FOLDER / "outputs"


def round_value(value):
    """Keep integers as integers and round decimal values to 4 places."""

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)):
        return round(float(value), 4)

    return value


def create_checks(raw, dedup_raw, clean, featured, modelling, medians):
    """Calculate all data-quality and delivery-time metrics."""

    # Calculate quartiles and IQR for delivery_hours.
    hours = modelling["delivery_hours"]

    q1 = hours.quantile(0.25)
    q3 = hours.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Store all metrics as name-value pairs.
    metrics = [

        ("raw_rows", len(raw)),

        ("duplicate_rows_removed",
         len(raw) - len(dedup_raw)),

        ("rows_after_dedup",
         len(dedup_raw)),

        ("missing_distance_km_before",
         dedup_raw["distance_km"].isna().sum()),

        ("missing_parcel_weight_before",
         dedup_raw["parcel_weight_kg"].isna().sum()),

        ("sentinel_weight_rows",
         (dedup_raw["parcel_weight_kg"] > 100).sum()),

        ("missing_declared_value_before",
         dedup_raw["declared_value"].isna().sum()),

        ("missing_payment_mode_before",
         dedup_raw["payment_mode"].isna().sum()),

        ("service_type_spellings_before",
         dedup_raw["service_type"].nunique()),

        ("service_type_spellings_after",
         clean["service_type"].nunique()),

        ("median_distance_km",
         medians["distance_km"]),

        ("median_parcel_weight_kg",
         medians["parcel_weight_kg"]),

        ("median_declared_value_inr",
         medians["declared_value_inr"]),

        ("rows_after_hub_merge",
         len(featured)),

        ("nulls_in_features_and_target",
         modelling[
             FEATURES + ["delivery_hours", "is_delayed"]
         ].isna().sum().sum()),

        ("delayed_count",
         modelling["is_delayed"].sum()),

        ("delay_rate",
         modelling["is_delayed"].mean()),

        ("hours_q1", q1),

        ("hours_q3", q3),

        ("hours_lower_bound", lower_bound),

        ("hours_upper_bound", upper_bound),

        ("hours_outlier_count",
         ((hours < lower_bound) | (hours > upper_bound)).sum()),

        ("declared_value_mean_inr",
         modelling["declared_value_inr"].mean()),

        ("declared_value_median_inr",
         modelling["declared_value_inr"].median()),

        ("log_declared_value_mean",
         modelling["log_declared_value"].mean()),

        ("log_declared_value_median",
         modelling["log_declared_value"].median())

    ]

    # Create the checks DataFrame.
    # Object dtype keeps integer values as integers.
    checks = pd.DataFrame({
        "metric": [name for name, value in metrics],

        "value": pd.Series(
            [round_value(value) for name, value in metrics],
            dtype="object"
        )
    })

    return checks


def create_delay_table(modelling):
    """Calculate delay rates for service type, hub tier and distance."""

    df = modelling.copy()

    # Create the four distance bands.
    df["distance_band"] = pd.cut(
        df["distance_km"],

        bins=[0, 200, 400, 600, 2000],

        labels=["0-200", "201-400", "401-600", "601+"],

        right=True,

        include_lowest=True
    )

    results = []

    # Group by service type, hub tier and distance band.
    for column in ["service_type", "hub_tier", "distance_band"]:

        groups = df.groupby(
            column,
            sort=True,
            observed=True
        )

        for group_name, group_data in groups:

            results.append({

                "dimension": column,

                "group": group_name,

                "shipments": len(group_data),

                "delayed": group_data["is_delayed"].sum(),

                "delay_rate": round(
                    group_data["is_delayed"].mean(), 4
                ),

                "mean_hours": round(
                    group_data["delivery_hours"].mean(), 4
                )

            })

    return pd.DataFrame(results, columns=[
        "dimension",
        "group",
        "shipments",
        "delayed",
        "delay_rate",
        "mean_hours"
    ])


def main():

    # Make sure the output folder exists.
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    # Read the input CSV files.
    raw = pd.read_csv(DATA_FOLDER / "raw_shipments.csv")

    hubs = pd.read_csv(DATA_FOLDER / "hubs.csv")

    # Remove duplicates separately for calculating the initial metrics.
    dedup_raw = raw.drop_duplicates().copy()

    # Clean the shipment data and calculate medians.
    clean, medians = clean_shipments(raw)

    # Add hub information and generate features.
    featured = build_features(clean, hubs)

    # Check that the hub merge did not change the row count.
    if len(featured) != len(clean):
        raise ValueError(
            "The hub merge changed the number of shipment rows."
        )

    # Create the target variable.
    # 1 = Delayed, 0 = Not delayed.
    featured["is_delayed"] = (
        featured["delivery_hours"] > featured["promised_hours"]
    ).astype(int)

    modelling = featured

    # Check that all required features and target columns exist.
    required_columns = FEATURES + [
        "delivery_hours",
        "is_delayed"
    ]

    # Count missing values in these columns.
    null_count = modelling[required_columns].isna().sum().sum()

    if null_count > 0:
        raise ValueError(
            f"There are {null_count} missing cells in required columns."
        )

    # Save the complete modelling table.
    modelling.to_csv(
        DATA_FOLDER / "modelling_table.csv",
        index=False
    )

    # Generate and save the Part A checks.
    checks = create_checks(
        raw,
        dedup_raw,
        clean,
        featured,
        modelling,
        medians
    )

    checks.to_csv(
        OUTPUT_FOLDER / "part_a_checks.csv",
        index=False
    )

    # Generate and save delay evidence.
    delay_table = create_delay_table(modelling)

    delay_table.to_csv(
        OUTPUT_FOLDER / "part_a_delay_by_group.csv",
        index=False
    )

    # Select all numeric columns.
    numeric_data = modelling.select_dtypes(
        include=[np.number]
    )

    # Calculate the correlation matrix.
    correlations = numeric_data.corr().round(4)

    # Keep row labels as the first column.
    correlations.index.name = "column"

    correlations.to_csv(
        OUTPUT_FOLDER / "part_a_correlations.csv",
        index=True
    )

    # --------------------------------------------------
    # Test the functions on new shipments.
    # Use the historical medians calculated earlier.
    # --------------------------------------------------

    new_raw = pd.read_csv(
        DATA_FOLDER / "new_shipments.csv"
    )

    clean_new, _ = clean_shipments(
        new_raw,
        medians
    )

    new_features = build_features(
        clean_new,
        hubs
    )

    # Check whether any of the 17 features are missing.
    new_null_count = new_features[FEATURES].isna().sum().sum()

    if new_null_count > 0:
        raise ValueError(
            f"New shipments have {new_null_count} missing feature values."
        )

    # Print the final summary.
    print("Part A completed successfully!")

    print("Raw rows:", len(raw))

    print("Rows after cleaning:", len(modelling))

    print("Median values:", medians)

    print("Delayed shipments:", modelling["is_delayed"].sum())

    print(
        "Overall delay rate:",
        round(modelling["is_delayed"].mean(), 4)
    )

    print("Missing features in new shipments:", new_null_count)

    print("Output files have been saved.")


if __name__ == "__main__":
    main()
