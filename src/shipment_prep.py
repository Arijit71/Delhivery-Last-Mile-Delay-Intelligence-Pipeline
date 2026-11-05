import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]


# Random bookkeeping columns (still required as features).
NOISE_COLUMNS = [
    "label_print_batch", "scanner_build_no", "tape_roll_no",
    "sticker_serial_mod", "shelf_slot_no"
]


# Columns whose missing values will be filled using medians.
MEDIAN_COLUMNS = [
    "distance_km",
    "parcel_weight_kg",
    "declared_value_inr"
]


def clean_shipments(raw, medians=None):
    """Clean raw shipment data and return (cleaned_data, median_values)."""

    # 1. Drop duplicate rows
    df = raw.drop_duplicates().copy()

    # 2. Remove extra spaces and make service names consistent.
    df["service_type"] = df["service_type"].str.strip().str.title()

    # 3. Replace missing payment modes with "Unknown".
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    # 4. Convert declared_value from text to a numeric value.
    value = df["declared_value"]

    value = value.str.replace("₹", "", regex=False)
    value = value.str.replace(",", "", regex=False)

    df["declared_value_inr"] = pd.to_numeric(
        value, errors="coerce"
    ).astype(float)

    # Remove the original text column.
    df = df.drop(columns=["declared_value"])

    # 5. Convert dispatch_time into datetime format.
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])

    # 6. Values above 100 kg are scanner errors.
    # Replace them with NaN.
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    # 7. Calculate medians or reuse previously calculated medians.
    if medians is None:

        median_values = {}

        for column in MEDIAN_COLUMNS:
            median_values[column] = float(df[column].median())

    else:

        # Reuse the historical median values.
        median_values = {}

        for column in MEDIAN_COLUMNS:
            median_values[column] = float(medians[column])

    # 8. Fill missing values using the respective medians.
    for column in MEDIAN_COLUMNS:
        df[column] = df[column].fillna(median_values[column])

    return df, median_values


def build_features(clean, hubs):
    """Merge hub information and create the required features."""

    # Attach hub information using a LEFT JOIN.
    df = clean.merge(
        hubs[["hub_id", "hub_city", "region",
              "hub_tier", "hub_daily_capacity"]],
        on="hub_id",
        how="left"
    )

    # 1. Promised delivery hours based on service type.
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS)

    # 2. COD = 1, Prepaid / Unknown = 0.
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)

    # 3. Saturday / Sunday = 1, other days = 0.
    df["is_weekend"] = (
        df["dispatch_time"].dt.dayofweek >= 5
    ).astype(int)

    # 4. Extract the dispatch hour.
    df["dispatch_hour"] = df["dispatch_time"].dt.hour

    # 5. Apply natural logarithm to declared value.
    df["log_declared_value"] = np.log(df["declared_value_inr"])

    return df