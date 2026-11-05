
from pathlib import Path

import joblib
import pandas as pd

from src.shipment_prep import clean_shipments, build_features


PROJECT_FOLDER = Path(__file__).resolve().parent
DATA_FOLDER = PROJECT_FOLDER / "data"
OUTPUT_FOLDER = PROJECT_FOLDER / "outputs"
MODEL_FOLDER = PROJECT_FOLDER / "models"


def main():

    # --------------------------------------------------
    # STEP 1: LOAD DATA AND SAVED MODEL FILES
    # --------------------------------------------------

    new_raw = pd.read_csv(
        DATA_FOLDER / "new_shipments.csv"
    )

    hubs = pd.read_csv(
        DATA_FOLDER / "hubs.csv"
    )

    model = joblib.load(
        MODEL_FOLDER / "delay_model.pkl"
    )

    threshold = float(
        pd.read_csv(
            MODEL_FOLDER / "delay_threshold.csv"
        )["threshold"].iloc[0]
    )

    feature_order = pd.read_csv(
        MODEL_FOLDER / "feature_order.csv"
    )["feature"].tolist()

    impute_values = pd.read_csv(
        MODEL_FOLDER / "impute_values.csv"
    )

    medians = dict(
        zip(
            impute_values["column"],
            impute_values["median"]
        )
    )

    hub_segments = pd.read_csv(
        OUTPUT_FOLDER / "hub_segments.csv"
    )

    # --------------------------------------------------
    # STEP 2: CLEAN NEW SHIPMENTS
    # --------------------------------------------------

    # Use the historical medians saved by package_model.py.
    clean, _ = clean_shipments(
        new_raw,
        medians
    )

    # Create the same features used during training.
    featured = build_features(
        clean,
        hubs
    )

    # --------------------------------------------------
    # STEP 3: ATTACH HUB SEGMENTS
    # --------------------------------------------------

    # Keep track of the original shipment order.
    featured["_row_order"] = range(len(featured))

    featured = featured.merge(
        hub_segments[["hub_id", "segment"]],
        on="hub_id",
        how="left"
    )

    featured = featured.sort_values(
        "_row_order"
    )

    # --------------------------------------------------
    # STEP 4: PREDICT DELAY PROBABILITIES
    # --------------------------------------------------

    # Arrange features in the saved training order.
    X_new = featured[feature_order]

    # Get the probability of class 1 (delayed).
    probabilities = model.predict_proba(
        X_new
    )[:, 1]

    # Round probabilities to four decimal places.
    probabilities = pd.Series(
        probabilities
    ).round(4)

    # Flag shipments whose rounded probability
    # is greater than or equal to the threshold.
    flagged = (
        probabilities >= threshold
    ).astype(int)

    # --------------------------------------------------
    # STEP 5: ASSIGN OPERATIONAL ACTIONS
    # --------------------------------------------------

    actions = []

    for is_flagged, segment in zip(
        flagged,
        featured["segment"]
    ):

        if is_flagged == 0:

            actions.append("No action")

        elif segment == "Escalate":

            actions.append("Escalate to hub manager")

        else:

            actions.append("Proactive customer call")

    # --------------------------------------------------
    # STEP 6: CREATE FINAL ACTION LIST
    # --------------------------------------------------

    result = pd.DataFrame({

        "shipment_id": featured["shipment_id"].values,

        "hub_id": featured["hub_id"].values,

        "segment": featured["segment"].values,

        "delay_probability": probabilities.values,

        "flagged": flagged.values,

        "action": actions

    })

    # Save the results in the original shipment order.
    result.to_csv(
        OUTPUT_FOLDER / "new_shipment_actions.csv",
        index=False,
        float_format="%.4f"
    )

    print("New shipment scoring completed successfully!")

    print("Shipments scored:", len(result))

    print("Flagged shipments:", int(result["flagged"].sum()))

    print(
        "Hub-manager escalations:",
        int((result["action"] == "Escalate to hub manager").sum())
    )


if __name__ == "__main__":
    main()
