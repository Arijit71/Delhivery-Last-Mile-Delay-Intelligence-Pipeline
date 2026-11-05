
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.shipment_prep import FEATURES, clean_shipments


PROJECT_FOLDER = Path(__file__).resolve().parent
DATA_FOLDER = PROJECT_FOLDER / "data"
OUTPUT_FOLDER = PROJECT_FOLDER / "outputs"
MODEL_FOLDER = PROJECT_FOLDER / "models"


def main():

    MODEL_FOLDER.mkdir(exist_ok=True)

    OUTPUT_FOLDER.mkdir(exist_ok=True)

    # --------------------------------------------------
    # STEP 1: READ FINAL CONFIGURATION FROM PART C
    # --------------------------------------------------

    config = pd.read_csv(
        OUTPUT_FOLDER / "part_c_final_config.csv"
    ).iloc[0]

    selector_k = int(config["selector_k"])

    n_estimators = int(config["n_estimators"])

    max_depth = config["max_depth"]

    if str(max_depth).lower() == "unlimited":
        max_depth = None
    else:
        max_depth = int(max_depth)

    min_samples_leaf = int(config["min_samples_leaf"])

    threshold = float(config["threshold"])

    # --------------------------------------------------
    # STEP 2: RECREATE PART C DATA SPLIT
    # --------------------------------------------------

    data = pd.read_csv(
        DATA_FOLDER / "modelling_table.csv"
    )

    X = data[FEATURES]

    y = data["is_delayed"]

    # First split: training and temporary data.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    # Second split: validation and test.
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )

    # --------------------------------------------------
    # STEP 3: REBUILD AND TRAIN FINAL PIPELINE
    # --------------------------------------------------

    model = Pipeline([

        ("selector", SelectKBest(
            score_func=f_classif,
            k=selector_k
        )),

        ("model", RandomForestClassifier(

            n_estimators=n_estimators,

            max_depth=max_depth,

            min_samples_leaf=min_samples_leaf,

            random_state=42

        ))

    ])

    # Fit using training rows only.
    model.fit(X_train, y_train)

    # --------------------------------------------------
    # STEP 4: SAVE MODEL AND SUPPORTING FILES
    # --------------------------------------------------

    # Save the trained model.
    joblib.dump(
        model,
        MODEL_FOLDER / "delay_model.pkl"
    )

    # Save alert threshold.
    pd.DataFrame({
        "threshold": [threshold]
    }).to_csv(
        MODEL_FOLDER / "delay_threshold.csv",
        index=False
    )

    # Save feature names in the exact order.
    pd.DataFrame({
        "feature": FEATURES
    }).to_csv(
        MODEL_FOLDER / "feature_order.csv",
        index=False
    )

    # Calculate historical medians using Part A's function.
    raw = pd.read_csv(
        DATA_FOLDER / "raw_shipments.csv"
    )

    _, medians = clean_shipments(raw)

    # Save the three imputation values.
    impute_values = pd.DataFrame([

        {
            "column": column,
            "median": median
        }

        for column, median in medians.items()

    ])

    impute_values.to_csv(
        MODEL_FOLDER / "impute_values.csv",
        index=False,
        float_format="%.4f"
    )

    # --------------------------------------------------
    # STEP 5: RELOAD AND VERIFY THE MODEL
    # --------------------------------------------------

    reloaded_model = joblib.load(
        MODEL_FOLDER / "delay_model.pkl"
    )

    # Probabilities from the original model.
    original_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # Probabilities from the reloaded model.
    reloaded_probabilities = reloaded_model.predict_proba(
        X_test
    )[:, 1]

    # Find maximum absolute difference.
    max_difference = np.max(
        np.abs(
            original_probabilities -
            reloaded_probabilities
        )
    )

    # Save reload verification.
    reload_check = pd.DataFrame({

        "metric": [
            "test_rows_scored",
            "reload_max_abs_probability_diff",
            "threshold"
        ],

        "value": pd.Series(
            [
                len(X_test),
                float(max_difference),
                threshold
            ],
            dtype="object"
        )

    })

    reload_check.to_csv(
        OUTPUT_FOLDER / "part_d_reload_check.csv",
        index=False
    )

    print("Model packaged successfully!")

    print("Test rows scored:", len(X_test))

    print("Reload probability difference:", max_difference)

    print("Alert threshold:", threshold)


if __name__ == "__main__":
    main()
