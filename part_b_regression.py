
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.shipment_prep import FEATURES


# Project folders
PROJECT_FOLDER = Path(__file__).resolve().parent
DATA_FOLDER = PROJECT_FOLDER / "data"
OUTPUT_FOLDER = PROJECT_FOLDER / "outputs"


# Function to calculate regression metrics.
def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    mse = mean_squared_error(actual, predicted)

    rmse = np.sqrt(mse)

    r2 = r2_score(actual, predicted)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }


def main():

    # Create outputs folder if it does not exist.
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    # --------------------------------------------------
    # STEP 1: READ THE MODELLING TABLE
    # --------------------------------------------------

    data = pd.read_csv(DATA_FOLDER / "modelling_table.csv")

    # Select the 17 input features.
    X = data[FEATURES]

    # Target variable.
    y = data["delivery_hours"]

    # Check for missing values.
    if X.isna().any().any() or y.isna().any():
        raise ValueError("Missing values found in the modelling table.")

    # --------------------------------------------------
    # STEP 2: SPLIT THE DATA
    # --------------------------------------------------

    # First split: 70% training, 30% temporary.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42
    )

    # Second split: 50% validation, 50% test.
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=42
    )

    # Expected split: 840 / 180 / 180.
    if (len(X_train), len(X_val), len(X_test)) != (840, 180, 180):
        raise ValueError("Incorrect train, validation or test sizes.")

    # --------------------------------------------------
    # STEP 3: SCALING WITHOUT DATA LEAKAGE
    # --------------------------------------------------

    scaler = StandardScaler()

    # Fit scaler ONLY on training data.
    X_train_scaled = scaler.fit_transform(X_train)

    # Use the same scaler on test data.
    X_test_scaled = scaler.transform(X_test)

    # Find the largest absolute column mean.
    train_mean = np.abs(X_train_scaled.mean(axis=0)).max()

    test_mean = np.abs(X_test_scaled.mean(axis=0)).max()

    # Save scaling checks.
    checks = pd.DataFrame({
        "metric": [
            "train_rows",
            "val_rows",
            "test_rows",
            "train_scaled_max_abs_mean",
            "test_scaled_max_abs_mean"
        ],

        "value": [
            len(X_train),
            len(X_val),
            len(X_test),
            round(float(train_mean), 4),
            round(float(test_mean), 4)
        ]
    })

    checks.to_csv(
        OUTPUT_FOLDER / "part_b_checks.csv",
        index=False
    )

    # --------------------------------------------------
    # STEP 4: MEAN BASELINE
    # --------------------------------------------------

    # Predict the training mean for every validation row.
    baseline_prediction = np.full(
        len(y_val),
        y_train.mean()
    )

    baseline_scores = calculate_metrics(
        y_val,
        baseline_prediction
    )

    # Store all model results here.
    results = []

    results.append({
        "model": "mean_baseline",
        "split": "val",
        **baseline_scores
    })

    # --------------------------------------------------
    # STEP 5: DECISION TREE DEPTH EXPERIMENT
    # --------------------------------------------------

    depths = [1, 2, 3, 4, 6, 8, 12, None]

    tree_results = []

    fitted_trees = {}

    validation_errors = {}

    for depth in depths:

        tree = DecisionTreeRegressor(
            max_depth=depth,
            random_state=42
        )

        # Train the decision tree.
        tree.fit(X_train, y_train)

        # Predictions on training and validation data.
        train_prediction = tree.predict(X_train)

        val_prediction = tree.predict(X_val)

        # Calculate RMSE for both.
        train_rmse = np.sqrt(
            mean_squared_error(y_train, train_prediction)
        )

        val_rmse = np.sqrt(
            mean_squared_error(y_val, val_prediction)
        )

        # Write unlimited instead of None.
        depth_name = "unlimited" if depth is None else depth

        tree_results.append({
            "max_depth": depth_name,
            "train_rmse": round(float(train_rmse), 4),
            "val_rmse": round(float(val_rmse), 4)
        })

        validation_errors[depth] = val_rmse

        # Store the trained tree.
        fitted_trees[depth] = tree

    # Save the depth experiment.
    tree_table = pd.DataFrame(tree_results)

    tree_table.to_csv(
        OUTPUT_FOLDER / "part_b_tree_depth.csv",
        index=False
    )

    # Select the depth having the lowest validation RMSE.
    # If tied, choose the smaller depth.
    best_depth = min(
        depths,
        key=lambda d: (
            validation_errors[d],
            float("inf") if d is None else d
        )
    )

    best_tree = fitted_trees[best_depth]

    # --------------------------------------------------
    # STEP 6: FOUR CANDIDATE REGRESSION MODELS
    # --------------------------------------------------

    models = {

        "linear_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),

        "decision_tree": best_tree,

        "random_forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ),

        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )

    }

    validation_scores = {}

    for name, model in models.items():

        # The best decision tree has already been trained.
        # Train the other three models here.
        if name != "decision_tree":
            model.fit(X_train, y_train)

        # Predict validation data.
        val_prediction = model.predict(X_val)

        # Calculate evaluation metrics.
        scores = calculate_metrics(
            y_val,
            val_prediction
        )

        validation_scores[name] = scores

        results.append({
            "model": name,
            "split": "val",
            **scores
        })

    # --------------------------------------------------
    # STEP 7: GRADIENT BOOSTING LEARNING-RATE EXPERIMENT
    # --------------------------------------------------

    learning_rates = [0.01, 0.1, 1.0]

    learning_rate_results = []

    for lr in learning_rates:

        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=lr,
            random_state=42
        )

        # Train the model.
        gb_model.fit(X_train, y_train)

        # Predict validation data.
        val_prediction = gb_model.predict(X_val)

        # Calculate metrics.
        scores = calculate_metrics(
            y_val,
            val_prediction
        )

        learning_rate_results.append({
            "learning_rate": lr,
            "val_rmse": round(float(scores["RMSE"]), 4),
            "val_r2": round(float(scores["R2"]), 4)
        })

    # Save learning-rate results.
    learning_rate_table = pd.DataFrame(learning_rate_results)

    learning_rate_table.to_csv(
        OUTPUT_FOLDER / "part_b_gb_learning_rate.csv",
        index=False
    )

    # --------------------------------------------------
    # STEP 8: CROSS-VALIDATION
    # --------------------------------------------------

    # Rank models by validation RMSE.
    ranked_models = sorted(
        validation_scores,
        key=lambda name: validation_scores[name]["RMSE"]
    )

    # Take the two best models.
    best_two = ranked_models[:2]

    # Combine training and validation data.
    X_train_val = pd.concat([X_train, X_val])

    y_train_val = pd.concat([y_train, y_val])

    # Create one KFold object for both models.
    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    cv_results = []

    for name in best_two:

        # cross_val_score creates and fits fresh copies
        # of the model in each fold.
        cv_scores = cross_val_score(
            models[name],
            X_train_val,
            y_train_val,
            cv=kfold
        )

        # Default regression scoring is R-squared.
        cv_results.append({
            "model": name,
            "mean_r2": round(float(cv_scores.mean()), 4),
            "std_r2": round(float(cv_scores.std()), 4)
        })

    # Save cross-validation results.
    cv_table = pd.DataFrame(cv_results)

    cv_table.to_csv(
        OUTPUT_FOLDER / "part_b_cv.csv",
        index=False
    )

    # --------------------------------------------------
    # STEP 9: FINAL TEST EVALUATION
    # --------------------------------------------------

    # Select the model with the lowest validation RMSE.
    final_model_name = ranked_models[0]

    final_model = models[final_model_name]

    # Evaluate the selected model on test data ONCE.
    test_prediction = final_model.predict(X_test)

    test_scores = calculate_metrics(
        y_test,
        test_prediction
    )

    # Add the single final test result.
    results.append({
        "model": final_model_name,
        "split": "test",
        **test_scores
    })

    # Save all model metrics.
    metrics_table = pd.DataFrame(
        results,
        columns=["model", "split", "MAE", "MSE", "RMSE", "R2"]
    )

    # Round all decimal values to 4 places.
    for column in ["MAE", "MSE", "RMSE", "R2"]:
        metrics_table[column] = metrics_table[column].round(4)

    metrics_table.to_csv(
        OUTPUT_FOLDER / "part_b_metrics.csv",
        index=False
    )

    # --------------------------------------------------
    # STEP 10: PRINT FINAL RESULTS
    # --------------------------------------------------

    print("Part B completed successfully!")

    print("Training rows:", len(X_train))

    print("Validation rows:", len(X_val))

    print("Test rows:", len(X_test))

    print(
        "Best decision tree depth:",
        "unlimited" if best_depth is None else best_depth
    )

    print("Best validation model:", final_model_name)

    print(
        "Best validation RMSE:",
        round(validation_scores[final_model_name]["RMSE"], 4)
    )

    print("Final test R2:", round(test_scores["R2"], 4))

    print("Part B output files have been saved.")


if __name__ == "__main__":
    main()
