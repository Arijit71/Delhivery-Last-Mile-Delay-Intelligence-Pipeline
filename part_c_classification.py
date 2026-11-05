
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.shipment_prep import FEATURES, NOISE_COLUMNS


# Project folders
PROJECT_FOLDER = Path(__file__).resolve().parent
DATA_FOLDER = PROJECT_FOLDER / "data"
OUTPUT_FOLDER = PROJECT_FOLDER / "outputs"


# Function to calculate classification metrics.
def calculate_metrics(actual, predicted):

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1]
    ).ravel()

    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(
            actual, predicted, zero_division=0
        ),
        "recall": recall_score(
            actual, predicted, zero_division=0
        ),
        "f1": f1_score(
            actual, predicted, zero_division=0
        )
    }


def main():

    # --------------------------------------------------
    # STEP 1: READ DATA AND STRATIFIED SPLITTING
    # --------------------------------------------------

    data = pd.read_csv(DATA_FOLDER / "modelling_table.csv")

    # Input features and target.
    X = data[FEATURES]
    y = data["is_delayed"]

    # First split: 70% training and 30% temporary.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    # Second split: 50% validation and 50% test.
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )

    # --------------------------------------------------
    # STEP 2: BASELINE AND THREE CLASSIFIERS
    # --------------------------------------------------

    model_rows = []

    # 1. Blind baseline: predict every shipment as on time.
    blind_prediction = np.zeros(len(y_val), dtype=int)

    blind_scores = calculate_metrics(
        y_val, blind_prediction
    )

    model_rows.append({
        "model": "blind_all_on_time",
        **blind_scores
    })

    # 2. Logistic Regression with StandardScaler.
    logistic_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression())
    ])

    logistic_model.fit(X_train, y_train)

    logistic_prediction = logistic_model.predict(X_val)

    logistic_scores = calculate_metrics(
        y_val, logistic_prediction
    )

    model_rows.append({
        "model": "logistic_regression",
        **logistic_scores
    })

    # 3. Decision Tree: find the best depth using validation F1.
    depths = [2, 3, 4, 5, 6, 8, 10]

    best_tree = None
    best_depth = None
    best_f1 = -1

    for depth in depths:

        tree = DecisionTreeClassifier(
            criterion="gini",
            max_depth=depth,
            random_state=42
        )

        tree.fit(X_train, y_train)

        prediction = tree.predict(X_val)

        current_f1 = f1_score(
            y_val,
            prediction,
            zero_division=0
        )

        # A tie keeps the smaller depth.
        if current_f1 > best_f1:

            best_f1 = current_f1
            best_tree = tree
            best_depth = depth

    # Evaluate the selected decision tree.
    tree_prediction = best_tree.predict(X_val)

    tree_scores = calculate_metrics(
        y_val, tree_prediction
    )

    model_rows.append({
        "model": f"decision_tree_depth_{best_depth}",
        **tree_scores
    })

    # 4. Default Random Forest.
    default_forest = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    default_forest.fit(X_train, y_train)

    forest_prediction = default_forest.predict(X_val)

    forest_scores = calculate_metrics(
        y_val, forest_prediction
    )

    model_rows.append({
        "model": "random_forest_default",
        **forest_scores
    })

    # Save classifier comparison.
    models_table = pd.DataFrame(
        model_rows,
        columns=[
            "model", "TN", "FP", "FN", "TP",
            "accuracy", "precision", "recall", "f1"
        ]
    )

    metrics = ["accuracy", "precision", "recall", "f1"]

    models_table[metrics] = models_table[metrics].round(4)

    models_table.to_csv(
        OUTPUT_FOLDER / "part_c_models.csv",
        index=False
    )

    # --------------------------------------------------
    # STEP 3: FEATURE SCORES AND FEATURE SELECTION
    # --------------------------------------------------

    # Fit Random Forest on all 17 training features.
    importance_forest = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    importance_forest.fit(X_train, y_train)

    # Calculate F-scores for all features.
    selector_all = SelectKBest(
        score_func=f_classif,
        k="all"
    )

    selector_all.fit(X_train, y_train)

    f_scores = selector_all.scores_

    # Rank features by F-score.
    f_ranks = (
        pd.Series(f_scores)
        .rank(method="min", ascending=False)
        .astype(int)
    )

    # Create feature score table.
    feature_table = pd.DataFrame({

        "feature": FEATURES,

        "f_score": f_scores,

        "f_rank": f_ranks.values,

        "rf_importance": importance_forest.feature_importances_,

        "is_noise": [
            feature in NOISE_COLUMNS
            for feature in FEATURES
        ]

    })

    feature_table[["f_score", "rf_importance"]] = feature_table[
        ["f_score", "rf_importance"]
    ].round(4)

    feature_table.to_csv(
        OUTPUT_FOLDER / "part_c_feature_scores.csv",
        index=False
    )

    # Test different numbers of selected features.
    selectk_rows = []

    for k in [5, 7, 10, 12, 17]:

        model = Pipeline([

            ("selector", SelectKBest(f_classif, k=k)),

            ("model", RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ))

        ])

        model.fit(X_train, y_train)

        prediction = model.predict(X_val)

        val_f1 = f1_score(
            y_val,
            prediction,
            zero_division=0
        )

        selectk_rows.append({
            "k": k,
            "val_f1": round(float(val_f1), 4)
        })

    selectk_table = pd.DataFrame(
        selectk_rows,
        columns=["k", "val_f1"]
    )

    selectk_table.to_csv(
        OUTPUT_FOLDER / "part_c_selectk.csv",
        index=False
    )

    # From now onwards, use exactly 7 features.
    selector = SelectKBest(
        score_func=f_classif,
        k=7
    )

    # Fit selector on training data only.
    X_train_7 = selector.fit_transform(X_train, y_train)

    # Transform validation data using the same selector.
    X_val_7 = selector.transform(X_val)

    # --------------------------------------------------
    # STEP 4: GRID SEARCH AND RANDOMIZED SEARCH
    # --------------------------------------------------

    parameter_grid = {

        "n_estimators": [50, 100, 150],

        "max_depth": [4, 6, 8],

        "min_samples_leaf": [1, 5, 10]

    }

    # Grid Search: 27 combinations and 5-fold CV.
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        parameter_grid,
        cv=5
    )

    grid_search.fit(X_train_7, y_train)

    grid_prediction = grid_search.best_estimator_.predict(X_val_7)

    grid_f1 = f1_score(
        y_val,
        grid_prediction,
        zero_division=0
    )

    # Randomized Search: 9 combinations and 5-fold CV.
    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        parameter_grid,
        n_iter=9,
        cv=5,
        random_state=42
    )

    random_search.fit(X_train_7, y_train)

    random_prediction = random_search.best_estimator_.predict(X_val_7)

    random_f1 = f1_score(
        y_val,
        random_prediction,
        zero_division=0
    )

    # Default Random Forest on the same 7 features.
    default_7_forest = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    default_7_forest.fit(X_train_7, y_train)

    default_prediction = default_7_forest.predict(X_val_7)

    default_f1 = f1_score(
        y_val,
        default_prediction,
        zero_division=0
    )

    # Save tuning results.
    tuning_rows = [

        {
            "search": "grid_search",
            "candidates": 27,
            "cv_folds": 5,
            "total_fits": 135,
            "best_params": str(grid_search.best_params_),
            "best_cv_accuracy": grid_search.best_score_,
            "val_f1": grid_f1
        },

        {
            "search": "randomized_search",
            "candidates": 9,
            "cv_folds": 5,
            "total_fits": 45,
            "best_params": str(random_search.best_params_),
            "best_cv_accuracy": random_search.best_score_,
            "val_f1": random_f1
        },

        {
            "search": "default_forest_same_7_features",
            "candidates": 1,
            "cv_folds": 0,
            "total_fits": 1,
            "best_params": "defaults",
            "best_cv_accuracy": np.nan,
            "val_f1": default_f1
        }

    ]

    tuning_table = pd.DataFrame(
        tuning_rows,
        columns=[
            "search", "candidates", "cv_folds", "total_fits",
            "best_params", "best_cv_accuracy", "val_f1"
        ]
    )

    tuning_table[["best_cv_accuracy", "val_f1"]] = tuning_table[
        ["best_cv_accuracy", "val_f1"]
    ].round(4)

    tuning_table.to_csv(
        OUTPUT_FOLDER / "part_c_tuning.csv",
        index=False
    )

    # --------------------------------------------------
    # STEP 5: CHOOSE THE FINAL CONFIGURATION
    # --------------------------------------------------

    candidate_f1 = {

        "grid_search": grid_f1,

        "randomized_search": random_f1,

        "default_forest_same_7_features": default_f1

    }

    # Tie priority: Default forest, Grid Search, Randomized Search.
    priority = {

        "default_forest_same_7_features": 3,

        "grid_search": 2,

        "randomized_search": 1

    }

    chosen_from = max(
        candidate_f1,
        key=lambda name: (
            candidate_f1[name],
            priority[name]
        )
    )

    if chosen_from == "grid_search":

        chosen_parameters = grid_search.best_params_

    elif chosen_from == "randomized_search":

        chosen_parameters = random_search.best_params_

    else:

        chosen_parameters = {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_leaf": 1
        }

    # Create the final model pipeline.
    final_model = Pipeline([

        ("selector", SelectKBest(f_classif, k=7)),

        ("model", RandomForestClassifier(
            random_state=42,
            **chosen_parameters
        ))

    ])

    final_model.fit(X_train, y_train)

    # --------------------------------------------------
    # STEP 6: ALERT THRESHOLD EXPERIMENT
    # --------------------------------------------------

    # Predict probabilities of delayed shipments.
    val_probabilities = final_model.predict_proba(X_val)[:, 1]

    # Exact decimal thresholds.
    thresholds = [
        0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9
    ]

    threshold_rows = []

    for threshold in thresholds:

        # Predict delayed when probability >= threshold.
        prediction = (
            val_probabilities >= threshold
        ).astype(int)

        scores = calculate_metrics(
            y_val,
            prediction
        )

        threshold_rows.append({
            "threshold": threshold,
            **scores
        })

    # Save threshold comparison.
    threshold_table = pd.DataFrame(
        threshold_rows,
        columns=[
            "threshold", "TN", "FP", "FN", "TP",
            "accuracy", "precision", "recall", "f1"
        ]
    )

    threshold_table[metrics] = threshold_table[metrics].round(4)

    threshold_table.to_csv(
        OUTPUT_FOLDER / "part_c_thresholds.csv",
        index=False
    )

    # Select the highest threshold with recall >= 0.80.
    eligible = [
        row["threshold"]
        for row in threshold_rows
        if row["recall"] >= 0.80
    ]

    if eligible:
        chosen_threshold = max(eligible)
    else:
        chosen_threshold = 0.1

    # --------------------------------------------------
    # STEP 7: FINAL TEST EVALUATION
    # --------------------------------------------------

    # Get test probabilities.
    test_probabilities = final_model.predict_proba(X_test)[:, 1]

    # Apply the chosen threshold.
    test_prediction = (
        test_probabilities >= chosen_threshold
    ).astype(int)

    # Evaluate the test set ONCE.
    test_scores = calculate_metrics(
        y_test,
        test_prediction
    )

    final_test_table = pd.DataFrame([{
        "threshold": chosen_threshold,
        **test_scores
    }])

    final_test_table[metrics] = final_test_table[metrics].round(4)

    final_test_table.to_csv(
        OUTPUT_FOLDER / "part_c_final_test.csv",
        index=False
    )

    # --------------------------------------------------
    # STEP 8: SAVE FINAL CONFIGURATION
    # --------------------------------------------------

    max_depth = chosen_parameters["max_depth"]

    if max_depth is None:
        max_depth = "unlimited"

    final_config = pd.DataFrame([{

        "selector_k": 7,

        "n_estimators": chosen_parameters["n_estimators"],

        "max_depth": max_depth,

        "min_samples_leaf": chosen_parameters["min_samples_leaf"],

        "threshold": chosen_threshold,

        "chosen_from": chosen_from

    }])

    final_config.to_csv(
        OUTPUT_FOLDER / "part_c_final_config.csv",
        index=False
    )

    print("Part C completed successfully!")
    print("Best decision-tree depth:", best_depth)
    print("Chosen configuration:", chosen_from)
    print("Chosen threshold:", chosen_threshold)
    print("Final test F1:", round(test_scores["f1"], 4))


if __name__ == "__main__":
    main()
