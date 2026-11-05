# Delhivery — Last-Mile Delay Intelligence

A machine learning capstone project for predicting delivery time, identifying delayed shipments, and analysing logistics hub performance using synthetic shipment data.

**Program:** AI and ML Certification Program — E&ICT Academy, IIT Roorkee

## Project Overview

The project addresses three operational problems:

- **Delivery Time Prediction:** Estimate the delivery time of each shipment.
- **Delay Classification:** Identify shipments likely to miss their promised delivery time and generate proactive alerts.
- **Hub Segmentation:** Analyse hub performance to identify operational patterns and improvement opportunities.

The project uses synthetic data generated specifically for this assignment. No real Delhivery shipment records are used.

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Joblib

## Setup and Execution

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the scripts from the repository root in this order:

```bash
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
```

Each script generates its respective CSV outputs in the `data/` or `outputs/` directories.

---

# Part A — Data Foundation and Delay Evidence

## Data Cleaning Summary

| Metric | Result |
|---|---:|
| Initial shipment rows | 1,224 |
| Duplicate rows removed | 24 |
| Final unique shipments | 1,200 |
| Missing distance values | 36 |
| Missing parcel weights | 24 |
| Scanner-fault weight values | 5 |
| Missing declared values | 12 |
| Missing payment modes | 18 |
| Service spellings before cleaning | 12 |
| Service spellings after cleaning | 3 |

Missing values in distance, parcel weight and declared value were filled using their respective medians. Scanner-fault weights above 100 kg were treated as missing. Service names were standardised, and hub information was added using a left merge.

**Median values used for imputation:**

| Feature | Median |
|---|---:|
| distance_km | 314.95 |
| parcel_weight_kg | 1.83 |
| declared_value_inr | INR 1,121.50 |

## Delay Analysis

**Delay rate by service type**

| Service | Shipments | Delayed | Delay rate |
|---|---:|---:|---:|
| Economy | 315 | 27 | 8.57% |
| Express | 243 | 119 | 48.97% |
| Standard | 642 | 132 | 20.56% |

Express shipments have the highest delay rate.

**Delay rate by hub tier**

| Hub Tier | Shipments | Delayed | Delay rate |
|---|---:|---:|---:|
| 1 | 400 | 16 | 4.00% |
| 2 | 400 | 86 | 21.50% |
| 3 | 400 | 176 | 44.00% |

Delay rates increase substantially from Tier 1 to Tier 3.

**Delay rate by distance**

| Distance Band | Shipments | Delayed | Delay rate |
|---|---:|---:|---:|
| 0–200 km | 274 | 6 | 2.19% |
| 201–400 km | 563 | 77 | 13.68% |
| 401–600 km | 275 | 120 | 43.64% |
| 601+ km | 88 | 75 | 85.23% |

Longer shipment distances are associated with considerably higher delay rates.

## Correlation Analysis

The three features most strongly correlated with `delivery_hours`, ranked by absolute correlation, are:

| Rank | Feature | Correlation |
|---:|---|---:|
| 1 | promised_hours | +0.8372 |
| 2 | distance_km | +0.4013 |
| 3 | n_stops | +0.3247 |

## Declared Value Transformation

| Statistic | Original Value (INR) | Log-Transformed Value |
|---|---:|---:|
| Mean | 1,533.05 | 7.0277 |
| Median | 1,121.50 | 7.0224 |

The log transformation compresses large values and reduces the influence of the right-skewed declared-value distribution, bringing the mean and median closer together.

---

# Part B — Delivery-Hours Regression

## Data Splitting and Scaling

The 1,200 shipments were divided into training, validation and test sets using `random_state=42`.

| Dataset | Rows | Percentage |
|---|---:|---:|
| Training | 840 | 70% |
| Validation | 180 | 15% |
| Test | 180 | 15% |

`StandardScaler` was fitted only on training data and then applied to the test data.

| Scaling Check | Result |
|---|---:|
| Training maximum absolute column mean | 0.0000 |
| Test maximum absolute column mean | 0.2187 |

This confirms that the scaler was not fitted on the test data.

## Baseline and Model Comparison

The mean baseline achieved an R² of **-0.0212** on the validation set. Its negative R² indicates that predicting the training mean performs slightly worse than using the validation-set mean as the predictor.

**Validation performance:**

| Model | RMSE (Hours) | R² |
|---|---:|---:|
| Mean Baseline | 18.8552 | -0.0212 |
| Linear Regression | **4.8912** | **0.9313** |
| Decision Tree | 6.2796 | 0.8867 |
| Random Forest | 5.5095 | 0.9128 |
| Gradient Boosting | 4.9165 | 0.9306 |

**Best model: Linear Regression**, with a validation RMSE of 4.8912 hours.

## Decision Tree Depth Analysis

| Maximum Depth | Training RMSE | Validation RMSE |
|---:|---:|---:|
| 1 | 11.6334 | 12.1324 |
| 2 | 9.3687 | 10.4114 |
| 3 | 7.5295 | 8.7033 |
| 4 | 6.0999 | 7.3896 |
| 6 | 4.7114 | **6.2796** |
| 8 | 3.3360 | 6.7901 |
| 12 | 1.3613 | 7.3540 |
| Unlimited | 0.0000 | 7.8368 |

- **High bias:** Depth 1 has high training and validation errors, with a small gap of 0.4990.
- **High variance:** Depth 12 and unlimited depth have gaps of 5.9927 and 7.8368 respectively, indicating overfitting.
- **Best depth:** 6, with the lowest validation RMSE.

## Gradient Boosting Learning Rate

| Learning Rate | Validation RMSE | Validation R² |
|---:|---:|---:|
| 0.01 | 10.0018 | 0.7127 |
| 0.1 | **4.9165** | **0.9306** |
| 1.0 | 7.3701 | 0.8440 |

A learning rate of 0.1 performs best. A rate of 0.01 learns too slowly with 100 estimators, whereas 1.0 is too aggressive and overshoots.

## Cross-Validation

The two best validation models were evaluated using five-fold cross-validation on the combined training and validation data.

| Model | Mean R² | Standard Deviation |
|---|---:|---:|
| Linear Regression | 0.9261 | 0.0048 |
| Gradient Boosting | **0.9353** | 0.0045 |

Cross-validation favours Gradient Boosting, but Linear Regression is selected because it achieved the lowest validation RMSE, as required by the assignment.

## Final Test Evaluation

| Selected Model | Test RMSE | Test R² |
|---|---:|---:|
| Linear Regression | 4.5235 | 0.9493 |

Linear Regression performs well on the previously unseen test data.

---

# Part C — Delay Classification and Alert Threshold

## Stratified Data Splitting

| Dataset | Total Rows | Delayed Shipments |
|---|---:|---:|
| Training | 840 | 195 |
| Validation | 180 | 42 |
| Test | 180 | 41 |

Stratified splitting maintains the proportion of delayed and on-time shipments across all three datasets.

## Baseline and Classifier Comparison

The blind baseline predicts every shipment as on time. Although it achieves 76.67% accuracy, its precision and recall are both zero because it misses all 42 delayed validation shipments.

**Validation performance using the default 0.5 decision threshold:**

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Blind Baseline | 0.7667 | 0.0000 | 0.0000 | 0.0000 |
| Logistic Regression | 0.9056 | 0.7778 | 0.8333 | **0.8046** |
| Decision Tree (Depth 6) | 0.8833 | 0.7561 | 0.7381 | 0.7470 |
| Random Forest | 0.9056 | **0.8378** | 0.7381 | 0.7848 |

Logistic Regression achieves the highest F1 among the initial classifier comparisons.

## Feature Selection and Noise Analysis

The five bookkeeping columns are treated as noise features.

Their combined Random Forest importance is **0.2245 (22.45%)**. All five have F-scores below 10 and rank outside the top seven features.

**Top seven features by ANOVA F-score:**

| Rank | Feature | F-score |
|---:|---|---:|
| 1 | distance_km | 389.2131 |
| 2 | hub_tier | 155.3483 |
| 3 | hub_daily_capacity | 144.1347 |
| 4 | n_stops | 125.4798 |
| 5 | promised_hours | 84.4177 |
| 6 | rain_flag | 53.0321 |
| 7 | sale_event_flag | 36.7281 |

**SelectKBest experiment:**

| Selected Features (k) | Validation F1 |
|---:|---:|
| 5 | 0.6988 |
| 7 | **0.8333** |
| 10 | 0.7857 |
| 12 | 0.8049 |
| 17 | 0.7848 |

Selecting seven features produces the highest validation F1 and is used for subsequent experiments.

## Hyperparameter Tuning

| Search Method | Candidates | Total Fits | Best CV Accuracy | Validation F1 |
|---|---:|---:|---:|---:|
| Grid Search | 27 | 135 | 0.8952 | 0.7750 |
| Randomized Search | 9 | 45 | 0.8905 | 0.7848 |
| Default Random Forest | 1 | 1 | — | **0.8333** |

The untuned Random Forest performs best on validation, so it is selected over the tuned alternatives.

This shows that hyperparameter tuning does not necessarily improve validation performance. The small validation set of 180 shipments can also make model rankings sensitive to individual predictions. Note that the searches optimise accuracy by default, while the model selection here uses validation F1.

## Alert Threshold Selection

The operational requirement is to select the highest probability threshold that maintains validation recall of at least 0.80.

| Threshold | Validation Precision | Validation Recall | Validation F1 |
|---:|---:|---:|---:|
| 0.5 | 0.7955 | 0.8333 | 0.8140 |
| 0.6 | 0.8889 | 0.7619 | 0.8205 |

**Selected threshold: 0.5.**

Although 0.6 provides higher precision, its recall falls below the required 0.80. Therefore, 0.5 is the highest acceptable threshold.

## Final Test Evaluation

The selected Random Forest model was evaluated once on the untouched test set using the chosen threshold of 0.5.

| Metric | Test Result |
|---|---:|
| Accuracy | 0.8667 |
| Precision | 0.6889 |
| Recall | 0.7561 |
| F1 | 0.7209 |
| True Positives | 31 |
| False Negatives | 10 |

Test recall (0.7561) is lower than validation recall (0.8333), a decrease of 7.72 percentage points. The test set contains only 41 delayed shipments, so its recall estimate is based on a relatively small number of positive cases.

---

## Project Progress

| Part | Description | Status |
|---|---|---|
| Part A | Data Foundation and Delay Evidence | Completed |
| Part B | Delivery-Hours Regression | Completed |
| Part C | Delay Classification and Alert Threshold | Completed |
| Part D | Hub Segmentation | Pending |
| Final Pipeline | Model Packaging and Shipment Scoring | Pending |

## Repository Structure

```text
├── README.md
├── requirements.txt
├── generate_data.py
├── data/
├── src/
│   └── shipment_prep.py
├── models/
├── outputs/
├── part_a_clean_eda.py
├── part_b_regression.py
└── part_c_classification.py
```
