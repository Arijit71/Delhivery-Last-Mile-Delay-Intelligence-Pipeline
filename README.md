# Delhivery — Last-Mile Delay Intelligence

A machine learning capstone project focused on predicting delivery times, identifying delayed shipments and grouping logistics hubs according to their operating performance using synthetic shipment data.

**Program:** AI and ML Certification Program — E&ICT Academy, IIT Roorkee

---

# How to run

Clone the repository:

```bash
git clone https://github.com/<your-username>/delhivery-last-mile-delay-intelligence.git
cd delhivery-last-mile-delay-intelligence
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the scripts in the following order:

```bash
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

`package_model.py` regenerates `models/delay_model.pkl` using the final configuration from Part C. The committed pickle loads reliably only with the scikit-learn version pinned in `requirements.txt`. If the installed version differs, the grader re-runs `package_model.py` instead of loading the committed pickle.

---

# Part A — Data Foundation and Delay Evidence

## Data cleaning

The raw dataset initially contained 1,224 shipment records. After removing 24 duplicates, 1,200 unique shipments remained.

| Metric | Result |
|---|---:|
| Initial rows | 1,224 |
| Duplicates removed | 24 |
| Final rows | 1,200 |
| Missing distance values | 36 |
| Missing parcel weights | 24 |
| Scanner-fault weights | 5 |
| Missing declared values | 12 |
| Missing payment modes | 18 |
| Service spellings before cleaning | 12 |
| Service spellings after cleaning | 3 |

Missing values in distance, parcel weight and declared value were filled using their respective medians. Weight values above 100 kg were treated as scanner errors, and service names were standardised.

**Median values used for imputation:**

| Feature | Median |
|---|---:|
| distance_km | 314.95 |
| parcel_weight_kg | 1.83 |
| declared_value_inr | INR 1,121.50 |

## Delay analysis

**By service type**

| Service | Shipments | Delayed | Delay Rate |
|---|---:|---:|---:|
| Economy | 315 | 27 | 8.57% |
| Express | 243 | 119 | 48.97% |
| Standard | 642 | 132 | 20.56% |

Express shipments have the highest delay rate.

**By hub tier**

| Tier | Shipments | Delayed | Delay Rate |
|---|---:|---:|---:|
| 1 | 400 | 16 | 4.00% |
| 2 | 400 | 86 | 21.50% |
| 3 | 400 | 176 | 44.00% |

Delay rates increase from Tier 1 to Tier 3.

**By distance**

| Distance Band | Shipments | Delayed | Delay Rate |
|---|---:|---:|---:|
| 0–200 km | 274 | 6 | 2.19% |
| 201–400 km | 563 | 77 | 13.68% |
| 401–600 km | 275 | 120 | 43.64% |
| 601+ km | 88 | 75 | 85.23% |

Longer shipment distances are associated with higher delay rates.

## Correlation analysis

The three features most strongly correlated with `delivery_hours`, ranked by absolute correlation, are:

| Feature | Correlation |
|---|---:|
| promised_hours | +0.8372 |
| distance_km | +0.4013 |
| n_stops | +0.3247 |

## Declared value transformation

| Statistic | Original Value (INR) | Log-Transformed Value |
|---|---:|---:|
| Mean | 1,533.05 | 7.0277 |
| Median | 1,121.50 | 7.0224 |

The declared value distribution has a higher mean than median. Applying the natural log reduces the influence of large values and brings the mean and median closer together.

---

# Part B — Delivery-Hours Regression

## Data splitting and scaling

The 1,200 shipments were divided into training, validation and test sets.

| Dataset | Rows |
|---|---:|
| Training | 840 |
| Validation | 180 |
| Test | 180 |

`StandardScaler` was fitted only on the training data and then used to transform the test data.

| Scaling Check | Result |
|---|---:|
| Training maximum absolute column mean | 0.0000 |
| Test maximum absolute column mean | 0.2187 |

## Model comparison

The mean baseline achieved an R² of -0.0212 on validation data. This means that predicting the training mean performs slightly worse than using the validation-set mean.

**Validation results:**

| Model | RMSE (Hours) | R² |
|---|---:|---:|
| Mean Baseline | 18.8552 | -0.0212 |
| Linear Regression | **4.8912** | **0.9313** |
| Decision Tree | 6.2796 | 0.8867 |
| Random Forest | 5.5095 | 0.9128 |
| Gradient Boosting | 4.9165 | 0.9306 |

Linear Regression achieved the lowest validation RMSE of **4.8912 hours**.

## Decision tree depth

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

- **High bias:** Depth 1 has high training and validation errors, with a gap of 0.4990.
- **High variance:** Depth 12 and unlimited depth have gaps of 5.9927 and 7.8368 respectively, indicating overfitting.
- **Selected depth:** 6, with the lowest validation RMSE.

## Gradient boosting learning rate

| Learning Rate | Validation RMSE | Validation R² |
|---:|---:|---:|
| 0.01 | 10.0018 | 0.7127 |
| 0.1 | **4.9165** | **0.9306** |
| 1.0 | 7.3701 | 0.8440 |

A learning rate of 0.1 performs best. With 100 estimators, 0.01 learns too slowly, whereas 1.0 is too aggressive and overshoots.

## Cross-validation

The two best models from validation were compared using five-fold cross-validation on the combined training and validation data.

| Model | Mean R² | Standard Deviation |
|---|---:|---:|
| Linear Regression | 0.9261 | 0.0048 |
| Gradient Boosting | 0.9353 | 0.0045 |

Gradient Boosting performs better in cross-validation, but Linear Regression is selected because it has the lower validation RMSE, as required by the assignment.

## Final test evaluation

| Model | Test RMSE | Test R² |
|---|---:|---:|
| Linear Regression | 4.5235 | 0.9493 |

---

# Part C — Delay Classification and Alert Threshold

## Stratified data splitting

| Dataset | Total Rows | Delayed Shipments |
|---|---:|---:|
| Training | 840 | 195 |
| Validation | 180 | 42 |
| Test | 180 | 41 |

## Baseline and classifier comparison

The blind baseline predicts every shipment as on time. It achieves 0.7667 accuracy, but its precision and recall are both zero because it misses all 42 delayed validation shipments.

**Validation results using the default 0.5 threshold:**

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Blind Baseline | 0.7667 | 0.0000 | 0.0000 | 0.0000 |
| Logistic Regression | 0.9056 | 0.7778 | 0.8333 | **0.8046** |
| Decision Tree (Depth 6) | 0.8833 | 0.7561 | 0.7381 | 0.7470 |
| Random Forest | 0.9056 | 0.8378 | 0.7381 | 0.7848 |

Logistic Regression achieved the highest F1 score among the initial classifier comparisons.

## Feature selection and noise analysis

The five bookkeeping columns were treated as noise features. Together, they account for **0.2245 (22.45%)** of the Random Forest feature importance. All five have F-scores below 10 and rank outside the top seven features.

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

Using seven features gives the highest validation F1, so `k=7` is used in the remaining experiments.

## Hyperparameter tuning

| Search Method | Candidates | Total Fits | Best CV Accuracy | Validation F1 |
|---|---:|---:|---:|---:|
| Grid Search | 27 | 135 | 0.8952 | 0.7750 |
| Randomized Search | 9 | 45 | 0.8905 | 0.7848 |
| Default Random Forest | 1 | 1 | — | **0.8333** |

The default Random Forest performs best on validation, so it is selected instead of the tuned models.

Tuning did not improve validation F1 in this case. The searches use accuracy as their default scoring metric, whereas the final selection is based on F1. The validation set contains only 180 shipments, so its results can also be affected by a small number of predictions.

## Alert threshold selection

The highest threshold that maintains validation recall of at least 0.80 is selected.

| Threshold | Validation Precision | Validation Recall | Validation F1 |
|---:|---:|---:|---:|
| **0.5** | 0.7955 | 0.8333 | 0.8140 |
| 0.6 | 0.8889 | 0.7619 | 0.8205 |

The selected threshold is **0.5**. Although 0.6 gives higher precision, its recall falls below the required 0.80.

## Final test evaluation

The selected Random Forest was evaluated once on the untouched test set using the threshold of 0.5.

| Metric | Result |
|---|---:|
| Accuracy | 0.8667 |
| Precision | 0.6889 |
| Recall | 0.7561 |
| F1 | 0.7209 |
| True Positives | 31 |
| False Positives | 14 |
| False Negatives | 10 |
| True Negatives | 125 |

Test recall is 0.7561 compared with validation recall of 0.8333, a decrease of 7.72 percentage points. The test set contains only 41 delayed shipments.

---

# Part D — Hub Segmentation, Packaging and Operations Playbook

## Hub segmentation

The 24 hubs were grouped into three segments using K-Means on scaled delivery hours, delay rate, distance and number of stops.

| Segment | Hubs | Mean Delivery Hours | Delay Rate | Mean Distance (km) | Mean Stops |
|---|---:|---:|---:|---:|---:|
| Standard | 7 | 47.2396 | 2.57% | 188.9296 | 55.3686 |
| Watch | 8 | 56.8137 | 15.50% | 314.1762 | 70.7175 |
| Escalate | 9 | 64.6504 | 46.00% | 470.4757 | 84.3444 |

## Hubs in the Escalate segment

| Hub | City | Existing Tier | Delay Rate |
|---|---|---:|---:|
| HUB13 | Nagpur | 2 | 46% |
| HUB15 | Kochi | 2 | 36% |
| HUB17 | Patna | 3 | 46% |
| HUB18 | Ranchi | 3 | 38% |
| HUB19 | Guwahati | 3 | 62% |
| HUB20 | Raipur | 3 | 48% |
| HUB21 | Bhubaneswar | 3 | 40% |
| HUB23 | Siliguri | 3 | 44% |
| HUB24 | Agartala | 3 | 54% |

These nine hubs have the highest operating delay rates and should receive greater attention. Flagged shipments from these hubs should be escalated to their hub managers.

## Tier and segment mismatches

| Hub | City | Existing Tier | Actual Segment | Tier-Implied Segment |
|---|---|---:|---|---|
| HUB07 | Kolkata | 1 | Watch | Standard |
| HUB13 | Nagpur | 2 | Escalate | Watch |
| HUB15 | Kochi | 2 | Escalate | Watch |
| HUB22 | Dehradun | 3 | Watch | Escalate |

These hubs do not follow the operating pattern suggested by their existing tier labels. The tier can still be used as a reference, but the segment should guide the operational playbook.

## Alert policy

The final classifier uses a probability threshold of **0.5**.

Based on test results:

- About **6.9 out of every 10 flagged parcels** are actually delayed (precision: 0.6889).
- About **7.6 out of every 10 delayed parcels** are identified (recall: 0.7561).

## New shipment actions

The prediction script processed all 48 new shipments.

| Action | Shipments |
|---|---:|
| Escalate to hub manager | 6 |
| Proactive customer call | 2 |
| No action | 40 |
| **Total** | **48** |

Eight parcels were flagged, and six of them require hub-manager escalation.

## Selection of K = 3

| K | Across / Within Ratio | Cluster Sizes |
|---:|---:|---|
| 2 | 2.8940 | 15-9 |
| 3 | 5.3928 | 9-8-7 |
| 4 | 6.3047 | 8-7-5-4 |
| 5 | 6.8866 | 8-7-4-3-2 |

The separation ratio increases with K, but K=4 and K=5 produce clusters containing fewer than six hubs. K=3 gives three groups of 7, 8 and 9 hubs, making them more suitable for maintaining three separate operational playbooks.

## Model packaging and prediction

The final Random Forest classifier was saved as `models/delay_model.pkl`, along with its threshold, feature order and historical imputation values.

The saved model was reloaded and checked against the original model using the 180 test shipments.

| Check | Result |
|---|---:|
| Test rows scored | 180 |
| Maximum probability difference | 0.0 |
| Alert threshold | 0.5 |

The prediction script uses the shared `clean_shipments()` and `build_features()` functions, along with the stored historical medians and feature order, to process new shipments.

---

# Production plan

## Production readiness

| Characteristic | Implementation in this project |
|---|---|
| Accuracy | Regression and classification models were evaluated on separate test data. |
| Reliability | The saved model was reloaded successfully, with a maximum probability difference of 0.0. |
| Reproducibility | Fixed random states, the provided data generator and pinned package versions are used. |
| Performance (speed) | Batch prediction uses seven selected features and a 100-estimator Random Forest. No formal runtime benchmark was performed. |
| Maintainability | Separate scripts handle data preparation, regression, classification, segmentation, packaging and prediction. |
| Consistency | The same `clean_shipments()` and `build_features()` functions, historical medians and feature order are reused during prediction. |
| Monitorability | `new_shipment_actions.csv` records shipment probabilities, flags, hub segments and recommended actions. |

## Monitoring

The following are proposed starting thresholds for operational monitoring, not measured results from the synthetic dataset.

| Quantity | Trigger for review |
|---|---|
| Weekly realised delay rate | Above 28% for two consecutive comparable non-sale weeks. |
| Weekly flagged shipment share | Above 30% for two consecutive weeks. |
| Recall on shipments with known outcomes | Below 0.70 over a rolling four-week period. |

A trigger should lead to an investigation and model review.

## Retraining and data drift

- **Retraining schedule:** Retrain monthly using recent labelled shipment data and evaluate the updated model on a later holdout set.
- **Sale events:** The 15–28 September sale window should be compared with similar sale-event periods rather than ordinary weeks, since shipment volumes and delay patterns may differ.
- **Data drift:** Changes in distance, number of stops, service-type distribution, parcel weights and hub-level delay rates should be monitored. Significant changes in these distributions may indicate that current shipments differ from the training data.

---

# Limitations

1. **Synthetic data:** The dataset is generated for this project and does not fully represent the conditions of a real logistics network.

2. **Small test set:** The classification test set contains only 41 delayed shipments, so recall can change noticeably with a small number of different predictions.

3. **Threshold selection:** The alert threshold was selected using one validation split. Its performance may change with a different shipment mix or operating conditions.

4. **Hub segmentation:** The segmentation uses only 24 hubs and four aggregated operational features. Other factors that affect hub performance are not included.