
## Part A — Data Foundation and Delay Evidence

### Data Cleaning

The raw shipment dataset initially contained 1,224 rows.
After removing 24 duplicate records, 1,200 unique
shipments remained.

Missing values in distance, parcel weight and declared
value were filled using their respective medians.
Scanner-fault weight values above 100 kg were treated
as missing. Service names were standardised, and
shipment features were generated using hub information.

### Key Findings

**1. Delay by Service Type**

Express shipments have the highest delay rate of
0.4897 (48.97%), followed by Standard at 0.2056
(20.56%) and Economy at 0.0857 (8.57%).

**2. Delay by Hub Tier**

Delay rates increase with hub tier:
Tier 1 has a delay rate of 0.0400 (4%),
Tier 2 has 0.2150 (21.50%), and
Tier 3 has 0.4400 (44%).

**3. Delay by Distance**

Longer shipment distances are associated with
higher delay rates. The rates are:

- 0-200 km: 0.0219
- 201-400 km: 0.1368
- 401-600 km: 0.4364
- 601+ km: 0.8523

**4. Correlation with Delivery Hours**

The three features most strongly correlated with
delivery_hours are:

- promised_hours: +0.8372
- distance_km: +0.4013
- n_stops: +0.3247

**5. Declared Value Transformation**

The median-filled declared_value_inr column has
a mean of INR 1,533.0542 and a median of
INR 1,121.5000.

After natural log transformation, the mean is
7.0277 and the median is 7.0224.

The log transformation compresses large values,
reducing the influence of the right-skewed
declared-value distribution.

## Part B — Model Results (Regression)

### Data Splitting and Scaling

The 1,200 shipments were divided into 840 training, 180 validation and 180 test records using train_test_split with random_state=42.

StandardScaler was fitted only on the training data and then used to transform the test data. The maximum absolute column mean was 0.0000 for training data and 0.2187 for test data, confirming that the scaler was not fitted on the test set.

### Baseline Model

The mean baseline achieved an R² of -0.0212 on the validation set. This negative value indicates that predicting the training mean performs slightly worse than using the validation-set mean as the predictor. Therefore, the baseline provides a weak reference for comparison.

### Regression Model Comparison

Four regression models were compared using validation RMSE.

Linear Regression achieved the lowest validation RMSE of 4.8912 hours and an R² of 0.9313, making it the best model on the validation set.

Gradient Boosting performed closely, with an RMSE of 4.9165 hours and R² of 0.9306. Random Forest achieved an RMSE of 5.5095 hours, while the selected Decision Tree achieved 6.2796 hours.

### Decision Tree Depth Analysis

A depth of 1 shows high bias because both training and validation RMSE are high (11.6334 and 12.1324), with a small gap of 0.4990.

At depth 12 and without a depth limit, the model shows high variance. The training-validation gaps increase to 5.9927 and 7.8368 respectively, indicating overfitting.

Depth 6 provides the lowest validation RMSE of 6.2796 and is selected for the Decision Tree candidate.

### Gradient Boosting Learning Rate

A learning rate of 0.1 performs best, achieving a validation R² of 0.9306.

A learning rate of 0.01 is too slow to learn effectively with 100 estimators, resulting in an R² of 0.7127. A rate of 1.0 is too aggressive and overshoots, producing a lower R² of 0.8440.

### Cross-Validation

The two best validation models, Linear Regression and Gradient Boosting, were evaluated using five-fold cross-validation on the combined training and validation data.

Gradient Boosting achieved a mean R² of 0.9353 (std: 0.0045), while Linear Regression achieved 0.9261 (std: 0.0048).

Although cross-validation favours Gradient Boosting, Linear Regression is selected because it has the lowest validation RMSE, as required by the project. The difference between their validation RMSE values is small.

### Final Test Evaluation

Linear Regression was evaluated once on the previously untouched test set.

It achieved an RMSE of 4.5235 hours and an R² of 0.9493, demonstrating strong predictive performance on unseen shipments.