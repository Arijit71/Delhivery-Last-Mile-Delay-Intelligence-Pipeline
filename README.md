
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
