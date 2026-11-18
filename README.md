# 🏥 Medical Insurance Cost Predictor

A machine learning application that predicts a new client's **annual medical cost** based on demographics, health profile, and insurance plan selection. Designed to support a **frictionless 3-minute digital signup** experience for insurance firms.

---

## 📌 Business Problem

Insurance firms need to estimate how much a new client will cost them **before** issuing a policy, so they can price premiums accurately. This project builds a predictive model that uses only information available at sign-up time — no lab results, no claims history, no post-enrollment data.

**Goal:** Provide an instant, data-driven cost estimate to underwriters and customers during the application process.

---

## 🗂️ Dataset

- **Source:** `medical_insurance.csv`
- **Size:** ~thousands of historical policyholder records
- **Target:** `annual_medical_cost` (continuous, in USD)

---

## 🎯 Features Used (29 total)

Features were selected based on **what a client can reasonably provide in a 3-minute online signup**.

### 👤 Demographics (10)
| Feature | Type | Description |
|---|---|---|
| `age` | Numeric | Client age |
| `sex` | Categorical | Male / Female |
| `region` | Categorical | North, South, East, West, Central |
| `urban_rural` | Categorical | Urban, Suburban, Rural |
| `income` | Numeric | Annual income (USD) |
| `education` | Categorical | Highest education level |
| `marital_status` | Categorical | Single, Married, Divorced, Widowed |
| `employment_status` | Categorical | Employed, Self-employed, Unemployed, Retired |
| `household_size` | Numeric | Number of people in household |
| `dependents` | Numeric | Number of dependents |

### 🏃 Lifestyle (3)
| Feature | Type | Description |
|---|---|---|
| `bmi` | Numeric | Body Mass Index |
| `smoker` | Categorical | Yes / No / Former |
| `alcohol_freq` | Categorical | Never, Occasionally, Regularly, Daily |

### 🩺 Medical History (11)
| Feature | Type | Description |
|---|---|---|
| `hypertension` | Binary | Diagnosed with high blood pressure |
| `diabetes` | Binary | Diagnosed with diabetes |
| `asthma` | Binary | Diagnosed with asthma |
| `copd` | Binary | Diagnosed with COPD |
| `cardiovascular_disease` | Binary | Diagnosed with heart disease |
| `cancer_history` | Binary | History of cancer |
| `kidney_disease` | Binary | Diagnosed with kidney disease |
| `liver_disease` | Binary | Diagnosed with liver disease |
| `arthritis` | Binary | Diagnosed with arthritis |
| `mental_health` | Binary | Mental health condition |
| `hospitalizations_last_3yrs` | Numeric | Number of hospitalizations in last 3 years |

### 📋 Plan Selection (5)
| Feature | Type | Description |
|---|---|---|
| `plan_type` | Categorical | Bronze, Silver, Gold, Platinum |
| `network_tier` | Categorical | Basic, Standard, Premium |
| `deductible` | Numeric | Annual deductible (USD) |
| `copay` | Numeric | Per-visit copay (USD) |
| `policy_term_years` | Numeric | Length of policy (years) |

### 🚫 Features Intentionally Excluded
To prevent **data leakage** and reflect real-world sign-up conditions, the following were excluded:
- `person_id` — identifier, no predictive value
- `annual_medical_cost` — the target variable
- `annual_premium`, `monthly_premium` — pricing outputs (circular)
- `claims_count`, `avg_claim_amount`, `total_claims_paid` — post-enrollment data
- `risk_score`, `is_high_risk` — derived labels, likely leakage
- `systolic_bp`, `diastolic_bp`, `ldl`, `hba1c`, `medication_count` — require medical exam, not available in 3-minute signup
- `visits_last_year`, `days_hospitalized_last_3yrs`, `proc_*` — too granular for quick intake

---

## 🛠️ Methodology

### 1. Data Preprocessing
- Rows with missing `annual_medical_cost` (target) were **dropped** before splitting.
- Missing feature values were handled via **imputation inside a Pipeline**:
  - **Numeric features:** `SimpleImputer(strategy='median')`
  - **Categorical features:** `SimpleImputer(strategy='most_frequent')` → `OneHotEncoder(handle_unknown='ignore')`

### 2. Train / Validation Split
```python
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=1)
```
- 75% training / 25% validation
- Fixed `random_state` for reproducibility

### 3. Model Selection
**RandomForestRegressor** was chosen because:
- Handles mixed numeric/categorical data well
- Robust to outliers
- Non-linear relationships
- No feature scaling required
- Interpretable feature importance

### 4. Hyperparameter Tuning
Each hyperparameter was tuned **sequentially** by sweeping a grid and selecting the value that minimized validation MAE:

| Hyperparameter | Values Tested | Best Value | Best MAE |
|---|---|---|---|
| `max_leaf_nodes` | 2, 5, 10, 20, 30, 50, 100, 150, 200, 250, 300, 400, None | **100** | ~1833 |
| `min_samples_leaf` | 2, 3, 4, 5, 6, 7, 8, 9, 10 | **8** | ~1824 |
| `n_estimators` | 2 → 500 | **100** (diminishing returns after ~100) | ~1834 |
| `max_depth` | 2 → 500, None | **10** | ~1834 |

### 5. Final Model Configuration
```python
RandomForestRegressor(
    random_state=1,
    max_leaf_nodes=100,
    min_samples_leaf=8,
    max_depth=10,
    n_estimators=100
)
```

### 6. Pipeline Architecture
```
Pipeline
├── preprocessor (ColumnTransformer)
│   ├── num → SimpleImputer(median)
│   └── cat → SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown='ignore')
└── model → RandomForestRegressor
```

The entire pipeline is saved as a single `.joblib` artifact for production use.

---

## 📊 Performance

| Metric | Value | Interpretation |
|---|---|---|
| **MAE** | **$1,827** | On average, predictions are off by ~$1,827 per year |
| **MAPE** | **60.2%** | ~60% relative error (expected for high-variance cost data) |

### Performance Context
- For low-cost clients (e.g., $2,000/year), the error is proportionally larger.
- For high-cost clients (e.g., $50,000+/year), the error is proportionally smaller.
- The model is most useful for **ranking risk** and setting **baseline premiums**, not for exact dollar predictions.

---


> **🌐 Live Demo:** [Try the interactive web app here!](https://medicalinsurancecostearlypredictor-3c33nkjvyasymfuncdh2ha.streamlit.app/)
