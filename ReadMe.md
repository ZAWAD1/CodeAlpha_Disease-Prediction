## About

A machine learning pipeline that predicts whether a patient is likely to have diabetes based on medical measurements. Trained on the Pima Indians Diabetes dataset (UCI / Kaggle) — 768 female patients of Pima Indian heritage. Uses Logistic Regression selected after comparing 5 models with 5-fold cross validation. Includes a Streamlit dashboard for live patient risk assessment.

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-red) ![Logistic Regression](https://img.shields.io/badge/Model-Logistic%20Regression-green)

---

## Project Structure

```
disease-prediction/
├── diabetes_dashboard.py          # Streamlit web app
├── notebooks/
│   └── diabetes_prediction.ipynb  # Full EDA + model training
├── Model/
│   ├── diabetes_model.pkl          # Saved Logistic Regression model
│   └── diabetes_scaler.pkl         # Saved StandardScaler
├── data/
│   └── diabetes.csv                # Pima Indians dataset
└── README.md
```

---

## Dataset

**Pima Indians Diabetes Database** — UCI Machine Learning Repository / Kaggle

| Column | Plain English | Normal Range |
|---|---|---|
| `Pregnancies` | Number of times pregnant | 0–17 |
| `Glucose` | Plasma glucose (mg/dL) — 2hr oral test | 70–140 |
| `BloodPressure` | Diastolic blood pressure (mmHg) | 60–80 |
| `SkinThickness` | Triceps skin fold thickness (mm) | 10–40 |
| `Insulin` | 2-hour serum insulin (µU/mL) | 2–25 (fasting) |
| `BMI` | Body mass index (kg/m²) | 18.5–24.9 healthy |
| `DiabetesPedigreeFunction` | Genetic diabetes likelihood score | 0.078–2.42 |
| `Age` | Age in years | 21–81 |
| `Outcome` *(target)* | Diabetic? (1=yes, 0=no) | Binary |

**Class distribution:** 65% no diabetes · 35% diabetes

---

## Pipeline

### 1. Exploratory Data Analysis
- Class distribution — 65/35 split, less severe than typical credit datasets
- Hidden zeros detected — Glucose, BloodPressure, SkinThickness, Insulin, BMI all had impossible zero values disguised as missing data
- Correlation analysis — Glucose (0.49) is the strongest predictor, followed by BMI (0.31) and Age (0.24)
- Outlier detection — Insulin had values up to 846 (normal fasting under 200)

### 2. Preprocessing
- Hidden zeros replaced with `np.nan` then filled with column median
- Outlier clipping at 99th percentile for Insulin, BMI, SkinThickness
- `StandardScaler` applied for Logistic Regression and SVM

### 3. Feature Engineering

| Feature | Formula | Reason |
|---|---|---|
| `glucose_bmi` | `Glucose × BMI` | High glucose + high BMI together is a stronger diabetes signal than either alone |
| `age_risk` | `1 if Age >= 45 else 0` | Clinical evidence shows diabetes risk increases significantly after 45 |

### 4. Model Comparison

| Model | F1 (Diabetes) | Single AUC | CV Mean AUC | CV Std |
|---|---|---|---|---|
| **Logistic Regression** ✅ | **0.67** | **0.814** | **0.838** | **0.027** |
| SVM | 0.66 | 0.810 | 0.831 | 0.032 |
| Random Forest | 0.55 | 0.815 | 0.824 | 0.039 |
| XGBoost | 0.66 | 0.800 | 0.792 | 0.035 |
| Decision Tree | 0.61 | 0.752 | 0.779 | 0.032 |

**Logistic Regression won** — highest CV mean AUC (0.838), lowest std (0.027), best F1 score. On this small 768-row dataset with mostly linear relationships, Logistic Regression outperforms complex ensemble methods.

### 5. Cross Validation
5-fold cross validation was used to confirm model selection. Random Forest appeared strongest on the single split (0.815) but dropped to third place in CV (0.824) with the highest instability — revealing the single split result was partly luck. Logistic Regression's consistent 0.838 across all five folds confirmed it as the genuine winner.

---

## Installation & Usage

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn xgboost streamlit joblib matplotlib seaborn
```

### 2. Train the model
```bash
jupyter notebook notebooks/diabetes_prediction.ipynb
```
Run all cells top to bottom. Models save to `Model/`.

### 3. Run the dashboard
```bash
streamlit run diabetes_dashboard.py
```
Opens at `http://localhost:8501`

---

## Making a Prediction

-**For Demo run code below can be used, for the live prediction diabetes_dashboard.py should be used**

```python
import joblib
import pandas as pd

model  = joblib.load('Model/diabetes_model.pkl')
scaler = joblib.load('Model/diabetes_scaler.pkl')

patient = pd.DataFrame([{
    'Pregnancies': 2,
    'Glucose': 110,
    'BloodPressure': 72,
    'SkinThickness': 23,
    'Insulin': 80,
    'BMI': 28.0,
    'DiabetesPedigreeFunction': 0.35,
    'Age': 30,
    'glucose_bmi': 110 * 28.0,   # engineered feature
    'age_risk': 0                 # 0 = under 45
}])

probability = model.predict_proba(scaler.transform(patient))[0][1]
print(f"Diabetes probability: {probability * 100:.1f}%")
```

---

## Key Findings

- **Glucose is the #1 predictor** — correlation of 0.49 with diabetes outcome, far above any other feature
- **Cross validation is essential on small datasets** — single split results were misleading for Random Forest; CV revealed the true winner
- **Simple models beat complex ones on small data** — Logistic Regression outperformed XGBoost and Random Forest on 768 rows because the relationships are mostly linear
- **Hidden zeros are a unique challenge in medical data** — `isnull().sum()` showed 0 missing values, masking 374 rows of missing insulin data recorded as zero

---