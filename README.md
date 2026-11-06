# Student Performance Analysis

Machine learning on the UCI Student Performance dataset (two Portuguese
schools). The project predicts a student's final grade (G3, 0-20) and
flags students at risk of failing (G3 below 10), for both Mathematics
and Portuguese.

Each task runs in two modes:

- **early** - uses only background and behavior data (no exam grades yet)
- **later** - uses period grades G1 and G2 as well

Four algorithms are compared for every task: Linear/Logistic Regression,
Decision Tree, Random Forest and XGBoost. The best models are saved and
explored in a Streamlit dashboard with SHAP explanations.

## Results

Best model per task, on held-out test data (see `reports/` for all four
models per task):

| Task | Best model | R2 | RMSE (out of 20) |
|---|---|---|---|
| Math grade, later | Random Forest | 0.83 | 1.89 |
| Portuguese grade, later | Linear Regression | 0.85 | 1.21 |
| Math grade, early | Random Forest | 0.31 | 3.78 |
| Portuguese grade, early | XGBoost | 0.17 | 2.84 |

| Risk flag (G3 < 10) | Best model | F1 | ROC-AUC |
|---|---|---|---|
| Mathematics, later | Logistic Regression | 0.89 | 0.98 |
| Portuguese, later | Logistic Regression | 0.69 | 0.96 |

The pattern in these numbers is itself a finding: past grades predict
final grades strongly, while background information alone is a much
weaker predictor in both subjects.

## Data

Download the Student Performance dataset from the UCI Machine Learning
Repository:

https://archive.ics.uci.edu/dataset/320/student+performance

Place the two CSV files here (this folder is git-ignored):

```
data/raw/school/student-mat.csv
data/raw/school/student-por.csv
```

## Setup

Python 3.11 or newer recommended.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run order

```bash
# 1. Download the data (above)

# 2. Train and evaluate all models (writes reports/*.csv and saves models)
python -m src.models.train_regression
python -m src.models.train_classification

# 3. Generate the evaluation charts (writes reports/figures/)
python -m src.models.generate_evaluation

# 4. Open the dashboard
streamlit run dashboard/app.py

# 5. Run the tests (27 tests)
pytest
```

## Project structure

```
src/
  data/           dataset loading and preprocessing
  features/       feature engineering (early / later modes)
  models/         training, evaluation, prediction
  explainability/ SHAP analysis
dashboard/        Streamlit app (6 pages)
tests/            pytest suite
reports/          metrics CSVs and figures
```

## A note on responsible use

The early mode leans on sensitive background attributes (family
situation, parents' education, alcohol use) to label students "at risk".
This is a learning project on a public research dataset. Its predictions
must never be used to judge or make decisions about real students.
